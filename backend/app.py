import os
import sys

# Optimize memory before loading TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import gc
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import cv2
import base64

app = Flask(__name__)
# Enable CORS so your local frontend can securely call this API
CORS(app)

# Load the fine-tuned VGG16 model using an absolute file path descriptor
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'best_vgg16.keras')
model = tf.keras.models.load_model(MODEL_PATH)
CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']

def make_gradcam_heatmap(img_array, model):
    # Locate the base backbone layer (VGG16 model instantiation instance)
    backbone = None
    for layer in model.layers:
        if 'vgg' in layer.name.lower() or isinstance(layer, tf.keras.models.Model):
            backbone = layer
            break
            
    # Fallback directly to sequential parent lookup if features are unnested
    if backbone is None:
        backbone = model

    # Isolate the absolute final Conv2D activation block within the VGG backbone
    last_conv_layer = None
    for layer in backbone.layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer = layer

    # Reconstruct a dual-output computational graph for gradient inspection
    grad_model = tf.keras.models.Model(
        inputs=[backbone.inputs],
        outputs=[last_conv_layer.output, backbone.output]
    )

    # Contextually record forward activations to evaluate backpropagated weights
    with tf.GradientTape() as tape:
        conv_outputs, backbone_out = grad_model(img_array)
        
        # Route the intermediate feature tensor through your custom classification head
        x = backbone_out
        for layer in model.layers[model.layers.index(backbone) + 1:]:
            x = layer(x)
            
        predictions = x
        pred_class = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_class]

    # Compute spatial map gradients and pool neuron tracking weight matrices
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    
    return heatmap.numpy(), int(pred_class), float(tf.reduce_max(predictions))

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
        
    file = request.files['image']
    
    try:
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # Standardize dimensional scale matrix to match VGG16 input specifications (128x128)
        img_resized = cv2.resize(img_rgb, (128, 128))

        # Apply feature normalization parameters matching the training profile
        img_array = tf.cast(img_resized, tf.float32) / 255.0
        img_array = (img_array - 0.5) / 0.5
        img_array = tf.expand_dims(img_array, axis=0)

        # Trigger inference execution and extract localization maps
        heatmap, pred_class_idx, confidence_score = make_gradcam_heatmap(img_array, model)

        # Process activation heatmap scales and format colormaps
        heatmap_resized = cv2.resize(heatmap, (128, 128))
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        
        # Apply alpha blending: Overlay high-contrast JET map onto original MRI slice
        superimposed = cv2.addWeighted(img_resized, 0.6, cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB), 0.4, 0)
        superimposed_bgr = cv2.cvtColor(superimposed, cv2.COLOR_RGB2BGR)
        
        # Serialize the combined output canvas to Base64 format for HTTP payload delivery
        _, img_encoded = cv2.imencode('.jpg', superimposed_bgr)
        base64_string = base64.b64encode(img_encoded).decode('utf-8')
        
        # Clear tensor components from memory
        del img_bgr, img_rgb, img_resized, img_array, heatmap, heatmap_resized, heatmap_uint8, heatmap_color, superimposed
        gc.collect()
        
        return jsonify({
            'class': CLASSES[pred_class_idx].upper(),
            'confidence': f"{confidence_score * 100:.2f}%",
            'heatmap_image': f"data:image/jpeg;base64,{base64_string}"
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
