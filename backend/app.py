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
# Enable CORS so your Hugging Face frontend can securely call this Render API
CORS(app)

# Load the trained model cleanly
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'best_model.keras')
model = tf.keras.models.load_model(MODEL_PATH)
CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']

def make_gradcam_heatmap(img_array, model):
    mobilenet = model.get_layer('mobilenetv2_1.00_224')
    
    last_conv_layer = None
    for layer in mobilenet.layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer = layer

    grad_model = tf.keras.models.Model(
        inputs=[mobilenet.inputs],
        outputs=[last_conv_layer.output, mobilenet.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, mobilenet_out = grad_model(img_array)
        x = model.get_layer('global_average_pooling2d')(mobilenet_out)
        x = model.get_layer('batch_normalization')(x)
        x = model.get_layer('dense')(x)
        x = model.get_layer('dropout')(x)
        x = model.get_layer('dense_1')(x)
        x = model.get_layer('dropout_1')(x)
        predictions = model.get_layer('dense_2')(x)
        pred_class = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_class]

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
        img_resized = cv2.resize(img_rgb, (224, 224))

        img_array = tf.cast(img_resized, tf.float32) / 255.0
        img_array = (img_array - 0.5) / 0.5
        img_array = tf.expand_dims(img_array, axis=0)

        heatmap, pred_class_idx, confidence_score = make_gradcam_heatmap(img_array, model)

        heatmap_resized = cv2.resize(heatmap, (224, 224))
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        
        superimposed = cv2.addWeighted(img_resized, 0.6, cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB), 0.4, 0)
        superimposed_bgr = cv2.cvtColor(superimposed, cv2.COLOR_RGB2BGR)
        
        _, img_encoded = cv2.imencode('.jpg', superimposed_bgr)
        base64_string = base64.b64encode(img_encoded).decode('utf-8')
        
        # Memory Cleanup: Force release RAM allocations right away
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
    # Render tells apps which port to use via the PORT environment variable
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
