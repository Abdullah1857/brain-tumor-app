const panelRight = document.getElementById('panelRight');
const imageUpload = document.getElementById('imageUpload');
const previewContainer = document.getElementById('previewContainer');
const imagePreview = document.getElementById('imagePreview');
const predictBtn = document.getElementById('predictBtn');
const resultSection = document.getElementById('resultSection');
const predictionText = document.getElementById('predictionText');
const confidenceText = document.getElementById('confidenceText');
const fileName = document.getElementById('fileName');

// Progress Bar Targets
const progressContainer = document.getElementById('progressContainer');
const progressBarFill = document.getElementById('progressBarFill');
const progressStatus = document.getElementById('progressStatus');

const heatmapImg = document.createElement('img');
heatmapImg.id = 'heatmapResult';

// 1. Reset state and update file information instantly on image load
imageUpload.addEventListener('change', function() {
    const file = this.files[0];
    if (file) {
        fileName.innerText = `Selected: ${file.name}`;
        
        const reader = new FileReader();
        reader.onload = function(e) {
            imagePreview.src = e.target.result;
            previewContainer.classList.remove('hidden');
            predictBtn.classList.remove('hidden');
            
            // Clear right side panels layout structures instantly with zero shifts
            panelRight.classList.add('hidden');
            resultSection.classList.add('hidden');
            progressContainer.classList.add('hidden');
        }
        reader.readAsDataURL(file);
    } else {
        fileName.innerText = "No file chosen";
    }
});

// 2. Perform Local API Query & Manage Displays Instantly
predictBtn.addEventListener('click', async () => {
    const file = imageUpload.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('image', file);

    // Show the right panel layout structures instantly with zero shifting animation
    panelRight.classList.remove('hidden');
    progressContainer.classList.remove('hidden');
    resultSection.classList.add('hidden');
    predictBtn.disabled = true;
    
    progressBarFill.style.width = "25%";
    progressStatus.innerText = "Transmitting image data array...";

    setTimeout(() => {
        if(predictBtn.disabled) {
            progressBarFill.style.width = "60%"
            progressStatus.innerText = "Processing feature map evaluations...";
        }
    }, 500);

    setTimeout(() => {
        if(predictBtn.disabled) {
            progressBarFill.style.width = "85%";
            progressStatus.innerText = "Extracting deep layer weights via GradCAM...";
        }
    }, 1200);

    try {
        // FIXED: Pointing explicitly to your localhost Flask instance
        const response = await fetch('http://127.0.0.1:5000/predict', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (response.ok) {
            progressBarFill.style.width = "100%";
            progressStatus.innerText = "Success! Compiling metrics.";

            setTimeout(() => {
                progressContainer.classList.add('hidden');
                
                // Print clinical diagnostics data parameters
                predictionText.innerHTML = `Diagnosis: <span class="result-badge">${data.class}</span>`;
                confidenceText.innerText = `Confidence: ${data.confidence}`;
                
                // FIXED: Check to overwrite instead of piling multiple image objects
                let existingHeatmap = document.getElementById('heatmapResult');
                if (!existingHeatmap) {
                    heatmapImg.src = data.heatmap_image;
                    resultSection.appendChild(heatmapImg);
                } else {
                    existingHeatmap.src = data.heatmap_image;
                }
                
                resultSection.classList.remove('hidden');
            }, 400);

        } else {
            progressContainer.classList.add('hidden');
            resultSection.classList.remove('hidden');
            predictionText.innerText = `Error: ${data.error}`;
        }
    } catch (error) {
        progressContainer.classList.add('hidden');
        resultSection.classList.remove('hidden');
        predictionText.innerText = "Server communication failure.";
        console.error(error);
    } finally {
        predictBtn.disabled = false;
    }
});
