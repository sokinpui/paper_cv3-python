### Prerequisites

- NVIDIA GPU + CUDA Toolkit
- Python 3.8+

### Setup

Start the virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate
```

```bash
pip install -r requirements.txt
```

## Usage

### Web Interface

Launch the interactive dashboard to upload images, adjust preprocessing in real-time, and visualize heatmaps.

```bash
# Start the web server
python app.py

# Start with a specific image directory for quick selection
python app.py -i ./my_samples/
```

---

Please read [doc.pdf](./doc.pdf) for details explanation for the following:

```
## 1. Oklab Color Space (Perceptual Distance)
### Mathematical Implementation
### Distance Calculation
### References
## 2. SSIM (Structural Similarity Index)
### Mathematical Implementation
### References
## 3. Cosine Distance
## 4. DBSCAN Clustering
### Algorithm Logic
### Auto-Epsilon (K-Distance Graph)
### References
## 5. Distance Transformations
### Sigmoid Contrast Stretch
### Power Transformation
```
