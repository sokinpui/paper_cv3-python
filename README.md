### Prerequisites

- NVIDIA GPU + CUDA Toolkit
- [uv](https://github.com/astral-sh/uv) (Recommended) or Python 3.12+

### Setup

Create the environment and install dependencies. To ensure GPU support, we point to the PyTorch CUDA wheels:

```bash
uv venv
source .venv/bin/activate
```

```bash
# Install the package and its dependencies
uv pip install --find-links https://download.pytorch.org/whl/cu121 -e .
```

### Network Optimization

If downloading packages is slow, use `tpip` to automatically find and set the fastest PyPI mirror:

```bash
# Install tpip
pip install tpip

# Test mirrors and set the best one
tpip set
```

Note: If you are using **uv**, it will respect your global pip configuration. Alternatively, you can specify an index explicitly: `uv pip install --index-url <MIRROR_URL> ...`

To permanently set a mirror for this project using **uv**, add the following to your `pyproject.toml`:

```toml
[[tool.uv.index]]
name = "tsinghua"
url = "https://pypi.tuna.tsinghua.edu.cn/simple"
default = true
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
```
