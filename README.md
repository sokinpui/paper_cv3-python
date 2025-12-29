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

### Command Line Interface (CLI)

For batch processing or automation.

```bash
python main.py path/to/image.jpg \
    --height 200 --width 200 \
    --metric ssim \
    --algorithm global \
    --top_n 5 \
    --output result.png
```

#### CLI Options:

- `--height`, `--width`: Dimensions of the tiles.
- `--overlap`: Ratio [0.0 - 0.9] between tiles.
- `--metric`: `ssim`, `cielab`, `moments`, `texture`, `grad_color`, `hist`.
- `--algorithm`: `global` or `local`.
- `--sort_by`: `mean`, `median`, `std_dev`, `max_score`.
- `--clahe`: Set local contrast enhancement limit (e.g., 2.0).
- `--sharpen`: Set sharpening strength (e.g., 1.0).

## Metrics Overview

| Metric              | Best For                                                       |
| :------------------ | :------------------------------------------------------------- |
| **SSIM**            | Structural changes, repeating patterns, and fabric alignment.  |
| **Texture & Color** | Surface defects, scratches, and stains on uniform backgrounds. |
| **CIELab Delta E**  | Precise color distance and perceived color shifts.             |
| **Histograms**      | Material distribution and global color composition.            |
| **Lab Moments**     | Quick detection of brightness and color variance.              |

## Project Structure

- `app.py`: Gradio Web UI implementation.
- `main.py`: CLI entry point.
- `analyzer.py`: Logic for Global and Local detection algorithms.
- `metrics.py`: GPU-accelerated similarity and distance strategies.
- `processor.py`: Image tiling, preprocessing, and visualization tools.

---
