# GPU Image Unit Detection

A highly-optimized tool for anomaly and defect detection in large images by comparing image units (patches) using CUDA-accelerated pairwise distance metrics.

## Features

- **CUDA Acceleration:** Fast computation of large similarity/distance matrices.
- **Multiple Metrics:** Supports perceptual (Oklab, Gradient, etc.) and traditional metrics.
- **Web UI & CLI:** Interactive web interface and command-line tool for batch processing.
- **Clustering:** Supports K-Means, Hierarchical, Spectral, and DBSCAN clustering on distance matrices or unit stats.

## Setup

The project requires Python and an NVIDIA GPU with CUDA.

```bash
# Assuming you have a clean environment
pip install -r requirements.txt
```

## Web UI Usage (app.py)

Run the interactive web interface to load an image, adjust unit size, select a metric, and visualize results (heatmap, top-N, or clustering).

```bash
# Run the UI, optionally pointing to a directory for image examples
python app.py -i ./sample_images/
```

Access the UI at `http://localhost:7860`.

## Command Line Usage (main.py)

Run the command-line tool for automated processing and JSON output.

```bash
python main.py <image_path> \
 --height 50 \
 --width 50 \
 --metric human_eye \
 --top_n 5 \
 --sort_by l2_norm \
 --power_transform 0.4 \
 --output annotated_result.png
```

### Example Arguments

| Argument | Description | Default |
|---|---|---|
| `image_path` | Path to the input image. | - |
| `--height` | Height of the analysis unit. | **Required** |
| `--width` | Width of the analysis unit. | **Required** |
| `--metric` | Comparison metric (e.g., `human_eye`, `grad_color`, `ssim`). | `human_eye` |
| `--sort_by` | Statistic to rank units (e.g., `mean`, `l2_norm`). | `mean` |
| `--top_n` | Number of top units to report. | 5 |
| `--output` | Path to save the annotated image. | - |
