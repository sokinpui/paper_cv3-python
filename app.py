import argparse
import glob
import json
import os
import sys
import time

import gradio as gr
import torch

from analyzer import PatchAnalyzer
from metrics import CIELabMetric, SSIMMetric
from processor import ImageProcessor

# 1. Initialize CUDA Device once
try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"Web UI running on: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}"
    )
except Exception as e:
    print(f"Warning: {e}")
    device = torch.device("cpu")


def run_analysis(image_path, height, width, metric_name, top_n, sort_by, descending):
    """
    The core function called when user clicks 'Run Detection'
    """
    if image_path is None:
        return None, "Please upload an image.", ""

    try:
        # Setup Components
        processor = ImageProcessor(device)

        if metric_name == "SSIM (Structure)":
            metric = SSIMMetric()
        else:
            metric = CIELabMetric()

        analyzer = PatchAnalyzer(metric)

        # Pipeline
        # 1. Load
        image_tensor = processor.load_image(image_path)

        # 2. Tile
        patches, grid_shape = processor.extract_patches(
            image_tensor, int(height), int(width)
        )

        # 3. Analyze
        # 'descending' comes from UI Checkbox (True/False)
        # Note: Logic in main.py was 'ascending = not descending'.

        t0 = time.time()
        stats = analyzer.analyze(
            patches,
            grid_shape,
            top_n=int(top_n),
            sort_by=sort_by,
            ascending=not descending,
        )
        t1 = time.time()

        # Performance Stats
        duration = t1 - t0
        N = patches.shape[0]
        total_pairs = N * N
        cps = total_pairs / duration if duration > 0 else 0

        perf_text = (
            f"### ⚡ Performance Metrics\n"
            f"- **Analysis Time:** {duration:.4f} seconds\n"
            f"- **Total Units:** {N} (Grid: {grid_shape})\n"
            f"- **Total Comparisons:** {total_pairs:,}\n"
            f"- **Throughput:** {cps:,.0f} comparisons/sec"
        )

        # 4. Visualize
        result_image = processor.get_annotated_rgb(
            image_tensor, stats, int(height), int(width)
        )

        # 5. Format JSON result
        json_result = json.dumps([u.to_dict() for u in stats], indent=4)

        return result_image, json_result, perf_text

    except Exception as e:
        import traceback

        traceback.print_exc()
        return None, f"Error: {str(e)}", ""


# --- Build the UI ---


def create_ui(input_dir=None):
    with gr.Blocks(title="GPU Image Anomaly Detection") as demo:
        gr.Markdown("# 🔍 GPU Image Unit Detection")
        gr.Markdown(
            "Upload an image to find significant/unique blocks using CUDA acceleration."
        )

        with gr.Row():
            with gr.Column(scale=1):
                # Input Controls
                img_input = gr.Image(type="filepath", label="Input Image")

                # Display examples from server directory if provided
                if input_dir and os.path.isdir(input_dir):
                    exts = [
                        "*.bmp",
                        "*.png",
                        "*.jpg",
                        "*.jpeg",
                        "*.tiff",
                        "*.BMP",
                        "*.PNG",
                        "*.JPG",
                        "*.JPEG",
                    ]
                    server_images = []
                    for ext in exts:
                        server_images.extend(glob.glob(os.path.join(input_dir, ext)))
                    server_images = sorted(server_images)

                    if server_images:
                        gr.Examples(
                            examples=server_images,
                            inputs=img_input,
                            label=f"Select from Server Directory: {input_dir}",
                        )

                with gr.Row():
                    h_input = gr.Number(value=32, label="Unit Height", precision=0)
                    w_input = gr.Number(value=32, label="Unit Width", precision=0)

                metric_input = gr.Radio(
                    choices=["SSIM (Structure)", "CIELAB (Color)"],
                    value="SSIM (Structure)",
                    label="Comparison Metric",
                )

                with gr.Row():
                    top_n_input = gr.Number(value=5, label="Top N Units", precision=0)
                    sort_input = gr.Dropdown(
                        choices=["mean", "median", "std_dev", "min_score", "max_score"],
                        value="mean",
                        label="Sort By Stat",
                    )

                desc_input = gr.Checkbox(value=False, label="Sort Descending")

                btn_run = gr.Button("🚀 Run Detection", variant="primary")

            with gr.Column(scale=2):
                # Outputs
                img_output = gr.Image(label="Detected Units")
                perf_output = gr.Markdown()
                json_output = gr.Code(language="json", label="Statistics")

        # Event Binding
        # Auto-adjust sort direction: CIELAB -> Descending (True), SSIM -> Ascending (False)
        metric_input.change(
            fn=lambda x: x == "CIELAB (Color)", inputs=metric_input, outputs=desc_input
        )

        btn_run.click(
            fn=run_analysis,
            inputs=[
                img_input,
                h_input,
                w_input,
                metric_input,
                top_n_input,
                sort_input,
                desc_input,
            ],
            outputs=[img_output, json_output, perf_output],
        )
    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPU Image Anomaly Detection Web UI")
    parser.add_argument(
        "-i",
        "--input_dir",
        type=str,
        help="Directory containing images to list in UI",
        default=None,
    )
    args = parser.parse_args()

    # server_name="0.0.0.0" makes it accessible from external IP (SSH tunnel/remote)
    demo = create_ui(args.input_dir)
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
