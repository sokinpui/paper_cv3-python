import argparse
import glob
import json
import os
import sys
import time

import gradio as gr
import torch

from analyzer import PatchAnalyzer, LocalAnomalyAnalyzer
from metrics import (
    CIELabMetric,
    GradientColorMetric,
    HistogramMetric,
    LabMomentsMetric,
    SSIMMetric,
    TextureColorMetric,
)
from processor import ImageProcessor

# Compatibility for older Gradio versions (Pre-5.0)
if not hasattr(gr, "Modal"):
    print(
        "Warning: Gradio version does not support Modals. Falling back to inline Group."
    )
    gr.Modal = gr.Group

# 0. Configuration
METRICS_CONFIG = [
    ("SSIM (Structure)", SSIMMetric),
    ("Gradient & Color (Lines)", GradientColorMetric),
    ("Texture & Color (Defects)", TextureColorMetric),
    ("Color Histogram", HistogramMetric),
    ("LAB Moments (Color Stats)", LabMomentsMetric),
]

# 1. Initialize CUDA Device
try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"Web UI running on: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}"
    )
except Exception as e:
    print(f"Warning: {e}")
    device = torch.device("cpu")


def generate_preview(image_path, brightness, contrast, blur, sharpen, clahe, grayscale, quantize):
    """
    Lightweight function to generate a preview of the pre-processing.
    Returns: (Modal Update, Image)
    """
    if image_path is None:
        return gr.update(visible=False), None

    try:
        processor = ImageProcessor(device)
        image_tensor = processor.load_image(image_path)

        # Adjustments
        image_tensor = processor.adjust_image(
            image_tensor, float(brightness), float(contrast)
        )
        image_tensor = processor.apply_preprocessing(
            image_tensor, float(blur), float(sharpen), float(clahe), grayscale, quantize
        )

        img_np = image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        img_np = (img_np * 255).clip(0, 255).astype("uint8")

        return gr.update(visible=True), img_np
    except Exception as e:
        print(f"Preview Error: {e}")
        return gr.update(visible=False), None


def run_analysis(
    image_path,
    height,
    width,
    top_n,
    sort_by,
    descending,
    brightness,
    contrast,
    blur,
    sharpen,
    clahe,
    grayscale,
    quantize,
    overlap,
    algo_mode,
    action_mode,
):
    """
    The core function called when user clicks 'Run Detection'
    action_mode: 'top_n', 'all', 'matrix'
    """
    # Initialize output structure: [Det, Map, Perf] per metric + [JSON]
    num_metrics = len(METRICS_CONFIG)
    # Fill with None/Empty strings
    current_outputs = [None] * (num_metrics * 3) + [""]

    if image_path is None:
        current_outputs[-1] = "Please upload an image."
        yield tuple(current_outputs)
        return

    try:
        # Setup Components
        processor = ImageProcessor(device)

        # Pipeline
        # 1. Load
        image_tensor = processor.load_image(image_path)

        # 1.5 Adjust Image (Global Brightness/Contrast)
        image_tensor = processor.adjust_image(
            image_tensor, float(brightness), float(contrast)
        )

        # 1.6 Preprocessing (Blur, Sharpen, CLAHE)
        image_tensor = processor.apply_preprocessing(
            image_tensor, float(blur), float(sharpen), float(clahe), grayscale, quantize
        )

        # 2. Tile
        patches, grid_shape, strides = processor.extract_patches(
            image_tensor, int(height), int(width), float(overlap)
        )

        # 3. Analyze & Annotate (Detection Phase)
        t_det_start = time.time()

        # Determine effective top_n
        if action_mode in ["all", "matrix"]:
            # Use a number larger than any possible grid count
            actual_top_n = 999999
        else:
            actual_top_n = int(top_n)

        all_stats_collection = []

        for i, (name, MetricClass) in enumerate(METRICS_CONFIG):
            t_metric_start = time.time()

            # Instantiate and Analyze
            metric = MetricClass()

            if algo_mode.startswith("Local"):
                analyzer = LocalAnomalyAnalyzer(metric)
            else:
                analyzer = PatchAnalyzer(metric)

            stats = analyzer.analyze(
                patches,
                grid_shape,
                top_n=actual_top_n,
                sort_by=sort_by,
                ascending=not descending,
            )

            # 1. Detection Image
            det_img = processor.get_annotated_rgb(
                image_tensor, stats, int(height), int(width), grid_shape, strides
            )

            # 2. Heatmap Image
            heatmap_img = processor.create_heatmap(
                image_tensor,
                stats,
                grid_shape,
                strides,
                int(height),
                int(width),
                stat_name=sort_by,
            )

            t_metric_end = time.time()
            metric_duration = t_metric_end - t_metric_start

            # Performance Stats for this metric
            N = patches.shape[0]
            total_pairs = N * N
            cps = total_pairs / metric_duration if metric_duration > 0 else 0

            perf_text = (
                f"**{name} Performance:** "
                f"{metric_duration:.4f} s | "
                f"{cps:,.0f} pairs/sec"
            )

            # Update specific slots in the output list
            base_idx = i * 3
            current_outputs[base_idx] = det_img
            current_outputs[base_idx + 1] = heatmap_img
            current_outputs[base_idx + 2] = perf_text

            # Keep top 1 stat for JSON just to show something valid
            all_stats_collection.extend([s.to_dict() for s in stats[:1]])

            # Update JSON (accumulated)
            current_outputs[-1] = json.dumps(
                all_stats_collection[:actual_top_n], indent=4
            )

            # Yield current state
            yield tuple(current_outputs)

    except Exception as e:
        import traceback

        traceback.print_exc()
        # Yield error in the JSON field
        current_outputs[-1] = f"Error: {str(e)}"
        yield tuple(current_outputs)


# --- Build the UI ---


def create_ui(input_dir=None):
    with gr.Blocks(title="GPU Image Anomaly Detection") as demo:
        gr.Markdown("# 🔍 GPU Image Unit Detection")
        gr.Markdown(
            "Upload an image to find significant/unique blocks using CUDA acceleration."
        )

        # --- Popup Preview Modal ---
        with gr.Modal(visible=False) as preview_modal:
            gr.Markdown("### 🖼️ Pre-processing Preview")
            with gr.Row():
                preview_image = gr.Image(
                    label="Processed Image", type="numpy", interactive=False
                )

            btn_close_preview = gr.Button("Close Preview")
            btn_close_preview.click(
                lambda: gr.update(visible=False), None, preview_modal
            )

        with gr.Row():
            with gr.Column(scale=1):
                # Action Buttons
                with gr.Row():
                    btn_run = gr.Button("🚀 Top N", variant="primary")
                    btn_all = gr.Button("👀 All Units")
                    btn_matrix = gr.Button("📊 Matrix")
                    btn_preview = gr.Button("🖼️ Preview")

                gr.Markdown("### Settings")

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
                    h_input = gr.Number(value=200, label="Unit Height", precision=0)
                    w_input = gr.Number(value=200, label="Unit Width", precision=0)

                with gr.Row():
                    overlap_input = gr.Slider(
                        minimum=0.0,
                        maximum=0.9,
                        value=0.0,
                        step=0.05,
                        label="Overlap Ratio",
                    )

                with gr.Row():
                    brightness_input = gr.Slider(
                        minimum=-0.5,
                        maximum=0.5,
                        value=0.0,
                        step=0.05,
                        label="Brightness",
                    )
                    contrast_input = gr.Slider(
                        minimum=0.5,
                        maximum=3.0,
                        value=1.0,
                        step=0.1,
                        label="Contrast (Global)",
                    )

                with gr.Row():
                    blur_input = gr.Slider(
                        minimum=0, maximum=5, value=0, step=1, label="Blur (Denoise)"
                    )
                    sharpen_input = gr.Slider(
                        minimum=0.0, maximum=2.0, value=0.0, step=0.1, label="Sharpen"
                    )
                    clahe_input = gr.Slider(
                        minimum=0.0,
                        maximum=5.0,
                        value=0.0,
                        step=0.5,
                        label="CLAHE (Texture)",
                    )

                with gr.Row():
                    grayscale_input = gr.Checkbox(
                        value=False, label="Convert to Grayscale"
                    )
                    quantize_input = gr.Checkbox(
                        value=False, label="4-bit Grayscale (16 Levels)"
                    )

                with gr.Row():
                    top_n_input = gr.Number(value=5, label="Top N Units", precision=0)
                    sort_input = gr.Dropdown(
                        choices=["mean", "median", "std_dev", "min_score", "max_score"],
                        value="mean",
                        label="Sort By Stat",
                    )

                algo_input = gr.Radio(
                    choices=["Global (Coherence)", "Local (Neighbors)"],
                    value="Global (Coherence)",
                    label="Detection Algorithm",
                )

                desc_input = gr.Checkbox(
                    value=True, label="Sort Descending (High Score = Significant)"
                )

            with gr.Column(scale=3):
                gr.Markdown("### 📊 Analysis Results (Metric by Metric)")

                # Dynamically create output rows for each metric
                metric_outputs = []
                for name, _ in METRICS_CONFIG:
                    gr.Markdown(f"**{name}**")
                    with gr.Row():
                        m_det = gr.Image(label=f"Detection ({name})", type="numpy")
                        m_map = gr.Image(label=f"Heatmap ({name})", type="numpy")
                    m_perf = gr.Markdown(value="Waiting...")
                    metric_outputs.extend([m_det, m_map, m_perf])

                # perf_output = gr.Markdown() # Removed global perf
                json_output = gr.Code(language="json", label="Statistics")

        # Common inputs for all buttons
        common_inputs = [
            img_input,
            h_input,
            w_input,
            top_n_input,
            sort_input,
            desc_input,
            brightness_input,
            contrast_input,
            blur_input,
            sharpen_input,
            clahe_input,
            grayscale_input,
            quantize_input,
            overlap_input,
            algo_input,
        ]
        common_outputs = metric_outputs + [json_output]

        btn_run.click(
            fn=run_analysis,
            inputs=common_inputs + [gr.State("top_n")],
            outputs=common_outputs,
        )

        btn_all.click(
            fn=run_analysis,
            inputs=common_inputs + [gr.State("all")],
            outputs=common_outputs,
        )

        btn_matrix.click(
            fn=run_analysis,
            inputs=common_inputs + [gr.State("matrix")],
            outputs=common_outputs,
        )

        # Preview Button Logic
        preview_inputs = [
            img_input,
            brightness_input,
            contrast_input,
            blur_input,
            sharpen_input,
            clahe_input,
            grayscale_input,
            quantize_input,
        ]
        btn_preview.click(
            fn=generate_preview,
            inputs=preview_inputs,
            outputs=[preview_modal, preview_image],
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
