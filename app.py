import argparse
import glob
import json
import os
import sys
import time

import gradio as gr
import torch

from analyzer import PatchAnalyzer
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
    print("Warning: Gradio version does not support Modals. Falling back to inline Group.")
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


def generate_preview(image_path, brightness, contrast, blur, sharpen, clahe, grayscale):
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
            image_tensor, float(blur), float(sharpen), float(clahe), grayscale
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
    n_clusters,
    cluster_method,
    brightness,
    contrast,
    blur,
    sharpen,
    clahe,
    grayscale,
    action_mode,
):
    """
    The core function called when user clicks 'Run Detection'
    action_mode: 'top_n', 'all', 'matrix'
    """
    if image_path is None:
        return None, None, "Please upload an image.", ""

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
            image_tensor, float(blur), float(sharpen), float(clahe), grayscale
        )

        # 2. Tile
        patches, grid_shape = processor.extract_patches(
            image_tensor, int(height), int(width)
        )

        # 3. Analyze & Annotate (Detection Phase)
        t_det_start = time.time()

        # Determine effective top_n
        if action_mode in ["all", "matrix"]:
            # Use a number larger than any possible grid count
            actual_top_n = 999999
        else:
            actual_top_n = int(top_n)

        if action_mode == "cluster":
            # For clustering, we pick a robust default metric (Texture usually best for grouping)
            metric = TextureColorMetric()
            analyzer = PatchAnalyzer(metric)
            labels = analyzer.cluster(patches, int(n_clusters), method=cluster_method)

            result_image = processor.create_cluster_heatmap(
                image_tensor, labels, grid_shape, int(height), int(width)
            )

            t_det_end = time.time()
            perf_text = (
                f"### 🧩 Clustering Metrics\n"
                f"- **Time:** {t_det_end - t_det_start:.4f} s\n"
                f"- **Units:** {patches.shape[0]}\n"
                f"- **Clusters:** {n_clusters}\n"
                f"- **Method:** {cluster_method.title()}"
            )

            # Clustering only has one result image. We put it in the first slot.
            outputs = [result_image, None] + [None] * (len(METRICS_CONFIG) * 2 - 2)
            return tuple(outputs + [json.dumps(labels), perf_text])

        # --- Multi-Metric Detection Loop ---
        all_image_outputs = []
        all_stats_collection = []

        for name, MetricClass in METRICS_CONFIG:
            # Instantiate and Analyze
            metric = MetricClass()
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
                image_tensor, stats, int(height), int(width)
            )

            # 2. Heatmap Image
            heatmap_img = processor.create_heatmap(
                image_tensor,
                stats,
                grid_shape,
                int(height),
                int(width),
                stat_name=sort_by,
            )

            all_image_outputs.append(det_img)
            all_image_outputs.append(heatmap_img)

            # Keep top 1 stat for JSON just to show something valid
            all_stats_collection.extend([s.to_dict() for s in stats[:1]])

        t_det_end = time.time()
        det_duration = t_det_end - t_det_start

        # Performance Stats
        N = patches.shape[0]
        total_pairs = N * N
        # multiplied by number of metrics
        cps = (
            (total_pairs * len(METRICS_CONFIG)) / det_duration
            if det_duration > 0
            else 0
        )

        perf_text = (
            f"### ⚡ Performance Metrics\n"
            f"- **Detection Time (All Metrics):** {det_duration:.4f} s\n"
            f"- **Total Units:** {N}\n"
            f"- **Throughput:** {cps:,.0f} pairs/sec"
        )

        # We just dump the first metric's stats to JSON to save space, or a summary
        json_result = json.dumps(all_stats_collection[:actual_top_n], indent=4)

        return tuple(all_image_outputs + [json_result, perf_text])

    except Exception as e:
        import traceback

        traceback.print_exc()
        # Return list of Nones matching output count
        count = len(METRICS_CONFIG) * 2
        return tuple([None] * count + [f"Error: {str(e)}", ""])


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
                preview_image = gr.Image(label="Processed Image", type="numpy", interactive=False)
            
            btn_close_preview = gr.Button("Close Preview")
            btn_close_preview.click(lambda: gr.update(visible=False), None, preview_modal)

        with gr.Row():
            with gr.Column(scale=1):
                # Action Buttons
                with gr.Row():
                    btn_run = gr.Button("🚀 Top N", variant="primary")
                    btn_all = gr.Button("👀 All Units")
                    btn_matrix = gr.Button("📊 Matrix")
                    btn_cluster = gr.Button("🧩 Cluster (Texture)")
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
                    grayscale_input = gr.Checkbox(value=False, label="Convert to Grayscale")

                with gr.Row():
                    cluster_method_input = gr.Dropdown(
                        choices=["hierarchical", "kmeans"],
                        value="hierarchical",
                        label="Cluster Method",
                    )
                    n_cluster_input = gr.Slider(
                        minimum=2,
                        maximum=20,
                        step=1,
                        value=3,
                        label="Number of Clusters",
                    )

                with gr.Row():
                    top_n_input = gr.Number(value=5, label="Top N Units", precision=0)
                    sort_input = gr.Dropdown(
                        choices=["mean", "median", "std_dev", "min_score", "max_score"],
                        value="mean",
                        label="Sort By Stat",
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
                        metric_outputs.extend([m_det, m_map])

                perf_output = gr.Markdown()
                json_output = gr.Code(language="json", label="Statistics")

        # Common inputs for all buttons
        common_inputs = [
            img_input,
            h_input,
            w_input,
            top_n_input,
            sort_input,
            desc_input,
            n_cluster_input,
            cluster_method_input,
            brightness_input,
            contrast_input,
            blur_input,
            sharpen_input,
            clahe_input,
            grayscale_input,
        ]
        common_outputs = metric_outputs + [json_output, perf_output]

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

        btn_cluster.click(
            fn=run_analysis,
            inputs=common_inputs + [gr.State("cluster")],
            outputs=common_outputs,
        )

        # Preview Button Logic
        preview_inputs = [
            img_input, brightness_input, contrast_input, 
            blur_input, sharpen_input, clahe_input, grayscale_input
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
