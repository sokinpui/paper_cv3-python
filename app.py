import argparse
import glob
import json
import os
import sys
import time

import gradio as gr
import torch

from analyzer import PatchAnalyzer
from metrics import CIELabMetric, SSIMMetric, LabMomentsMetric, TextureColorMetric, GradientColorMetric, HistogramMetric
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


def run_analysis(
    image_path,
    height,
    width,
    metric_name,
    top_n,
    sort_by,
    descending,
    use_local,
    n_clusters,
    cluster_method,
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

        if metric_name == "SSIM (Structure)":
            metric = SSIMMetric()
        elif metric_name == "Texture & Color (Defects)":
            metric = TextureColorMetric()
        elif metric_name == "Gradient & Color (Lines/Defects)":
            metric = GradientColorMetric()
        elif metric_name == "Color Histogram":
            metric = HistogramMetric()
        else:
            # Fallback / Default
            metric = LabMomentsMetric()

        analyzer = PatchAnalyzer(metric)

        # Pipeline
        # 1. Load
        image_tensor = processor.load_image(image_path)

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

        # Local Neighbor Radius: 1 means 3x3 window. None means Global.
        radius = 1 if use_local else None

        if action_mode == "cluster":
            labels = analyzer.cluster(patches, int(n_clusters), method=cluster_method)
            
            result_image = processor.create_cluster_heatmap(
                image_tensor, labels, grid_shape, int(height), int(width)
            )
            matrix_image = None
            
            # Performance stats for clustering
            t_det_end = time.time()
            det_duration = t_det_end - t_det_start
            
            perf_text = (
                f"### 🧩 Clustering Metrics\n"
                f"- **Time:** {det_duration:.4f} s\n"
                f"- **Units:** {patches.shape[0]}\n"
                f"- **Clusters:** {n_clusters}\n"
                f"- **Method:** {cluster_method.title()}"
            )
            
            return result_image, None, json.dumps(labels), perf_text

        stats = analyzer.analyze(
            patches,
            grid_shape,
            top_n=actual_top_n,
            sort_by=sort_by,
            ascending=not descending,
            neighbor_radius=radius,
        )

        result_image = processor.get_annotated_rgb(
            image_tensor, stats, int(height), int(width)
        )

        matrix_image = None
        if action_mode == "matrix":
            matrix_image = processor.create_heatmap(
                image_tensor, stats, grid_shape, int(height), int(width), stat_name=sort_by
            )

        t_det_end = time.time()
        det_duration = t_det_end - t_det_start

        # Clear GPU cache before running PaddleSeg to avoid OOM/Lag
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Performance Stats
        N = patches.shape[0]
        total_pairs = N * N
        cps = total_pairs / det_duration if det_duration > 0 else 0

        perf_text = (
            f"### ⚡ Performance Metrics\n"
            f"- **Detection Time (End-to-End):** {det_duration:.4f} s\n"
            f"- **Total Units:** {N} (Grid: {grid_shape})\n"
            f"- **Total Comparisons:** {total_pairs:,}\n"
            f"- **Detection Throughput:** {cps:,.0f} pairs/sec"
        )

        # 6. Format JSON result
        json_result = json.dumps([u.to_dict() for u in stats], indent=4)

        return result_image, matrix_image, json_result, perf_text

    except Exception as e:
        import traceback

        traceback.print_exc()
        return None, None, f"Error: {str(e)}", ""


# --- Build the UI ---


def create_ui(input_dir=None):
    with gr.Blocks(title="GPU Image Anomaly Detection") as demo:
        gr.Markdown("# 🔍 GPU Image Unit Detection")
        gr.Markdown(
            "Upload an image to find significant/unique blocks using CUDA acceleration."
        )

        with gr.Row():
            with gr.Column(scale=1):
                # Action Buttons
                with gr.Row():
                    btn_run = gr.Button("🚀 Top N", variant="primary")
                    btn_all = gr.Button("👀 All Units")
                    btn_matrix = gr.Button("📊 Matrix")
                    btn_cluster = gr.Button("🧩 Cluster")

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

                metric_input = gr.Radio(
                    choices=[
                        "SSIM (Structure)", 
                        "Gradient & Color (Lines/Defects)",
                        "Texture & Color (Defects)", 
                        "Color Histogram",
                        "LAB Moments (Color Stats)"
                    ],
                    value="SSIM (Structure)",
                    label="Comparison Metric",
                )

                with gr.Row():
                    cluster_method_input = gr.Dropdown(
                        choices=["hierarchical", "kmeans"], value="hierarchical", label="Cluster Method"
                    )
                    n_cluster_input = gr.Slider(minimum=2, maximum=20, step=1, value=3, label="Number of Clusters")


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

                local_input = gr.Checkbox(
                    value=False,
                    label="Local Comparison Only (Best for Line/Edge Defects)",
                )

            with gr.Column(scale=2):
                # Outputs
                with gr.Row():
                    img_output = gr.Image(label="My Detection (Unit Stats)")
                    matrix_output = gr.Image(label="Score Matrix Heatmap")

                perf_output = gr.Markdown()
                json_output = gr.Code(language="json", label="Statistics")

        # Common inputs for all buttons
        common_inputs = [
            img_input,
            h_input,
            w_input,
            metric_input,
            top_n_input,
            sort_input,
            desc_input,
            local_input,
            n_cluster_input,
            cluster_method_input,
        ]
        common_outputs = [img_output, matrix_output, json_output, perf_output]

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
