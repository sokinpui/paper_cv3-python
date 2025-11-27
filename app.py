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
    HumanEyeColorMetric,
    LabMomentsMetric,
    PixelWiseColorMetric,
    SSIMColorMixedMetric,
    SSIMHalfMetric,
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
    ("Oklab", HumanEyeColorMetric),
    ("SSIM-Half (Structure Only)", SSIMHalfMetric),
    ("SSIM & Color (Mixed)", SSIMColorMixedMetric),
    ("Gradient & Color (Lines)", GradientColorMetric),
    ("Texture & Color (Defects)", TextureColorMetric),
    ("Color Histogram", HistogramMetric),
    ("LAB Moments (Color Stats)", LabMomentsMetric),
    ("CIELAB", CIELabMetric),
    ("Pixel-wise Color (Full Lab)", PixelWiseColorMetric),
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


def run_analysis(
    image_path,
    height,
    width,
    top_n,
    sort_by,
    descending,
    overlap,
    action_mode_ui,
    k_clusters,
    cluster_show_scores,
    cluster_metric,
    cluster_threshold_n,
    selected_distance_functions,
):
    """
    The core function called when user clicks 'Run Detection'
    action_mode: 'top_n', 'all', 'heatmap', 'clustering'
    action_mode_ui: 'Top N', 'All Units', 'Heatmap', 'Clustering', 'Clustering (K-means)'
    """
    # Map UI string to internal mode
    mode_map = {
        "Top N": "top_n",
        "All Units": "all",
        "Heatmap": "heatmap",
        "Clustering": "clustering",
        "Clustering (K-means)": "clustering2",
        "Clustering (Hierarchical)": "clustering_hierarchical",
    }
    action_mode = mode_map.get(action_mode_ui, "top_n")

    # Initialize output structure: [Img, Perf] per metric + [JSON]
    num_metrics = len(METRICS_CONFIG)
    # Fill with None/Empty strings
    # Structure: [Header, Image, Perf] per metric
    current_outputs = [gr.update(visible=False)] * (num_metrics * 3) + [""]

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

        # 2. Tile
        patches, grid_shape, strides = processor.extract_patches(
            image_tensor, int(height), int(width), float(overlap)
        )

        # 3. Analyze & Annotate (Detection Phase)
        t_det_start = time.time()

        # Determine effective top_n
        if action_mode in [
            "all",
            "heatmap",
            "clustering",
            "clustering2",
            "clustering_hierarchical",
        ]:
            # Use a number larger than any possible grid count
            actual_top_n = 999999
        else:
            actual_top_n = int(top_n)

        all_stats_collection = []

        for i, (name, MetricClass) in enumerate(METRICS_CONFIG):
            base_idx = i * 3
            if name not in selected_distance_functions:
                current_outputs[base_idx] = gr.update(visible=False)
                current_outputs[base_idx + 1] = gr.update(visible=False)
                current_outputs[base_idx + 2] = gr.update(visible=False)
                continue

            t_metric_start = time.time()

            # Instantiate and Analyze
            metric = MetricClass()
            analyzer = PatchAnalyzer(metric)

            # If clustering2, we do clustering inside analyze on the matrix
            do_matrix_cluster = action_mode in [
                "clustering2",
                "clustering_hierarchical",
            ]
            algo = (
                "hierarchical" if action_mode == "clustering_hierarchical" else "kmeans"
            )

            # For hierarchical, we ignore k_clusters input and let analyzer decide (pass 0)
            k_val = 0 if action_mode == "clustering_hierarchical" else int(k_clusters)

            stats = analyzer.analyze(
                patches,
                grid_shape,
                top_n=actual_top_n,
                sort_by=sort_by,
                ascending=not descending,
                cluster_on_matrix=do_matrix_cluster,
                k=k_val,
                clustering_algorithm=algo,
            )

            # Perform Clustering (Stats-based) if requested
            if action_mode == "clustering":
                stats = analyzer.cluster_stats(
                    stats,
                    int(k_clusters),
                    metric=cluster_metric,
                    threshold_n=float(cluster_threshold_n),
                )

            # Generate Result Image based on Mode
            if action_mode in ["clustering", "clustering2", "clustering_hierarchical"]:
                result_img = processor.create_cluster_map(
                    image_tensor,
                    stats,
                    grid_shape,
                    strides,
                    int(height),
                    int(width),
                    show_scores=cluster_show_scores,
                )
            elif action_mode == "heatmap":
                result_img = processor.create_heatmap(
                    image_tensor,
                    stats,
                    grid_shape,
                    strides,
                    int(height),
                    int(width),
                    stat_name=sort_by,
                )
            else:
                # top_n or all: Show annotated boxes
                result_img = processor.get_annotated_rgb(
                    image_tensor, stats, int(height), int(width), grid_shape, strides
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
            current_outputs[base_idx] = gr.update(visible=True)
            current_outputs[base_idx + 1] = gr.update(visible=True, value=result_img)
            current_outputs[base_idx + 2] = gr.update(visible=True, value=perf_text)

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

        with gr.Row():
            with gr.Column(scale=1):
                # Action Buttons
                mode_input = gr.Radio(
                    choices=[
                        "Clustering (K-means)",
                        "Clustering (Hierarchical)",
                    ],
                    value="Clustering (K-means)",
                    label="Analysis Mode",
                )
                with gr.Row():
                    btn_run = gr.Button("🚀 Run Analysis", variant="primary")

                # Distance Function Selection
                metric_names = [m[0] for m in METRICS_CONFIG]
                distance_funcs_input = gr.CheckboxGroup(
                    choices=metric_names,
                    value=["Gradient & Color (Lines)"],
                    label="Distance Functions",
                )

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

                # Dynamic Settings
                top_n_input = gr.Number(
                    value=5, label="Top N Units", precision=0, visible=False
                )
                sort_input = gr.Dropdown(
                    choices=["mean", "median", "std_dev", "min_score", "max_score"],
                    value="mean",
                    label="Sort By Stat",
                    visible=False,
                )

                desc_input = gr.Checkbox(
                    value=True,
                    label="Sort Descending (High Score = Significant)",
                    visible=False,
                )

                k_input = gr.Slider(
                    minimum=2,
                    maximum=8,
                    value=2,
                    step=1,
                    label="K Clusters (for Clustering)",
                    visible=True,
                )

                cluster_metric_input = gr.Dropdown(
                    choices=["mean", "std_dev", "threshold"],
                    value="mean",
                    label="Clustering Score",
                    visible=False,
                )

                cluster_threshold_n_input = gr.Number(
                    value=1.0,
                    label="Threshold N (Mean + N * Std)",
                    visible=False,
                )

                cluster_show_scores = gr.Checkbox(
                    value=False, label="Show Scores on Map", visible=True
                )

                # Visibility Logic
                def update_visibility(mode, metric):
                    is_top_n = mode == "Top N"
                    is_all = mode == "All Units"
                    is_heatmap = mode == "Heatmap"
                    is_cluster = mode == "Clustering"
                    is_cluster2 = mode == "Clustering (K-means)"
                    is_cluster_h = mode == "Clustering (Hierarchical)"
                    is_threshold = is_cluster and (metric == "threshold")

                    return (
                        gr.update(visible=is_top_n),  # top_n
                        gr.update(visible=(is_top_n or is_all or is_heatmap)),  # sort
                        gr.update(visible=(is_top_n or is_all)),  # desc
                        gr.update(
                            visible=(is_cluster or is_cluster2)
                        ),  # k (Hidden for Hierarchical)
                        gr.update(
                            visible=(is_cluster or is_cluster2 or is_cluster_h)
                        ),  # show_scores
                        gr.update(
                            visible=is_cluster
                        ),  # cluster_metric (only for stats clustering)
                        gr.update(visible=is_threshold),  # threshold_n
                    )

                mode_input.change(
                    fn=update_visibility,
                    inputs=[mode_input, cluster_metric_input],
                    outputs=[
                        top_n_input,
                        sort_input,
                        desc_input,
                        k_input,
                        cluster_show_scores,
                        cluster_metric_input,
                        cluster_threshold_n_input,
                    ],
                )

                cluster_metric_input.change(
                    fn=update_visibility,
                    inputs=[mode_input, cluster_metric_input],
                    outputs=[
                        top_n_input,
                        sort_input,
                        desc_input,
                        k_input,
                        cluster_show_scores,
                        cluster_metric_input,
                        cluster_threshold_n_input,
                    ],
                )

            with gr.Column(scale=3):
                gr.Markdown("### 📊 Analysis Results (By Distance Function)")

                # Dynamically create output rows for each metric
                metric_outputs = []
                for name, _ in METRICS_CONFIG:
                    m_header = gr.Markdown(f"**{name}**")
                    m_img = gr.Image(label=f"Result ({name})", type="numpy")
                    m_perf = gr.Markdown(value="Waiting...")
                    metric_outputs.extend([m_header, m_img, m_perf])

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
            overlap_input,
            mode_input,
            k_input,
            cluster_show_scores,
            cluster_metric_input,
            cluster_threshold_n_input,
            distance_funcs_input,
        ]
        common_outputs = metric_outputs + [json_output]

        btn_run.click(
            fn=run_analysis,
            inputs=common_inputs,
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
