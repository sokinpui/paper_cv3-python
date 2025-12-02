import argparse
import glob
import json
import os
import sys
import time

import gradio as gr
import numpy as np
import torch

from analyzer import PatchAnalyzer
from metrics import GradientColorMetric, HumanEyeColorMetric
from processor import ImageProcessor

# Compatibility for older Gradio versions (Pre-5.0)
if not hasattr(gr, "Modal"):
    print(
        "Warning: Gradio version does not support Modals. Falling back to inline Group."
    )
    gr.Modal = gr.Group

# 0. Configuration
METRICS_CONFIG = [
    ("Oklab", HumanEyeColorMetric),
    ("Gradient & Color (Lines)", GradientColorMetric),
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


def find_unit_index_from_click(x, y, grid_info):
    """
    Finds the unit index closest to the click (x, y).
    grid_info: dict with grid_shape, strides, unit_size, img_shape
    """
    rows, cols = grid_info["grid_shape"]
    stride_h, stride_w = grid_info["strides"]
    unit_h, unit_w = grid_info["unit_size"]
    H, W = grid_info["img_shape"]

    best_idx = -1
    min_dist_sq = float("inf")

    # Iterate all units to find the one containing the click or closest to center
    # Index i = r * cols + c
    for r in range(rows):
        for c in range(cols):
            # Calculate top-left (bx, by)
            if r == rows - 1:
                by = H - unit_h
            else:
                by = r * stride_h

            if c == cols - 1:
                bx = W - unit_w
            else:
                bx = c * stride_w

            # Check if click is strictly inside box
            if bx <= x < bx + unit_w and by <= y < by + unit_h:
                # Calculate distance to center for tie-breaking overlapping units
                cx = bx + unit_w / 2
                cy = by + unit_h / 2
                dist_sq = (x - cx) ** 2 + (y - cy) ** 2

                if dist_sq < min_dist_sq:
                    min_dist_sq = dist_sq
                    best_idx = r * cols + c

    return best_idx


def on_unit_click(metric_name, evt: gr.SelectData, state, vec_a, vec_b):
    """
    Handles click on the result image.
    Populates Vector A or Vector B with the distance vector of the clicked unit.
    """
    if not state or metric_name not in state:
        return vec_a, vec_b

    data = state[metric_name]
    # evt.index is [x, y]
    idx = find_unit_index_from_click(evt.index[0], evt.index[1], data)

    if idx < 0:
        return vec_a, vec_b

    matrix = data["matrix"]
    if idx >= len(matrix):
        return vec_a, vec_b

    # Get row vector (distances from this unit to all others)
    vector = matrix[idx]
    # Replace NaN (self-comparison) with 0.0
    vector = np.nan_to_num(vector, nan=0.0)

    # Format as string for the text box
    # Using 4 decimal places for conciseness
    vec_str = ", ".join([f"{x:.4f}" for x in vector])

    # Logic: Fill A if empty. If A is full, fill B.
    # If B is also full, overwrite B (most recent click replaces B).
    if not vec_a:
        # Fill A
        return vec_str, vec_b
    elif not vec_b:
        # Fill B
        return vec_a, vec_str
    else:
        # Overwrite B
        return vec_a, vec_str


def create_click_handler(metric_name):
    """
    Creates a closure for the click handler to avoid partial introspection issues
    in Gradio. Captures metric_name.
    """

    def handler(evt: gr.SelectData, state, vec_a, vec_b):
        return on_unit_click(metric_name, evt, state, vec_a, vec_b)

    return handler


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
    hierarchical_method,
    dbscan_eps,
    dbscan_min_samples,
    power_transform_degree,
    oklab_multiplier,
    oklab_exponent,
    current_state,
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
        "Clustering (Spectral)": "clustering_spectral",
        "Clustering (DBSCAN)": "clustering_dbscan",
    }
    action_mode = mode_map.get(action_mode_ui, "top_n")

    # Initialize output structure: [Img, Perf] per metric + [JSON]
    num_metrics = len(METRICS_CONFIG)
    # Fill with None/Empty strings
    # Structure: [Header, Image, Perf] per metric
    # + [JSON] + [State]
    current_outputs = [gr.update(visible=False)] * (num_metrics * 3) + [
        "",
        current_state,
    ]

    if image_path is None:
        current_outputs[-2] = "Please upload an image."
        yield tuple(current_outputs)
        return

    new_state = {}

    try:
        # Setup Components
        processor = ImageProcessor(device)

        # Pipeline
        # 1. Load
        image_tensor = processor.load_image(image_path)
        img_h, img_w = image_tensor.shape[2], image_tensor.shape[3]

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
            "clustering_spectral",
            "clustering_dbscan",
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
            if name == "Oklab":
                metric = MetricClass(
                    multiplier=float(oklab_multiplier), exponent=float(oklab_exponent)
                )
            else:
                metric = MetricClass()

            analyzer = PatchAnalyzer(metric)

            # If clustering2, we do clustering inside analyze on the matrix
            do_matrix_cluster = action_mode in [
                "clustering2",
                "clustering_hierarchical",
                "clustering_spectral",
                "clustering_dbscan",
            ]
            algo = "kmeans"
            if action_mode == "clustering_hierarchical":
                algo = "hierarchical"
            elif action_mode == "clustering_spectral":
                algo = "spectral"
            elif action_mode == "clustering_dbscan":
                algo = "dbscan"

            # For hierarchical, we ignore k_clusters input and let analyzer decide (pass 0)
            k_val = 0 if action_mode == "clustering_hierarchical" else int(k_clusters)

            stats, matrix = analyzer.analyze(
                patches,
                grid_shape,
                top_n=actual_top_n,
                sort_by=sort_by,
                ascending=not descending,
                cluster_on_matrix=do_matrix_cluster,
                k=k_val,
                clustering_algorithm=algo,
                hierarchical_method=hierarchical_method,
                eps=float(dbscan_eps),
                min_samples=int(dbscan_min_samples),
                power_transform_degree=float(power_transform_degree),
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
            if action_mode in [
                "clustering",
                "clustering2",
                "clustering_hierarchical",
                "clustering_spectral",
                "clustering_dbscan",
            ]:
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
                # top_n, all: Show annotated boxes
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

            # Store Data in State
            new_state[name] = {
                "matrix": matrix.detach().cpu().numpy(),  # Store as numpy
                "grid_shape": grid_shape,
                "strides": strides,
                "unit_size": (int(height), int(width)),
                "img_shape": (img_h, img_w),
            }

            # Update specific slots in the output list
            current_outputs[base_idx] = gr.update(visible=True)
            current_outputs[base_idx + 1] = gr.update(visible=True, value=result_img)
            current_outputs[base_idx + 2] = gr.update(visible=True, value=perf_text)

            # Keep top 1 stat for JSON just to show something valid
            all_stats_collection.extend([s.to_dict() for s in stats[:1]])

            # Update JSON (accumulated)
            current_outputs[-2] = json.dumps(
                all_stats_collection[:actual_top_n], indent=4
            )
            current_outputs[-1] = new_state

            # Yield current state
            yield tuple(current_outputs)

    except Exception as e:
        import traceback

        traceback.print_exc()
        # Yield error in the JSON field
        current_outputs[-2] = f"Error: {str(e)}"
        yield tuple(current_outputs)


def calculate_vector_distance(vec_a, vec_b):
    if not vec_a and not vec_b:
        return "Please provide at least one vector."

    results = []
    v_a, v_b = None, None

    try:

        def parse(s):
            s = (
                s.strip()
                .replace("[", "")
                .replace("]", "")
                .replace("(", "")
                .replace(")", "")
            )
            arr = np.array([float(x) for x in s.split(",") if x.strip()])
            return np.nan_to_num(arr, nan=0.0)

        if vec_a:
            v_a = parse(vec_a)
            results.append("--- Vector A ---")
            results.append(f"L1 Norm (Manhattan): {np.sum(np.abs(v_a)):.6f}")
            results.append(f"L2 Norm (Euclidean): {np.linalg.norm(v_a):.6f}")

        if vec_b:
            v_b = parse(vec_b)
            if vec_a:
                results.append("")  # Add a newline for separation
            results.append("--- Vector B ---")
            results.append(f"L1 Norm (Manhattan): {np.sum(np.abs(v_b)):.6f}")
            results.append(f"L2 Norm (Euclidean): {np.linalg.norm(v_b):.6f}")

        if v_a is not None and v_b is not None:
            results.append("")
            results.append("--- Distance (A vs B) ---")
            if v_a.shape != v_b.shape:
                results.append(f"Shape Mismatch: {v_a.shape} vs {v_b.shape}")
            else:
                dist_euc = np.linalg.norm(v_a - v_b)
                dist_man = np.sum(np.abs(v_a - v_b))
                n_a, n_b = np.linalg.norm(v_a), np.linalg.norm(v_b)
                dist_cos = (
                    1.0 - (np.dot(v_a, v_b) / (n_a * n_b))
                    if n_a > 0 and n_b > 0
                    else 1.0
                )
                results.append(f"Euclidean: {dist_euc:.6f}")
                results.append(f"Cosine:    {dist_cos:.6f}")
                results.append(f"Manhattan: {dist_man:.6f}")

        return "\n".join(results)
    except Exception as e:
        return f"Error: {e}"


def run_and_plot_distribution(
    image_path,
    height,
    width,
    overlap,
    metric_name,
    power_transform_degree,
    oklab_multiplier,
    oklab_exponent,
):
    """
    Performs a dedicated analysis and generates a bar chart of the results.
    """
    if not image_path or not metric_name:
        return None

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            "Error: Matplotlib is required. Please install it: pip install matplotlib"
        )
        return None

    try:
        # Setup Components
        processor = ImageProcessor(device)
        MetricClass = dict(METRICS_CONFIG)[metric_name]
        if metric_name == "Oklab":
            metric = MetricClass(
                multiplier=float(oklab_multiplier), exponent=float(oklab_exponent)
            )
        else:
            metric = MetricClass()
        analyzer = PatchAnalyzer(metric)

        # Run a silent analysis
        image_tensor = processor.load_image(image_path)
        patches, grid_shape, _ = processor.extract_patches(
            image_tensor, int(height), int(width), float(overlap)
        )

        # Get stats for all units
        stats, matrix = analyzer.analyze(
            patches,
            grid_shape,
            top_n=999999,
            sort_by="mean",  # Not critical as we sort later
            ascending=True,
            power_transform_degree=float(power_transform_degree),
        )

        if not stats:
            return None

        # Calculate L2 norm of each unit's distance vector
        vec_norms = torch.sqrt(torch.nansum(matrix**2, dim=1))
        values = vec_norms.cpu().numpy().tolist()
        values.sort()

        # Plotting logic
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(range(len(values)), values)
        ax.set_title(f"Distribution of Distance Vector Lengths for '{metric_name}'")
        ax.set_xlabel("Unit Index (Sorted by Score)")
        ax.set_ylabel("L2 Norm of Distance Vector")
        ax.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()

        return fig

    except Exception as e:
        print(f"Error during plotting analysis: {e}")
        import traceback

        traceback.print_exc()
        return None


# --- Build the UI ---


def clear_vector_inputs():
    return "", "", ""


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
                        # "Clustering (K-means)",
                        # "Clustering (Spectral)",
                        # "Clustering (Hierarchical)",
                        "Clustering (DBSCAN)",
                    ],
                    value="Clustering (DBSCAN)",
                    label="Analysis Mode",
                )
                with gr.Row():
                    btn_run = gr.Button("🚀 Run Analysis", variant="primary")

                # Distance Function Selection
                metric_names = [m[0] for m in METRICS_CONFIG]
                distance_funcs_input = gr.CheckboxGroup(
                    choices=metric_names,
                    value=["Oklab"],
                    label="Distance Functions",
                )

                with gr.Group() as oklab_settings:
                    gr.Markdown("##### Oklab Settings")
                    oklab_multiplier_input = gr.Slider(
                        minimum=1.0,
                        maximum=100.0,
                        value=1.0,
                        step=1,
                        label="Distance Multiplier",
                        info="Amplifies raw distance before exponent. Higher = more sensitive.",
                    )
                    oklab_exponent_input = gr.Slider(
                        minimum=1,
                        maximum=20.0,
                        value=1,
                        step=0.1,
                        label="Distance Exponent",
                        info="Power to raise distance to. >1 exaggerates large distances.",
                    )

                def update_oklab_visibility(selected_metrics):
                    return gr.update(visible="Oklab" in selected_metrics)

                distance_funcs_input.change(
                    fn=update_oklab_visibility,
                    inputs=distance_funcs_input,
                    outputs=[oklab_settings],
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
                            examples_per_page=20,
                        )

                unit_size_presets = {
                    "30x30 (Tiny)": (30, 30),
                    "50x50 (Small)": (50, 50),
                    "100x100 (Medium)": (100, 100),
                    "150x150 (Large)": (150, 150),
                    "200x200 (X-Large)": (200, 200),
                    "250x250 (Huge)": (250, 250),
                }
                unit_preset_input = gr.Radio(
                    choices=list(unit_size_presets.keys()),
                    value="50x50 (Small)",
                    label="Unit Size Presets",
                )

                with gr.Row():
                    h_input = gr.Number(value=50, label="Unit Height", precision=0)
                    w_input = gr.Number(value=50, label="Unit Width", precision=0)

                with gr.Row():
                    overlap_input = gr.Slider(
                        minimum=0.0,
                        maximum=0.9,
                        value=0.0,
                        step=0.05,
                        label="Overlap Ratio",
                    )

                power_transform_input = gr.Slider(
                    minimum=0.1,
                    maximum=5.0,
                    value=1.0,
                    step=0.1,
                    label="Power Transformation",
                    info="Raise distance to a power (distance^n). >1 exaggerates large distances, <1 flattens them.",
                )

                def update_unit_size(preset_key):
                    h, w = unit_size_presets[preset_key]
                    return h, w

                unit_preset_input.change(
                    fn=update_unit_size,
                    inputs=unit_preset_input,
                    outputs=[h_input, w_input],
                )

                # Dynamic Settings
                top_n_input = gr.Number(
                    value=5, label="Top N Units", precision=0, visible=False
                )
                sort_input = gr.Dropdown(
                    choices=[
                        "mean", "median", "std_dev", "min_score", "max_score", "l2_norm"
                    ],
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

                h_method_input = gr.Dropdown(
                    choices=["ward", "single", "complete", "average"],
                    value="ward",
                    label="Linkage Method (Hierarchical Only)",
                    visible=False,
                )

                # DBSCAN Settings
                dbscan_eps_input = gr.Number(
                    value=0.0,
                    label="DBSCAN Eps",
                    info="Distance threshold. Set to 0.0 for auto-detection (K-needle).",
                    visible=False,
                )
                dbscan_min_input = gr.Number(
                    value=4,
                    label="DBSCAN Min Samples",
                    precision=0,
                    visible=False,
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
                def update_visibility(mode, metric, dbscan_eps):
                    is_top_n = mode == "Top N"
                    is_all = mode == "All Units"
                    is_heatmap = mode == "Heatmap"
                    is_cluster = mode == "Clustering"
                    is_cluster2 = mode == "Clustering (K-means)"
                    is_cluster_h = mode == "Clustering (Hierarchical)"
                    is_cluster_s = mode == "Clustering (Spectral)"
                    is_cluster_d = mode == "Clustering (DBSCAN)"

                    is_threshold = is_cluster and (metric == "threshold")
                    is_dbscan_mode = is_cluster_d
                    is_clustering_any = (
                        is_cluster
                        or is_cluster2
                        or is_cluster_h
                        or is_cluster_s
                        or is_cluster_d
                    )

                    return (
                        gr.update(visible=is_top_n),  # top_n
                        gr.update(visible=(is_top_n or is_all or is_heatmap)),  # sort
                        gr.update(visible=(is_top_n or is_all)),  # desc
                        gr.update(
                            visible=(is_cluster or is_cluster2 or is_cluster_s)
                            and not is_dbscan_mode
                        ),  # k (Hidden for Hierarchical)
                        gr.update(visible=is_clustering_any),  # show_scores
                        gr.update(
                            visible=is_cluster
                        ),  # cluster_metric (only for stats clustering)
                        gr.update(visible=is_cluster_h),  # linkage method
                        gr.update(visible=is_threshold),  # threshold_n
                        gr.update(visible=is_dbscan_mode),  # dbscan eps
                        gr.update(visible=is_dbscan_mode),  # dbscan min
                    )

                mode_input.change(
                    fn=update_visibility,
                    inputs=[mode_input, cluster_metric_input, dbscan_eps_input],
                    outputs=[
                        top_n_input,
                        sort_input,
                        desc_input,
                        k_input,
                        cluster_show_scores,
                        cluster_metric_input,
                        h_method_input,
                        cluster_threshold_n_input,
                        dbscan_eps_input,
                        dbscan_min_input,
                    ],
                )

                for comp in [cluster_metric_input, dbscan_eps_input]:
                    comp.change(
                        fn=update_visibility,
                        inputs=[mode_input, cluster_metric_input, dbscan_eps_input],
                        outputs=[
                            top_n_input,
                            sort_input,
                            desc_input,
                            k_input,
                            cluster_show_scores,
                            cluster_metric_input,
                            h_method_input,
                            cluster_threshold_n_input,
                            dbscan_eps_input,
                            dbscan_min_input,
                        ],
                    )

                # Trigger visibility update on load to match default mode
                demo.load(
                    fn=update_visibility,
                    inputs=[mode_input, cluster_metric_input, dbscan_eps_input],
                    outputs=[
                        top_n_input,
                        sort_input,
                        desc_input,
                        k_input,
                        cluster_show_scores,
                        cluster_metric_input,
                        h_method_input,
                        cluster_threshold_n_input,
                        dbscan_eps_input,
                        dbscan_min_input,
                    ],
                )

                # Trigger oklab visibility on load
                demo.load(
                    fn=update_oklab_visibility,
                    inputs=distance_funcs_input,
                    outputs=[oklab_settings],
                )

            with gr.Column(scale=3):
                gr.Markdown("### 📊 Analysis Results (By Distance Function)")

                # Dynamically create output rows for each metric
                metric_outputs = []
                metric_images = []  # Keep track of image components to bind events

                for name, _ in METRICS_CONFIG:
                    m_header = gr.Markdown(f"**{name}**", visible=False)
                    m_img = gr.Image(
                        label=f"Result ({name})", type="numpy", visible=False
                    )
                    m_perf = gr.Markdown(value="Waiting...", visible=False)
                    metric_outputs.extend([m_header, m_img, m_perf])
                    metric_images.append((name, m_img))

                gr.Markdown("### 📐 Vector Calculator")
                with gr.Group():
                    with gr.Row():
                        vc_a = gr.Textbox(label="Vector A", placeholder="1.0, 2.0, ...")
                        vc_b = gr.Textbox(label="Vector B", placeholder="3.0, 4.0, ...")
                    with gr.Row():
                        vc_btn = gr.Button("Calculate Distance", variant="primary")
                        vc_clear = gr.Button("Clear")
                    vc_res = gr.Textbox(label="Results", lines=8)
                    vc_btn.click(
                        calculate_vector_distance,
                        inputs=[vc_a, vc_b],
                        outputs=vc_res,
                    )
                    vc_clear.click(
                        clear_vector_inputs, inputs=None, outputs=[vc_a, vc_b, vc_res]
                    )

                # perf_output = gr.Markdown() # Removed global perf
                json_output = gr.Code(language="json", label="Statistics")
                analysis_state = gr.State({})  # Store matrix data per session

                gr.Markdown("### 📈 Score Distribution")
                with gr.Group():
                    gr.Markdown(
                        "Generate a plot of the sorted distance scores for all units using a specific metric. This runs a separate, dedicated analysis."
                    )
                    with gr.Row():
                        plot_metric_select = gr.Dropdown(
                            choices=[m[0] for m in METRICS_CONFIG],
                            value="Oklab",
                            label="Select Metric to Plot",
                        )
                    plot_run_btn = gr.Button(
                        "📊 Generate Distribution Plot", variant="primary"
                    )
                    score_dist_plot = gr.Plot(label="Score Distribution")

                    plot_run_btn.click(
                        fn=run_and_plot_distribution,
                        inputs=[
                            img_input,
                            h_input,
                            w_input,
                            overlap_input,
                            plot_metric_select,
                            power_transform_input,
                            oklab_multiplier_input,
                            oklab_exponent_input,
                        ],
                        outputs=[score_dist_plot],
                    )

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
            h_method_input,
            dbscan_eps_input,
            dbscan_min_input,
            power_transform_input,
            oklab_multiplier_input,
            oklab_exponent_input,
            analysis_state,
        ]
        common_outputs = metric_outputs + [json_output, analysis_state]

        btn_run.click(
            fn=run_analysis,
            inputs=common_inputs,
            outputs=common_outputs,
        )

        # Wire Select/Click Events for Result Images
        for name, img_comp in metric_images:
            img_comp.select(
                fn=create_click_handler(name),
                inputs=[analysis_state, vc_a, vc_b],
                outputs=[vc_a, vc_b],
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
