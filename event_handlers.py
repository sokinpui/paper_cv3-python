import json
import time

import cv2
import gradio as gr
import numpy as np
import torch

from analyzer import PatchAnalyzer
from config import METRICS_CONFIG
from globals import DEVICE
from processor import ImageProcessor
from ui_helpers import (
    _redraw_metric_image,
    calculate_vector_distance,
    find_unit_index_from_click,
)


def on_unit_click(metric_name, evt: gr.SelectData, state, vec_a, vec_b):
    """
    Handles click on the result image.
    1. Populates Vector A or Vector B and immediately calculates distance.
    2. Shows the clicked unit and its neighbors in the Unit Inspector.
    3. Highlights the clicked unit on the result image.
    """
    new_vec_a, new_vec_b = vec_a, vec_b
    gallery_update = gr.update()
    image_update = gr.update()  # For the specific metric's image

    if not (state and metric_name in state):
        return (
            new_vec_a,
            new_vec_b,
            calculate_vector_distance(new_vec_a, new_vec_b),
            gallery_update,
            image_update,
        )

    data = state[metric_name]
    idx = find_unit_index_from_click(evt.index[0], evt.index[1], data)

    # If click is invalid, do nothing.
    if idx < 0 or idx >= len(data["matrix"]):
        return (
            vec_a,
            vec_b,
            calculate_vector_distance(vec_a, vec_b),
            gallery_update,
            image_update,
        )

    # 1. Update selection state (toggle)
    current_selected = data.get("selected_unit_idx", -1)
    if current_selected == idx:
        data["selected_unit_idx"] = -1  # Deselect
    else:
        data["selected_unit_idx"] = idx  # Select

    # 2. Vector Calculator Logic
    matrix = data["matrix"]
    vector = matrix[idx]
    vector = np.nan_to_num(vector, nan=0.0)
    vec_str = ", ".join([f"{x:.4f}" for x in vector])

    if not vec_a:
        new_vec_a = vec_str
    elif not vec_b:
        new_vec_b = vec_str
    else:
        new_vec_b = vec_str  # Overwrite B if A and B are full

    # 3. Unit Inspector Logic (Copied from original, unchanged)
    gallery_images = []
    rows, cols = data["grid_shape"]
    stride_h, stride_w = data["strides"]
    unit_h, unit_w = data["unit_size"]
    img_np_chw = state["image_tensor_np"].squeeze(0)
    img_np_hwc = np.transpose(img_np_chw, (1, 2, 0))
    img_np_hwc = (img_np_hwc * 255).clip(0, 255).astype(np.uint8)
    H, W, _ = img_np_hwc.shape
    clicked_r, clicked_c = divmod(idx, cols)
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            r, c = clicked_r + dr, clicked_c + dc
            if 0 <= r < rows and 0 <= c < cols:
                y = H - unit_h if r == rows - 1 else r * stride_h
                x = W - unit_w if c == cols - 1 else c * stride_w
                patch = img_np_hwc[y : y + unit_h, x : x + unit_w, :].copy()
                if dr == 0 and dc == 0:
                    cv2.rectangle(
                        patch, (0, 0), (unit_w - 1, unit_h - 1), (255, 255, 0), 2
                    )
                gallery_images.append(patch)
            else:
                placeholder = np.zeros((unit_h, unit_w, 3), dtype=np.uint8)
                gallery_images.append(placeholder)
    gallery_update = gr.update(value=gallery_images)

    # 4. Redraw image with highlight
    image_tensor = torch.from_numpy(state["image_tensor_np"]).to(DEVICE)
    overlay_visible = state.get("overlay_visible", True)

    result_img = _redraw_metric_image(
        image_tensor,
        data,
        data["selected_unit_idx"],
        show_overlay=overlay_visible,
    )
    image_update = gr.update(value=result_img)

    return (
        new_vec_a,
        new_vec_b,
        calculate_vector_distance(new_vec_a, new_vec_b),
        gallery_update,
        image_update,
    )


def create_click_handler(metric_name):
    """
    Creates a closure for the click handler to avoid partial introspection issues
    in Gradio. Captures metric_name.
    """

    def handler(evt: gr.SelectData, state, vec_a, vec_b):
        return on_unit_click(metric_name, evt, state, vec_a, vec_b)

    return handler


def toggle_annotations(state):
    """
    Toggles the visibility of annotations on the result images.
    Redraws the images based on the 'overlay_visible' flag in the state.
    """
    if not state or "image_tensor_np" not in state:
        # Return updates to do nothing if analysis hasn't run
        return tuple([gr.update()] * len(METRICS_CONFIG) + [state])

    # Toggle visibility state
    overlay_visible = not state.get("overlay_visible", True)
    state["overlay_visible"] = overlay_visible

    # --- Prepare for redrawing ---
    image_tensor = torch.from_numpy(state["image_tensor_np"]).to(DEVICE)

    # Get raw image as displayable numpy array
    img_tensor_for_display = image_tensor
    if img_tensor_for_display.dim() == 4:
        img_tensor_for_display = img_tensor_for_display.squeeze(0)
    img_np_rgb = img_tensor_for_display.permute(1, 2, 0).cpu().numpy()
    img_np_rgb = (img_np_rgb * 255).clip(0, 255).astype(np.uint8)

    image_outputs = []
    image_tensor = torch.from_numpy(state["image_tensor_np"]).to(DEVICE)

    for name, _ in METRICS_CONFIG:
        if name not in state:
            image_outputs.append(gr.update())
            continue

        # --- Redraw annotations ---
        metric_data = state[name]
        selected_idx = metric_data.get("selected_unit_idx", -1)

        result_img = _redraw_metric_image(
            image_tensor,
            metric_data,
            selected_idx,
            show_overlay=overlay_visible,
        )

        image_outputs.append(gr.update(value=result_img))

    return tuple(image_outputs + [state])


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
        "Clustering (DBSCAN2)": "clustering_dbscan2",
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
        processor = ImageProcessor(DEVICE)

        # Pipeline
        # 1. Load
        image_tensor = processor.load_image(image_path)
        img_h, img_w = image_tensor.shape[2], image_tensor.shape[3]

        # Store data needed for toggling annotations
        new_state["image_tensor_np"] = image_tensor.cpu().numpy()
        new_state["overlay_visible"] = True

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
            "clustering_dbscan2",
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
                metric = MetricClass()
            else:
                metric = MetricClass()

            analyzer = PatchAnalyzer(metric)

            # If clustering2, we do clustering inside analyze on the matrix
            do_matrix_cluster = action_mode in [
                "clustering2",
                "clustering_hierarchical",
                "clustering_spectral",
                "clustering_dbscan",
                "clustering_dbscan2",
            ]
            algo = "kmeans"
            if action_mode == "clustering_hierarchical":
                algo = "hierarchical"
            elif action_mode == "clustering_spectral":
                algo = "spectral"
            elif action_mode == "clustering_dbscan":
                algo = "dbscan"
            elif action_mode == "clustering_dbscan2":
                algo = "dbscan2"

            # For hierarchical, we ignore k_clusters input and let analyzer decide (pass 0)
            k_val = 0 if action_mode == "clustering_hierarchical" else int(k_clusters)

            stats, matrix, calculated_eps = analyzer.analyze(
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

            # Store Data in State for this metric
            new_state[name] = {
                "matrix": matrix.detach().cpu().numpy(),
                "grid_shape": grid_shape,
                "strides": strides,
                "unit_size": (int(height), int(width)),
                "img_shape": (img_h, img_w),
                "stats": stats,
                "action_mode": action_mode,
                "cluster_show_scores": cluster_show_scores,
                "sort_by": sort_by,
                "selected_unit_idx": -1,
            }

            # Generate Result Image based on Mode
            result_img = _redraw_metric_image(image_tensor, new_state[name])

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
            if action_mode.startswith("clustering_dbscan") and float(dbscan_eps) <= 0.0:
                perf_text += f" | Auto-Eps: {calculated_eps:.4f}"

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


def run_and_plot_distribution(
    image_path,
    height,
    width,
    overlap,
    metric_name,
    power_transform_degree,
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
        processor = ImageProcessor(DEVICE)
        MetricClass = dict(METRICS_CONFIG)[metric_name]
        if metric_name == "Oklab":
            metric = MetricClass()
        else:
            metric = MetricClass()
        analyzer = PatchAnalyzer(metric)

        # Run a silent analysis
        image_tensor = processor.load_image(image_path)
        patches, grid_shape, _ = processor.extract_patches(
            image_tensor, int(height), int(width), float(overlap)
        )

        # Get stats for all units
        stats, matrix, _ = analyzer.analyze(
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
