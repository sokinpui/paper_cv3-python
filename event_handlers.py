import json
import os
import tempfile
import time

import cv2
import gradio as gr
import numpy as np
import torch

from analyzer import PatchAnalyzer
from clustering import find_dbscan_eps, get_k_distances
from config import METRICS_CONFIG
from globals import DEVICE
from processor import ImageProcessor
from ui_helpers import _redraw_metric_image  # noqa
from ui_helpers import calculate_vector_distance, find_unit_index_from_click


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
    stats = data.get("stats", [])
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

    # Determine which vector to populate and store indices
    if not vec_a:
        new_vec_a = vec_str
        data["vec_a_idx"] = idx
        if "vec_b_idx" in data:
            del data["vec_b_idx"]
    elif not vec_b:
        new_vec_b = vec_str
        data["vec_b_idx"] = idx
    else:
        new_vec_b = vec_str  # Overwrite B if A and B are full
        data["vec_b_idx"] = idx

    # Generate result string
    distance_result = calculate_vector_distance(new_vec_a, new_vec_b)

    # Prepend Clicked Unit Stats (NN Dist, etc.)
    # FIX: stats list is sorted, so stats[idx] is NOT the unit at index idx.
    # We must find the unit with .index == idx
    u = next((s for s in stats if s.index == idx), None)

    if u:
        stat_info = f"--- Selected Unit #{idx + 1} ---\n"
        stat_info += f"Cluster ID: {u.cluster_id}\n"
        if hasattr(u, "nn_dist"):
            stat_info += f"1-NN Dist:  {u.nn_dist:.4f}\n"
        if hasattr(u, "neighbor_dist"):
            stat_info += f"k-Dist:     {u.neighbor_dist:.4f}\n"
        stat_info += f"Mean:       {u.mean:.4f}\n\n"
        distance_result = stat_info + distance_result

    # Add pairwise distance if available from click selections
    idx_a = data.get("vec_a_idx")
    idx_b = data.get("vec_b_idx")

    if idx_a is not None and idx_b is not None:
        pairwise_dist = matrix[idx_a, idx_b]
        dist_info = f"\n\n--- Pairwise Distance (from matrix) ---\n"
        dist_info += f"Distance(unit {idx_a}, unit {idx_b}): {pairwise_dist:.6f}"
        distance_result += dist_info

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
        distance_result,
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


def _redraw_with_updated_settings(state, cluster_show_scores, cluster_label_mode):
    """
    Helper: Updates state with latest UI params and redraws images.
    """
    if not state or "image_tensor_np" not in state:
        # Return updates to do nothing if analysis hasn't run
        return tuple([gr.update()] * len(METRICS_CONFIG) + [state])

    overlay_visible = state.get("overlay_visible", True)

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
        # Update display params in state
        if "cluster_show_scores" in state[name]:
            state[name]["cluster_show_scores"] = cluster_show_scores
        if "cluster_label_mode" in state[name]:
            state[name]["cluster_label_mode"] = cluster_label_mode

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


def toggle_annotations(state, cluster_show_scores, cluster_label_mode):
    """
    Toggles the visibility of annotations on the result images.
    Also updates the display settings from UI.
    """
    if not state:
        return tuple([gr.update()] * len(METRICS_CONFIG) + [state])

    # Toggle visibility state
    state["overlay_visible"] = not state.get("overlay_visible", True)

    return _redraw_with_updated_settings(state, cluster_show_scores, cluster_label_mode)


def update_annotation_settings(state, cluster_show_scores, cluster_label_mode):
    """
    Redraws the images with updated settings (e.g. label mode) without toggling visibility.
    """
    if not state:
        return tuple([gr.update()] * len(METRICS_CONFIG) + [state])
    return _redraw_with_updated_settings(state, cluster_show_scores, cluster_label_mode)


def run_analysis(
    image_path,
    height,
    width,
    overlap,
    action_mode_ui,
    cluster_show_scores,
    cluster_label_mode,
    selected_distance_functions,
    dbscan_eps,
    dbscan_min_samples,
    power_transform_degree,
    sigmoid_k,
    ssim_k1,
    ssim_k2,
    oklab_blur_sigma,
    oklab_w_l,
    oklab_w_a,
    oklab_w_b,
    oklab_p_norm,
    ssim_alpha,
    ssim_beta,
    current_state,
):
    """
    The core function called when user clicks 'Run Detection'
    action_mode: 'top_n', 'all', 'heatmap', 'clustering'
    action_mode_ui: 'Top N', 'All Units', 'Heatmap', 'Clustering', 'Clustering (K-means)'
    """
    # Map UI string to internal mode
    mode_map = {
        "Clustering (DBSCAN)": "clustering_dbscan",
        "Clustering (DBSCAN2)": "clustering_dbscan2",
    }
    action_mode = mode_map.get(action_mode_ui, "clustering_dbscan")

    # Initialize output structure: [Img, Perf] per metric + [JSON]
    num_metrics = len(METRICS_CONFIG)
    json_idx = num_metrics * 3
    state_idx = num_metrics * 3 + 1

    # Fill with None/Empty strings
    # Structure: [Header, Image, Perf] per metric
    # + [JSON] + [State]
    current_outputs = [gr.update(visible=False)] * (num_metrics * 3) + [
        "",
        current_state,
    ] + [gr.update(value=None)] * num_metrics

    if image_path is None:
        current_outputs[json_idx] = "Please upload an image."
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

        actual_top_n = 999999

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
            if name == "SSIM":
                metric = MetricClass(
                    k1=float(ssim_k1),
                    k2=float(ssim_k2),
                    alpha=float(ssim_alpha),
                    beta=float(ssim_beta),
                )
            elif name == "Oklab":
                metric = MetricClass(
                    blur_sigma=float(oklab_blur_sigma),
                    weights=(float(oklab_w_l), float(oklab_w_a), float(oklab_w_b)),
                    p_norm=float(oklab_p_norm),
                )
            else:
                metric = MetricClass()

            analyzer = PatchAnalyzer(metric)

            if action_mode == "clustering_dbscan":
                algo = "dbscan"
            elif action_mode == "clustering_dbscan2":
                algo = "dbscan2"
            else:
                # Should not happen with the current UI
                algo = "dbscan"

            stats, matrix, calculated_eps = analyzer.analyze(
                patches,
                grid_shape,
                top_n=actual_top_n,
                sort_by="mean",
                ascending=False,
                cluster_on_matrix=True,
                clustering_algorithm=algo,
                eps=float(dbscan_eps),
                min_samples=int(dbscan_min_samples),
                power_transform_degree=float(power_transform_degree),
                sigmoid_k=float(sigmoid_k),
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
                "cluster_label_mode": cluster_label_mode,
                "sort_by": "mean",
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
            current_outputs[json_idx] = json.dumps(
                all_stats_collection[:actual_top_n], indent=4
            )
            current_outputs[state_idx] = new_state

            # Yield current state
            yield tuple(current_outputs)

    except Exception as e:
        import traceback

        traceback.print_exc()
        # Yield error in the JSON field
        current_outputs[json_idx] = f"Error: {str(e)}"
        yield tuple(current_outputs)


def run_and_plot_k_distance(
    image_path,
    height,
    width,
    overlap,
    metric_name,
    power_transform_degree,
    sigmoid_k,
    min_samples,
    eps,
    ssim_k1,
    ssim_k2,
    oklab_blur_sigma,
    oklab_w_l,
    oklab_w_a,
    oklab_w_b,
    oklab_p_norm,
    ssim_alpha,
    ssim_beta,
):
    """
    Performs a dedicated analysis and generates the K-Distance Graph.
    """
    if not image_path or not metric_name or int(min_samples) < 2:
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
        if metric_name == "SSIM":
            metric = MetricClass(
                k1=float(ssim_k1),
                k2=float(ssim_k2),
                alpha=float(ssim_alpha),
                beta=float(ssim_beta),
            )
        elif metric_name == "Oklab":
            metric = MetricClass(
                blur_sigma=float(oklab_blur_sigma),
                weights=(float(oklab_w_l), float(oklab_w_a), float(oklab_w_b)),
                p_norm=float(oklab_p_norm),
            )
        else:
            metric = MetricClass()
        analyzer = PatchAnalyzer(metric)

        # Run a silent analysis
        image_tensor = processor.load_image(image_path)
        patches, grid_shape, _ = processor.extract_patches(
            image_tensor, int(height), int(width), float(overlap)
        )

        # 1. Compute Matrix with Transformations
        matrix = analyzer.compute_distance_matrix(
            patches, float(power_transform_degree), float(sigmoid_k)
        )

        N = matrix.shape[0]
        if N < int(min_samples):
            return None

        # 2. Calculate K-Distances
        k = max(1, int(min_samples) - 1)
        k_distances = get_k_distances(matrix, k)

        # 3. Auto-Determine eps if user wants it (for insight)
        calculated_eps = float(eps)
        if calculated_eps <= 0.0:
            # Only calculate if the matrix is large enough for elbow detection to be meaningful
            if N >= int(min_samples):
                calculated_eps = find_dbscan_eps(matrix, int(min_samples))
            else:
                calculated_eps = 0.0

        # Plotting logic
        fig, ax = plt.subplots(figsize=(8, 4))
        x_coords = np.arange(N)

        # The Curve: Blue line
        ax.plot(x_coords, k_distances, label=f"{k}-Distance", color="blue")

        # The Threshold: Red dashed line
        eps_to_plot = calculated_eps if calculated_eps > 0.0 else float(eps)

        if eps_to_plot > 0.0:
            ax.axhline(
                y=eps_to_plot,
                color="red",
                linestyle="--",
                label=f"Eps Threshold ({eps_to_plot:.4f})",
            )

        ax.set_title(f"K-Distance Graph (k={k}) for '{metric_name}'")
        ax.set_xlabel(f"Unit Index (Sorted by {k}-Distance)")
        ax.set_ylabel(f"Distance to {k}-th Nearest Neighbor")
        ax.grid(axis="y", linestyle="--", alpha=0.7)
        ax.legend()
        plt.tight_layout()

        return fig

    except Exception as e:
        print(f"Error during plotting analysis: {e}")
        import traceback

        traceback.print_exc()
        return None

def download_single_result(state, metric_name):
    """
    Combines the input image and a single visible result image into a
    side-by-side downloadable JPG image.
    """
    from visualizer import create_side_by_side_image

    if not state or "image_tensor_np" not in state or metric_name not in state:
        return None

    # 1. Get Input Image from state and convert to displayable numpy
    input_img_np = state["image_tensor_np"]
    if input_img_np.shape[0] == 1:  # Remove batch dim if present
        input_img_np = input_img_np.squeeze(0)
    input_img_np_rgb = np.transpose(input_img_np, (1, 2, 0))
    input_img_np_rgb = (input_img_np_rgb * 255).clip(0, 255).astype(np.uint8)

    # 2. Regenerate the specific result image from state
    image_tensor = torch.from_numpy(state["image_tensor_np"]).to(DEVICE)
    overlay_visible = state.get("overlay_visible", True)

    metric_data = state[metric_name]

    # Regenerate the image using the same function as other UI updates
    result_img = _redraw_metric_image(
        image_tensor,
        metric_data,
        metric_data.get("selected_unit_idx", -1),
        show_overlay=overlay_visible,
    )

    # 3. Stitch input and result side-by-side
    final_image = create_side_by_side_image(input_img_np_rgb, result_img)

    # 4. Save to a temporary JPG file
    if not os.path.exists("tmp"):
        os.makedirs("tmp")

    _, temp_path = tempfile.mkstemp(suffix=".jpg", dir="tmp")

    final_image_bgr = cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(temp_path, final_image_bgr)

    return temp_path

def create_download_handler(metric_name):
    """Closure to capture metric_name for the download handler."""

    def handler(state):
        return download_single_result(state, metric_name)

    return handler

def run_and_plot_stats(
    image_path,
    height,
    width,
    overlap,
    metric_name,
    power_transform_degree,
    sigmoid_k,
    min_samples,
    ssim_k1,
    ssim_k2,
    oklab_blur_sigma,
    oklab_w_l,
    oklab_w_a,
    oklab_w_b,
    oklab_p_norm,
    ssim_alpha,
    ssim_beta,
):
    """
    Performs a dedicated analysis and generates plots for key unit statistics.
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
        if metric_name == "SSIM":
            metric = MetricClass(
                k1=float(ssim_k1),
                k2=float(ssim_k2),
                alpha=float(ssim_alpha),
                beta=float(ssim_beta),
            )
        elif metric_name == "Oklab":
            metric = MetricClass(
                blur_sigma=float(oklab_blur_sigma),
                weights=(float(oklab_w_l), float(oklab_w_a), float(oklab_w_b)),
                p_norm=float(oklab_p_norm),
            )
        else:
            metric = MetricClass()
        analyzer = PatchAnalyzer(metric)

        # Run a silent analysis
        image_tensor = processor.load_image(image_path)
        patches, grid_shape, _ = processor.extract_patches(
            image_tensor, int(height), int(width), float(overlap)
        )

        N = patches.shape[0]
        if N < 2:
            return None

        # Get all stats, unsorted
        stats, _, _ = analyzer.analyze(
            patches,
            grid_shape,
            top_n=N,  # Get all stats
            sort_by="mean",  # doesn't matter, we re-sort
            ascending=True,
            min_samples=int(min_samples),
            power_transform_degree=float(power_transform_degree),
            sigmoid_k=float(sigmoid_k),
        )

        # Re-sort by index to get Z-order
        stats.sort(key=lambda s: s.index)

        # Extract data for plotting
        indices = np.array([s.index for s in stats])
        nn_dists = np.array([s.nn_dist for s in stats])
        k_dists = np.array([s.neighbor_dist for s in stats])
        mean_scores = np.array([s.mean for s in stats])
        max_scores = np.array([s.max_score for s in stats])

        # Plotting logic
        fig, axs = plt.subplots(5, 1, figsize=(10, 15), sharex=True)
        fig.suptitle(f"Unit Statistics for '{metric_name}' (Z-Order)", fontsize=16)

        axs[0].plot(indices, nn_dists, label="1-NN Distance", color="blue")
        axs[0].set_ylabel("1-NN Distance")
        axs[0].grid(True, linestyle="--", alpha=0.6)
        axs[0].legend(loc="upper left")

        k_val = max(1, int(min_samples) - 1)
        axs[1].plot(
            indices, k_dists, label=f"k-Distance (k={k_val})", color="green"
        )
        axs[1].set_ylabel("k-Distance")
        axs[1].grid(True, linestyle="--", alpha=0.6)
        axs[1].legend(loc="upper left")

        axs[2].plot(indices, mean_scores, label="Mean Score", color="red")
        axs[2].set_ylabel("Mean Score")
        axs[2].grid(True, linestyle="--", alpha=0.6)
        axs[2].legend(loc="upper left")

        axs[3].plot(indices, max_scores, label="Max Score", color="purple")
        axs[3].set_ylabel("Max Score")
        axs[3].grid(True, linestyle="--", alpha=0.6)
        axs[3].legend(loc="upper left")

        # --- New Combined Plot ---
        def normalize(arr):
            min_val = arr.min()
            max_val = arr.max()
            if np.isclose(max_val, min_val):
                return np.zeros_like(arr)
            return (arr - min_val) / (max_val - min_val)

        axs[4].set_title("Combined Normalized Statistics")
        axs[4].plot(
            indices,
            normalize(nn_dists),
            label="1-NN Dist (Norm)",
            color="blue",
            alpha=0.8,
        )
        axs[4].plot(
            indices,
            normalize(k_dists),
            label=f"k-Dist (k={k_val}) (Norm)",
            color="green",
            alpha=0.8,
        )
        axs[4].plot(
            indices,
            normalize(mean_scores),
            label="Mean (Norm)",
            color="red",
            alpha=0.8,
        )
        axs[4].plot(
            indices,
            normalize(max_scores),
            label="Max (Norm)",
            color="purple",
            alpha=0.8,
        )
        axs[4].set_ylabel("Normalized Value")
        axs[4].set_xlabel("Unit Index (Z-Order)")
        axs[4].grid(True, linestyle="--", alpha=0.6)
        axs[4].legend(loc="upper left")

        plt.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust for suptitle

        return fig

    except Exception as e:
        print(f"Error during plotting analysis: {e}")
        import traceback

        traceback.print_exc()
        return None


def plot_sigmoid_function(k: float):
    """
    Generates a plot of the sigmoid function based on the k value.
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print(
            "Error: Matplotlib/Numpy is required. Please install it: pip install matplotlib numpy"
        )
        return None

    k = float(k)
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.linspace(0, 1, 200)

    if k <= 0:
        ax.plot(
            [0, 1], [0, 1], label="Sigmoid Disabled (k=0)", color="gray", linestyle="--"
        )
        ax.set_title("Sigmoid Function (Disabled)")
    else:
        mu = 0.5
        y = 1.0 / (1.0 + np.exp(-k * (x - mu)))
        ax.plot(x, y, label=f"k = {k:.2f}")
        ax.set_title("Sigmoid Function Shape")

    ax.set_xlabel("Input Value (Normalized Distance)")
    ax.set_ylabel("Output Value (Stretched Distance)")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()
    plt.tight_layout()

    return fig
