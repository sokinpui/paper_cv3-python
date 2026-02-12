import os
import tempfile
import time

import gradio as gr
import numpy as np
import torch

from analyzer import PatchAnalyzer
from clustering import find_dbscan_eps, get_k_distances
from config import METRICS_CONFIG
from globals import DEVICE
from processor import ImageProcessor
from ui_helpers import _redraw_metric_image  # noqa
from ui_helpers import find_unit_index_from_click


def on_unit_click(metric_name, evt: gr.SelectData, state):
    """
    Handles click on the result image.
    Highlights the clicked unit on the result image.
    """
    image_update = gr.update()  # For the specific metric's image

    if not (state and metric_name in state):
        return image_update

    data = state[metric_name]
    idx = find_unit_index_from_click(evt.index[0], evt.index[1], data)

    # If click is invalid, do nothing.
    if idx < 0 or idx >= len(data["matrix"]):
        return image_update

    # 1. Update selection state (toggle)
    current_selected = data.get("selected_unit_idx", -1)
    if current_selected == idx:
        data["selected_unit_idx"] = -1  # Deselect
    else:
        data["selected_unit_idx"] = idx  # Select

    # 2. Redraw image with highlight
    image_tensor = torch.from_numpy(state["image_tensor_np"]).to(DEVICE)
    overlay_visible = state.get("overlay_visible", True)

    result_img = _redraw_metric_image(
        image_tensor,
        data,
        data["selected_unit_idx"],
        show_overlay=overlay_visible,
    )
    image_update = gr.update(value=result_img)

    return image_update


def create_click_handler(metric_name):
    """
    Creates a closure for the click handler to avoid partial introspection issues
    in Gradio. Captures metric_name.
    """

    def handler(evt: gr.SelectData, state):
        return on_unit_click(metric_name, evt, state)

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
    ssim_k1,
    ssim_k2,
    ssim_alpha,
    ssim_beta,
    oklab_threshold,
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

    # Initialize output structure: [Img, Perf] per metric
    num_metrics = len(METRICS_CONFIG)
    state_idx = num_metrics * 3

    current_outputs = [gr.update(visible=False)] * (num_metrics * 3) + [current_state]

    if image_path is None:
        gr.Warning("Please upload an image.")
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
                    weights=(1.0, 1.0, 1.0),
                    threshold=float(oklab_threshold),
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

            current_outputs[state_idx] = new_state

            # Yield current state
            yield tuple(current_outputs)

    except Exception as e:
        import traceback

        traceback.print_exc()
        gr.Error(f"Analysis failed: {str(e)}")
        yield tuple(current_outputs)
