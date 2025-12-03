import cv2
import gradio as gr
import numpy as np
import torch

import visualizer


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


def _redraw_metric_image(
    image_tensor, metric_data, selected_unit_idx=-1, show_overlay=True
):
    """Helper to redraw a single metric's result image."""
    # If overlay is off, just return the raw image
    if not show_overlay:
        if image_tensor.dim() == 4:
            image_tensor = image_tensor.squeeze(0)
        img_np_rgb = image_tensor.permute(1, 2, 0).cpu().numpy()
        return (img_np_rgb * 255).clip(0, 255).astype(np.uint8)

    action_mode = metric_data["action_mode"]
    stats = metric_data["stats"]
    grid_shape = metric_data["grid_shape"]
    strides = metric_data["strides"]
    height, width = metric_data["unit_size"]
    label_mode = metric_data.get("cluster_label_mode", "1-NN Distance")

    if action_mode in [
        "clustering",
        "clustering2",
        "clustering_hierarchical",
        "clustering_spectral",
        "clustering_dbscan",
        "clustering_dbscan2",
    ]:
        return visualizer.create_cluster_map(
            image_tensor,
            stats,
            grid_shape,
            strides,
            height,
            width,
            show_scores=metric_data["cluster_show_scores"],
            selected_unit_index=selected_unit_idx,
            label_mode=label_mode,
        )
    elif action_mode == "heatmap":
        return visualizer.create_heatmap(
            image_tensor,
            stats,
            grid_shape,
            strides,
            height,
            width,
            stat_name=metric_data["sort_by"],
            selected_unit_index=selected_unit_idx,
        )
    else:  # 'top_n' or 'all'
        return visualizer.get_annotated_rgb(
            image_tensor,
            stats,
            height,
            width,
            grid_shape,
            strides,
            selected_unit_index=selected_unit_idx,
        )


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


def clear_vector_inputs():
    return "", "", ""
