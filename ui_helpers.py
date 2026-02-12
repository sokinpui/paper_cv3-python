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


def _redraw_metric_image(image_tensor, metric_data, selected_unit_idx=-1):
    """Helper to redraw a single metric's result image."""
    stats = metric_data["stats"]
    grid_shape = metric_data["grid_shape"]
    strides = metric_data["strides"]
    height, width = metric_data["unit_size"]
    label_mode = metric_data.get("cluster_label_mode", "1-NN Distance")

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
