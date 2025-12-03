from typing import Tuple

import cv2
import numpy as np
import torch


def _draw_annotations(
    img: np.ndarray,
    units: list,
    unit_h: int,
    unit_w: int,
    grid_shape: Tuple[int, int],
    strides: Tuple[int, int],
    is_bgr: bool = False,
    selected_unit_index: int = -1,
):
    """
    Helper to draw individual rectangles and labels for units.
    """
    box_color = (0, 255, 0)  # Green
    selected_box_color = (255, 255, 0)  # Yellow
    text_color = (0, 0, 255) if is_bgr else (255, 0, 0)  # Red

    rows, cols = grid_shape
    stride_h, stride_w = strides
    H, W = img.shape[:2]

    for i, unit in enumerate(units):
        r, c = unit.row, unit.col

        # Calculate coords:
        # If it's the last row/col, it is back-shifted to align with edge.
        # Otherwise it follows the stride.
        if r == rows - 1:
            y = H - unit_h
        else:
            y = r * stride_h

        if c == cols - 1:
            x = W - unit_w
        else:
            x = c * stride_w

        # Draw Rectangle (Individual)
        is_selected = unit.index == selected_unit_index
        color = selected_box_color if is_selected else box_color
        thickness = 4 if is_selected else 2
        cv2.rectangle(img, (x, y), (x + unit_w, y + unit_h), color, thickness)

        label = f"#{i+1}"
        cv2.putText(
            img,
            label,
            (x + 5, y + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            text_color,
            2,
        )


def save_annotated_image(
    image: torch.Tensor,
    units: list,
    unit_h: int,
    unit_w: int,
    grid_shape: Tuple[int, int],
    strides: Tuple[int, int],
    output_path: str,
):
    """
    Draws rectangles around the specified units and saves the image.
    """
    # Ensure image is (C, H, W)
    if image.dim() == 4:
        image = image.squeeze(0)

    # Convert Tensor (C, H, W) to Numpy (H, W, C)
    # Detach from GPU, move to CPU, transform to numpy
    img_np = image.detach().permute(1, 2, 0).cpu().numpy()

    # Convert [0, 1] to [0, 255] BGR for OpenCV
    img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    _draw_annotations(img_np, units, unit_h, unit_w, grid_shape, strides, is_bgr=True)

    cv2.imwrite(output_path, img_np)
    print(f"Annotated image saved to {output_path}")


def get_annotated_rgb(
    image: torch.Tensor,
    units: list,
    unit_h: int,
    unit_w: int,
    grid_shape: Tuple[int, int],
    strides: Tuple[int, int],
    selected_unit_index: int = -1,
) -> np.ndarray:
    """
    Returns the annotated image as an RGB numpy array for Web UI display.
    """
    if image.dim() == 4:
        image = image.squeeze(0)

    # Tensor to Numpy (RGB)
    img_np = image.detach().permute(1, 2, 0).cpu().numpy()
    img_np = (img_np * 255).clip(0, 255).astype(np.uint8)

    # Make writable copy
    img_out = img_np.copy()

    _draw_annotations(
        img_out,
        units,
        unit_h,
        unit_w,
        grid_shape,
        strides,
        is_bgr=False,
        selected_unit_index=selected_unit_index,
    )

    return img_out


def create_heatmap(
    image: torch.Tensor,
    units: list,
    grid_shape: Tuple[int, int],
    strides: Tuple[int, int],
    unit_h: int,
    unit_w: int,
    stat_name: str = "mean",
    selected_unit_index: int = -1,
) -> np.ndarray:
    """
    Creates a heatmap overlay based on the 'mean' score of each unit.
    Colors range from Blue (Low) -> Green -> Red (High).
    """
    if image.dim() == 4:
        image = image.squeeze(0)

    # RGB Numpy
    img_np = image.detach().permute(1, 2, 0).cpu().numpy()
    img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
    H, W = img_np.shape[:2]

    overlay = img_np.copy()
    rows, cols = grid_shape
    stride_h, stride_w = strides

    # Extract scores for normalization
    # We assume units contains all units for the grid
    scores = [getattr(u, stat_name) for u in units]
    if not scores:
        return img_np

    min_s, max_s = min(scores), max(scores)
    rng = max_s - min_s if max_s != min_s else 1.0

    unit_map = {(u.row, u.col): u for u in units}

    for r in range(rows):
        for c in range(cols):
            if (r, c) not in unit_map:
                continue

            u = unit_map[(r, c)]
            val = getattr(u, stat_name)
            norm = (val - min_s) / rng

            # Color mapping: Blue (Low) -> Green -> Red (High)
            if norm < 0.5:
                # 0.0 (Blue) -> 0.5 (Green)
                n = norm * 2
                r_val, g_val, b_val = 0, int(255 * n), int(255 * (1 - n))
            else:
                # 0.5 (Green) -> 1.0 (Red)
                n = (norm - 0.5) * 2
                r_val, g_val, b_val = int(255 * n), int(255 * (1 - n)), 0

            if r == rows - 1:
                y = H - unit_h
            else:
                y = r * stride_h

            if c == cols - 1:
                x = W - unit_w
            else:
                x = c * stride_w

            # Draw filled rectangle
            cv2.rectangle(
                overlay, (x, y), (x + unit_w, y + unit_h), (r_val, g_val, b_val), -1
            )

    # Alpha blend
    alpha = 0.6
    cv2.addWeighted(overlay, alpha, img_np, 1 - alpha, 0, img_np)

    # Highlight selected unit on top of the heatmap
    if selected_unit_index >= 0:
        selected_unit = next((u for u in units if u.index == selected_unit_index), None)
        if selected_unit:
            r, c = selected_unit.row, selected_unit.col
            y = r * stride_h if r < rows - 1 else H - unit_h
            x = c * stride_w if c < cols - 1 else W - unit_w
            cv2.rectangle(
                img_np, (x, y), (x + unit_w, y + unit_h), (255, 255, 0), 4
            )  # Yellow, thick border

    # Draw borders for all grid cells
    border_color = (0, 0, 0)  # Black
    for r in range(rows):
        for c in range(cols):
            y = r * stride_h if r < rows - 1 else H - unit_h
            x = c * stride_w if c < cols - 1 else W - unit_w
            cv2.rectangle(img_np, (x, y), (x + unit_w, y + unit_h), border_color, 1)

    return img_np


def create_cluster_map(
    image: torch.Tensor,
    units: list,
    grid_shape: Tuple[int, int],
    strides: Tuple[int, int],
    unit_h: int,
    unit_w: int,
    show_scores: bool = False,
    selected_unit_index: int = -1,
    label_mode: str = "1-NN Distance",
) -> np.ndarray:
    """
    Creates a visualization where each unit is colored by its cluster_id.
    """
    if image.dim() == 4:
        image = image.squeeze(0)

    img_np = image.detach().permute(1, 2, 0).cpu().numpy()
    img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
    H, W = img_np.shape[:2]

    overlay = img_np.copy()
    rows, cols = grid_shape
    stride_h, stride_w = strides

    # Distinct colors for clusters (BGR)
    # K is usually small, define a palette
    palette = [
        (0, 255, 0),  # Green
        (0, 0, 255),  # Red
        (255, 0, 0),  # Blue
        (0, 255, 255),  # Yellow
        (255, 0, 255),  # Magenta
        (255, 255, 0),  # Cyan
        (128, 0, 128),  # Purple
        (128, 128, 0),  # Teal
    ]

    for u in units:
        if u.cluster_id < 0:
            continue

        color = palette[u.cluster_id % len(palette)]
        y = u.row * stride_h if u.row < rows - 1 else H - unit_h
        x = u.col * stride_w if u.col < cols - 1 else W - unit_w

        cv2.rectangle(overlay, (x, y), (x + unit_w, y + unit_h), color, -1)

    alpha = 0.5
    cv2.addWeighted(overlay, alpha, img_np, 1 - alpha, 0, img_np)

    # Highlight selected unit on top of the cluster map
    if selected_unit_index >= 0:
        selected_unit = next((u for u in units if u.index == selected_unit_index), None)
        if selected_unit:
            r, c = selected_unit.row, selected_unit.col
            y = r * stride_h if r < rows - 1 else H - unit_h
            x = c * stride_w if c < cols - 1 else W - unit_w
            cv2.rectangle(
                img_np, (x, y), (x + unit_w, y + unit_h), (255, 255, 0), 4
            )  # Yellow, thick border

    # Draw borders for all grid cells
    border_color = (0, 0, 0)  # Black
    for r in range(rows):
        for c in range(cols):
            y = r * stride_h if r < rows - 1 else H - unit_h
            x = c * stride_w if c < cols - 1 else W - unit_w
            cv2.rectangle(img_np, (x, y), (x + unit_w, y + unit_h), border_color, 1)

    # Draw scores on top of the blended image if requested
    if show_scores:
        for u in units:
            y = u.row * stride_h if u.row < rows - 1 else H - unit_h
            x = u.col * stride_w if u.col < cols - 1 else W - unit_w

            if label_mode == "1-NN Distance":
                val = u.nn_dist
            elif label_mode == "k-Distance":
                val = u.neighbor_dist
            elif label_mode == "Mean Score":
                val = u.mean
            elif label_mode == "Max Score":
                val = u.max_score
            else:  # Fallback to 1-NN Distance
                val = u.nn_dist

            text = f"{val:.3f}"
            font_scale = max(0.2, 0.4 * (min(unit_w, unit_h) / 50.0))
            thickness = max(1, int(min(unit_w, unit_h) / 50.0))

            (tw, th), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )

            if tw > unit_w * 0.9:
                font_scale *= (unit_w * 0.9) / tw

            (tw, th), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )

            tx = x + (unit_w - tw) // 2
            ty = y + (unit_h + th) // 2

            # Draw with outline for visibility
            cv2.putText(
                img_np,
                text,
                (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 0, 0),
                thickness + 2,
            )
            cv2.putText(
                img_np,
                text,
                (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                thickness,
            )

    return img_np


def create_composite_image(image_data: list) -> np.ndarray:
    """
    Stitches multiple images with headers into a single tall image.
    image_data: A list of tuples, where each tuple is (header_text, image_numpy_array).
    """
    if not image_data:
        # Return a blank placeholder image
        blank = np.full((200, 800, 3), 255, dtype=np.uint8)
        cv2.putText(
            blank,
            "No images to combine.",
            (50, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0),
            2,
        )
        return blank

    header_height = 60
    padding = 10
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.2
    font_thickness = 2
    header_text_color = (0, 0, 0)  # Black
    bg_color = (255, 255, 255)  # White

    # Assume all images have the same width
    total_width = image_data[0][1].shape[1]
    total_height = 0

    # Calculate total height
    for _, img in image_data:
        total_height += img.shape[0] + header_height + padding

    # Create canvas
    composite_image = np.full((total_height, total_width, 3), bg_color, dtype=np.uint8)

    current_y = 0
    for header, img in image_data:
        # --- Draw Header ---
        text_size, _ = cv2.getTextSize(header, font, font_scale, font_thickness)
        text_x = (total_width - text_size[0]) // 2
        text_y = current_y + (header_height + text_size[1]) // 2
        cv2.putText(
            composite_image,
            header,
            (text_x, text_y),
            font,
            font_scale,
            header_text_color,
            font_thickness,
        )
        current_y += header_height

        # --- Draw Image ---
        img_h, img_w, _ = img.shape
        composite_image[current_y : current_y + img_h, 0:img_w] = img
        current_y += img_h + padding

    return composite_image
