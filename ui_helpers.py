import cv2
import gradio as gr
import numpy as np
import torch

import visualizer


def generate_equation_markdown(
    selected_metrics,
    sigmoid_k,
    power_transform,
    dbscan_eps,
    dbscan_min_samples,
    oklab_blur,
    oklab_p,
    oklab_wl,
    oklab_wa,
    oklab_wb,
    ssim_k1,
    ssim_k2,
    ssim_alpha,
    ssim_beta,
):
    """
    Generates a Markdown string with LaTeX equations representing the current
    configuration of the app.
    """
    lines = []
    lines.append("### 🧮 Equation Visualizer")

    # 1. Distance Transform
    lines.append("**1. Distance & Transformation**")

    transform_tex = "\\text{Final } D_{ij} = "
    inner = "D_{ij}"

    if sigmoid_k > 0:
        # k is applied to (x - mu). mu is dynamic, we represent it as \mu
        k_val = f"{sigmoid_k:.1f}"
        inner = f"\\frac{{1}}{{1 + e^{{-{k_val}(D_{{ij}} - \\mu)}}}}"
        lines.append(
            f"*Sigmoid Contrast ($ k={k_val} $) applied around mean distance $ \\mu $.*"
        )

    if power_transform != 1.0:
        p_val = f"{power_transform:.1f}"
        transform_tex += f"\\left( {inner} \\right)^{{{p_val}}}"
    else:
        transform_tex += inner

    lines.append(f"$$ {transform_tex} $$")

    # 2. Clustering
    lines.append("**2. Anomaly Detection (DBSCAN)**")
    eps_val = f"{dbscan_eps:.3f}" if dbscan_eps > 0 else "\\text{auto}"
    min_samp = int(dbscan_min_samples)
    lines.append(
        f"$$ \\sum_{{j}} \\mathbb{{I}}(D_{{ij}} \\le {eps_val}) \\ge {min_samp} \\rightarrow \\text{{Cluster}} $$"
    )
    lines.append("*Points with fewer neighbors are considered anomalies (Noise).*")

    # 3. Metrics
    if not selected_metrics:
        return "\n\n".join(lines)

    lines.append("**3. Metric Logic**")
    for m in selected_metrics:
        if m == "Oklab":
            w_vec = f"[{oklab_wl:.1f}, {oklab_wa:.1f}, {oklab_wb:.1f}]"
            sigma = f"{oklab_blur:.1f}"
            p = f"{oklab_p:.1f}"
            lines.append("**Oklab**")
            params = f"\\quad \\text{{with }}\\sigma={sigma}, p={p}, \\mathbf{{W}}={w_vec}"
            eq = f"D_{{ij}} = \\left\\| G_{{{sigma}}}(\\mathbf{{W}} \\odot \\Phi(P_i)) - G_{{{sigma}}}(\\mathbf{{W}} \\odot \\Phi(P_j)) \\right\\|_{{{p}}} {params}"
            lines.append(f"$$ {eq} $$")
            lines.append(
                "*Where:*\n"
                "- $ P_i, P_j $: The i-th and j-th image units (patches).\n"
                "- $ \\Phi(P) $: Converts patch $ P $ from sRGB to the Oklab color space.\n"
                "- $ G_\\sigma $: Applies a Gaussian blur with standard deviation $ \\sigma $ to reduce noise."
            )
        elif m == "SSIM":
            alpha = f"{ssim_alpha:.2f}"
            beta = f"{ssim_beta:.2f}"
            lines.append("**SSIM**")
            params = f"\\quad \\text{{with }}\\alpha={alpha}, \\beta={beta}"
            eq = f"D_{{ij}} = 1 - (\\mathcal{{L}})^{{{alpha}}} \\cdot (\\mathcal{{C}}\\mathcal{{S}})^{{{beta}}} {params}"
            lines.append(f"$$ {eq} $$")
            lines.append(
                "*Where:*\n"
                "- $ \\mathcal{L} $: Luminance comparison, measuring brightness similarity.\n"
                "- $ \\mathcal{CS} $: Contrast-Structure comparison, measuring texture similarity."
            )
        elif m == "Cosine":
            lines.append("**Cosine**")
            lines.append(
                "$$ D_{ij} = 1 - \\frac{P_i \\cdot P_j}{\\|P_i\\| \\|P_j\\|} $$"
            )
            lines.append(
                "*Where $ P_i $ and $ P_j $ are the image units flattened into 1D vectors.*"
            )

    return "\n\n".join(lines)


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
