import base64
import uuid
from typing import List, Tuple

import cv2
import numpy as np
import torch


class ImageProcessor:
    def __init__(self, device: torch.device):
        self.device = device

    def load_image(self, path: str) -> torch.Tensor:
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Could not load image at {path}")

        # BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0

        # To Tensor (C, H, W)
        tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self.device)

    def adjust_image(
        self, image: torch.Tensor, brightness: float, contrast: float
    ) -> torch.Tensor:
        """
        Adjusts brightness and contrast of the image tensor.
        image: (B, C, H, W) in [0, 1]
        brightness: offset [-1.0, 1.0], default 0.0
        contrast: multiplier [0.0, 3.0], default 1.0
        """
        if brightness == 0.0 and contrast == 1.0:
            return image

        # Apply contrast (centered at 0.5) and brightness
        image = (image - 0.5) * contrast + 0.5 + brightness
        return torch.clamp(image, 0.0, 1.0)

    def apply_preprocessing(
        self,
        image: torch.Tensor,
        blur_radius: float = 0.0,
        sharpen_factor: float = 0.0,
        clahe_limit: float = 0.0,
        grayscale: bool = False,
    ) -> torch.Tensor:
        """
        Applies advanced CV preprocessing for texture analysis.
        Includes: Grayscale, Gaussian Blur (Denoise), Unsharp Mask (Sharpen), CLAHE (Local Contrast).
        """
        if (
            blur_radius <= 0
            and sharpen_factor <= 0
            and clahe_limit <= 0
            and not grayscale
        ):
            return image

        # Move to CPU/Numpy for OpenCV operations
        # (N, C, H, W) -> (H, W, C)
        if image.dim() == 4:
            img_t = image.squeeze(0)
        else:
            img_t = image

        img_np = img_t.permute(1, 2, 0).cpu().numpy()
        # Convert to [0, 255] uint8 for OpenCV
        img_cv = (img_np * 255).clip(0, 255).astype(np.uint8)

        # 0. Grayscale
        if grayscale:
            img_gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
            img_cv = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)

        # 1. Gaussian Blur (Reduce Fabric Grain/Noise)
        if blur_radius > 0:
            # Kernel size must be odd
            ksize = int(blur_radius) * 2 + 1
            img_cv = cv2.GaussianBlur(img_cv, (ksize, ksize), 0)

        # 2. CLAHE (Local Contrast Enhancement)
        # Best applied on 'L' channel of LAB to preserve color correctness
        if clahe_limit > 0:
            lab = cv2.cvtColor(img_cv, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=clahe_limit, tileGridSize=(8, 8))
            l = clahe.apply(l)
            lab = cv2.merge((l, a, b))
            img_cv = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        # 3. Sharpening (Unsharp Mask)
        # Highlights edges (tears, cuts)
        if sharpen_factor > 0:
            # Create a blurred version
            gaussian = cv2.GaussianBlur(img_cv, (0, 0), 3.0)
            # Weighted add: Original + Strength * (Original - Blurred)
            img_cv = cv2.addWeighted(
                img_cv, 1.0 + sharpen_factor, gaussian, -sharpen_factor, 0
            )

        # Convert back to Tensor
        img_final = img_cv.astype(np.float32) / 255.0
        tensor = torch.from_numpy(img_final).permute(2, 0, 1)

        if image.dim() == 4:
            tensor = tensor.unsqueeze(0)

        return tensor.to(self.device)

    def extract_patches(
        self, image: torch.Tensor, unit_h: int, unit_w: int, overlap_ratio: float = 0.0
    ) -> Tuple[torch.Tensor, Tuple[int, int], Tuple[int, int]]:
        """
        Divides image into non-overlapping patches.
        Returns:
            patches: (N, C, unit_h, unit_w)
            grid_shape: (rows, cols)
            strides: (stride_h, stride_w)
        """
        B, C, H, W = image.shape

        if H < unit_h or W < unit_w:
            raise ValueError(
                f"Image size ({H}x{W}) is smaller than unit size ({unit_h}x{unit_w})"
            )

        # Calculate strides based on overlap
        stride_h = int(unit_h * (1.0 - overlap_ratio))
        stride_w = int(unit_w * (1.0 - overlap_ratio))

        # Ensure at least 1 pixel stride
        stride_h = max(1, stride_h)
        stride_w = max(1, stride_w)

        # Generate coordinates
        # We ensure the last patch ends exactly at the image edge (back-shifted if needed)
        y_coords = []
        y = 0
        while y + unit_h <= H:
            y_coords.append(y)
            y += stride_h
        if y_coords[-1] + unit_h < H:
            y_coords.append(H - unit_h)

        x_coords = []
        x = 0
        while x + unit_w <= W:
            x_coords.append(x)
            x += stride_w
        if x_coords[-1] + unit_w < W:
            x_coords.append(W - unit_w)

        patches_list = []
        for y in y_coords:
            for x in x_coords:
                patches_list.append(image[..., y : y + unit_h, x : x + unit_w])

        # Stack to (N, C, H, W) assuming B=1. If B>1, this flattens batches to N.
        patches = torch.cat(patches_list, dim=0)

        rows = len(y_coords)
        cols = len(x_coords)

        return patches, (rows, cols), (stride_h, stride_w)

    def _draw_annotations(
        self,
        img: np.ndarray,
        units: list,
        unit_h: int,
        unit_w: int,
        grid_shape: Tuple[int, int],
        strides: Tuple[int, int],
        is_bgr: bool = False,
    ):
        """
        Helper to draw individual rectangles and labels for units.
        """
        box_color = (0, 255, 0)  # Green
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
            cv2.rectangle(img, (x, y), (x + unit_w, y + unit_h), box_color, 2)

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
        self,
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

        self._draw_annotations(
            img_np, units, unit_h, unit_w, grid_shape, strides, is_bgr=True
        )

        cv2.imwrite(output_path, img_np)
        print(f"Annotated image saved to {output_path}")

    def get_annotated_rgb(
        self,
        image: torch.Tensor,
        units: list,
        unit_h: int,
        unit_w: int,
        grid_shape: Tuple[int, int],
        strides: Tuple[int, int],
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

        self._draw_annotations(
            img_out, units, unit_h, unit_w, grid_shape, strides, is_bgr=False
        )

        return img_out

    def create_heatmap(
        self,
        image: torch.Tensor,
        units: list,
        grid_shape: Tuple[int, int],
        strides: Tuple[int, int],
        unit_h: int,
        unit_w: int,
        stat_name: str = "mean",
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

        # Draw borders for all grid cells
        border_color = (0, 0, 0)  # Black
        for r in range(rows):
            for c in range(cols):
                y = r * stride_h if r < rows - 1 else H - unit_h
                x = c * stride_w if c < cols - 1 else W - unit_w
                cv2.rectangle(img_np, (x, y), (x + unit_w, y + unit_h), border_color, 1)

        return img_np

    def create_cluster_map(
        self,
        image: torch.Tensor,
        units: list,
        grid_shape: Tuple[int, int],
        strides: Tuple[int, int],
        unit_h: int,
        unit_w: int,
        show_scores: bool = False,
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
                if u.cluster_id < 0:
                    continue

                y = u.row * stride_h if u.row < rows - 1 else H - unit_h
                x = u.col * stride_w if u.col < cols - 1 else W - unit_w

                text = f"{u.mean:.5f}"
                font_scale = 0.8
                thickness = 2
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

    def create_interactive_html(
        self,
        image_rgb: np.ndarray,
        units: list,
        grid_shape: Tuple[int, int],
        strides: Tuple[int, int],
        unit_h: int,
        unit_w: int,
    ) -> str:
        """
        Wraps the image in an SVG with transparent rectangles for tooltips.
        """
        # Encode image to base64 PNG
        img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        success, buffer = cv2.imencode(".png", img_bgr)
        if not success:
            return "<div>Error encoding image</div>"

        img_b64 = base64.b64encode(buffer).decode("utf-8")

        H, W = image_rgb.shape[:2]
        rows, cols = grid_shape
        stride_h, stride_w = strides

        # Generate a unique ID for this block to scope the JS events
        block_id = str(uuid.uuid4())

        # SVG Header
        html_parts = [
            f'<div id="container-{block_id}" class="unit-analysis-container">'
        ]

        svg_parts = [
            f'<svg viewBox="0 0 {W} {H}" style="width: 100%; height: auto; cursor: crosshair;" xmlns="http://www.w3.org/2000/svg">'
        ]
        svg_parts.append(
            f'<image href="data:image/png;base64,{img_b64}" width="{W}" height="{H}" />'
        )

        for u in units:
            # Calculate coords
            y = H - unit_h if u.row == rows - 1 else u.row * stride_h
            x = W - unit_w if u.col == cols - 1 else u.col * stride_w

            # --- START MODIFICATION ---
            # Create full precision vector string for copying
            full_vec_str = ", ".join([str(v) for v in (u.vector or [])])

            # Build details HTML using double quotes for attributes
            details_html = (
                f"<b>Unit #{u.index}</b><br>" f"Position: Row {u.row}, Col {u.col}<br>"
            )
            if u.cluster_id != -1:
                details_html += f"Cluster ID: {u.cluster_id}<br>"

            vec_id = f"vec-{block_id}-{u.index}"

            # JS to copy text and provide feedback
            js_copy = (
                f"var t=document.getElementById('{vec_id}');"
                f"t.select();"
                f"if(navigator.clipboard){{"
                f"navigator.clipboard.writeText(t.value).then(()=>{{this.innerHTML='✅';setTimeout(()=>this.innerHTML='📋',1500);}});"
                f"}}else{{"
                f"document.execCommand('copy');this.innerHTML='✅';setTimeout(()=>this.innerHTML='📋',1500);"
                f"}}"
            )

            # Wrap vector in a textarea with select-on-click functionality
            details_html += (
                f"Vector (Distance to others): "
                f'<button onclick="{js_copy}" style="cursor:pointer;font-size:10px;margin-left:5px;padding:2px 6px;border:1px solid #ccc;border-radius:3px;background:#fff" title="Copy to Clipboard">📋</button><br>'
                f'<textarea id="{vec_id}" onclick="this.select();" readonly style="width: 96%; height: 80px; font-family: monospace; font-size: 10px; white-space: pre-wrap; margin-top: 4px;">'
                f"[{full_vec_str}]"
                f"</textarea>"
            )

            # Sanitize for JS string (escape single quotes and newlines)
            details_js_safe = details_html.replace("'", "\\'").replace("\n", " ")

            # Construct Javascript call
            js_click = f"document.getElementById('details-{block_id}').innerHTML = '{details_js_safe}'; document.getElementById('details-{block_id}').style.display = 'block';"

            # Sanitize for HTML attribute (escape double quotes)
            js_click_attr = js_click.replace('"', "&quot;")

            tooltip = f"Unit #{u.index} (Click for details)"
            svg_parts.append(
                f'<rect x="{x}" y="{y}" width="{unit_w}" height="{unit_h}" fill="transparent" stroke="none" onclick="{js_click_attr}"><title>{tooltip}</title></rect>'
            )
            # --- END MODIFICATION ---

        svg_parts.append("</svg>")
        html_parts.append("".join(svg_parts))

        # Details Box
        html_parts.append(
            f'<div id="details-{block_id}" style="margin-top: 8px; padding: 10px; '
            f"background-color: #f0f2f6; border-radius: 4px; border: 1px solid #e5e7eb; "
            f'font-family: monospace; display: none;">'
            f"</div>"
        )
        html_parts.append("</div>")

        return "".join(html_parts)
