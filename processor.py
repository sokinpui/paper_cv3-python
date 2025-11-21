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

    def extract_patches(
        self, image: torch.Tensor, unit_h: int, unit_w: int
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Divides image into non-overlapping patches.
        Returns:
            patches: (N, C, unit_h, unit_w)
            grid_shape: (rows, cols)
        """
        B, C, H, W = image.shape

        # Pad if dimensions are not divisible by unit size
        pad_h = (unit_h - H % unit_h) % unit_h
        pad_w = (unit_w - W % unit_w) % unit_w

        if pad_h > 0 or pad_w > 0:
            image = torch.nn.functional.pad(
                image, (0, pad_w, 0, pad_h), mode="replicate"  # extend from edge color
            )

        H_pad, W_pad = image.shape[2], image.shape[3]

        # Unfold to extract patches
        # kernel_size=(unit_h, unit_w), stride=(unit_h, unit_w)
        patches = torch.nn.functional.unfold(
            image, kernel_size=(unit_h, unit_w), stride=(unit_h, unit_w)
        )

        # patches shape: (B, C*unit_h*unit_w, N_patches)
        # Reshape to (N_patches, C, unit_h, unit_w)
        patches = patches.transpose(1, 2).squeeze(0)
        N_patches = patches.shape[0]
        patches = patches.view(N_patches, C, unit_h, unit_w)

        rows = H_pad // unit_h
        cols = W_pad // unit_w

        return patches, (rows, cols)

    def _draw_annotations(
        self,
        img: np.ndarray,
        units: list,
        unit_h: int,
        unit_w: int,
        is_bgr: bool = False,
    ):
        """
        Helper to draw individual rectangles and labels for units.
        """
        box_color = (0, 255, 0)  # Green
        text_color = (0, 0, 255) if is_bgr else (255, 0, 0)  # Red

        for i, unit in enumerate(units):
            r, c = unit.row, unit.col
            y, x = r * unit_h, c * unit_w

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

        self._draw_annotations(img_np, units, unit_h, unit_w, is_bgr=True)

        cv2.imwrite(output_path, img_np)
        print(f"Annotated image saved to {output_path}")

    def get_annotated_rgb(
        self, image: torch.Tensor, units: list, unit_h: int, unit_w: int
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

        self._draw_annotations(img_out, units, unit_h, unit_w, is_bgr=False)

        return img_out

    def create_heatmap(
        self,
        image: torch.Tensor,
        units: list,
        grid_shape: Tuple[int, int],
        unit_h: int,
        unit_w: int,
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

        overlay = img_np.copy()
        rows, cols = grid_shape

        # Extract scores for normalization
        # We assume units contains all units for the grid
        scores = [u.mean for u in units]
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
                norm = (u.mean - min_s) / rng

                # Color mapping: Blue (Low) -> Green -> Red (High)
                if norm < 0.5:
                    # 0.0 (Blue) -> 0.5 (Green)
                    n = norm * 2
                    r_val, g_val, b_val = 0, int(255 * n), int(255 * (1 - n))
                else:
                    # 0.5 (Green) -> 1.0 (Red)
                    n = (norm - 0.5) * 2
                    r_val, g_val, b_val = int(255 * n), int(255 * (1 - n)), 0

                y, x = r * unit_h, c * unit_w
                # Draw filled rectangle
                cv2.rectangle(
                    overlay, (x, y), (x + unit_w, y + unit_h), (r_val, g_val, b_val), -1
                )

        # Alpha blend
        alpha = 0.6
        cv2.addWeighted(overlay, alpha, img_np, 1 - alpha, 0, img_np)
        return img_np
