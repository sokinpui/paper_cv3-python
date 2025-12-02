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
