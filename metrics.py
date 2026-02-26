from typing import Tuple

import torch
import torch.nn.functional as F

from utils import calculate_oklab_range


class MetricStrategy:
    def compute(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Input: patches (N, C, H, W)
        Output: Distance/Similarity Matrix (N, N)
        """
        raise NotImplementedError


class SSIMMetric(MetricStrategy):
    def __init__(
        self,
        k1: float = 0.01,
        k2: float = 0.03,
        alpha: float = 1.0,
        beta: float = 1.0,
    ):
        self.k1 = k1
        self.k2 = k2
        self.alpha = alpha
        self.beta = beta

    def compute(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Computes pairwise SSIM and converts to distance (1 - SSIM).
        High score = different.
        """
        N, C, H, W = patches.shape
        if C == 1:
            ssim_matrix = self._compute_ssim_matrix(patches)
        else:
            channel_ssims = []
            for i in range(C):
                channel_ssims.append(
                    self._compute_ssim_matrix(patches[:, i : i + 1, :, :])
                )
            ssim_matrix = torch.stack(channel_ssims).mean(dim=0)

        return 1.0 - ssim_matrix

    def _compute_ssim_matrix(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Computes SSIM matrix for a single channel.
        patches: (N, 1, H, W)
        """
        N, _, H, W = patches.shape

        L = 1.0
        C1 = (self.k1 * L) ** 2
        C2 = (self.k2 * L) ** 2

        mu = patches.mean(dim=(2, 3)).squeeze()
        sigma_sq = patches.var(dim=(2, 3), unbiased=False).squeeze()

        patches_centered = patches - mu.view(N, 1, 1, 1)
        patches_centered_flat = patches_centered.view(N, H * W)
        cov = (patches_centered_flat @ patches_centered_flat.T) / (H * W)

        mu_x = mu.unsqueeze(1)
        mu_y = mu.unsqueeze(0)
        sigma_x_sq = sigma_sq.unsqueeze(1)
        sigma_y_sq = sigma_sq.unsqueeze(0)

        # Luminance comparison (l)
        l_num = 2 * mu_x * mu_y + C1
        l_den = mu_x**2 + mu_y**2 + C1
        l_term = l_num / l_den

        # Contrast/Structure comparison (cs)
        cs_num = 2 * cov + C2
        cs_den = sigma_x_sq + sigma_y_sq + C2
        cs_term = cs_num / cs_den

        if self.alpha != 1.0:
            l_term = torch.pow(l_term.clamp(min=0.0), self.alpha)
        if self.beta != 1.0:
            cs_term = torch.pow(cs_term.clamp(min=0.0), self.beta)

        ssim_matrix = l_term * cs_term
        return ssim_matrix


class OklabMetric(MetricStrategy):
    def __init__(
        self,
        weights: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        threshold: float = float("inf"),
    ):
        self.weights = weights
        self.threshold = threshold

    def compute(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Uses Oklab color space (perceptually uniform) + Gaussian Blur.
        """
        # 1. Convert to Oklab
        oklab = self._rgb_to_oklab(patches)  # (N, 3, H, W)

        # 2. Apply Channel Weights (L, a, b)
        # Allows fine-tuning sensitivity to Lightness vs Color
        if self.weights != (1.0, 1.0, 1.0):
            # Shape (1, 3, 1, 1) to broadcast over N, H, W
            w_tensor = torch.tensor(
                self.weights, device=patches.device, dtype=patches.dtype
            ).view(1, 3, 1, 1)
            oklab = oklab * w_tensor

        oklab_blurred = oklab

        # 4. Flatten and Compute Euclidean Distance
        flat_vec = oklab_blurred.reshape(oklab_blurred.shape[0], -1)
        dists = torch.cdist(flat_vec, flat_vec, p=2.0)

        if 0 < self.threshold < float("inf"):
            _, _, H, W = patches.shape
            # 1^2 + 0.8^2 + 0.8^2 = 2.28
            # range of L = [0, 1]
            # range of a = [-0.4, 0.4]
            # range of b = [-0.4, 0.4]
            # maximum distance between two units are sqrt(2.28 x Height in pixel x Weight in pixel)
            # new_dist = dist x (maximum distance ^ 2)
            multipler = calculate_oklab_range(H, W)
            dists = torch.where(dists > self.threshold, dists * multipler, dists)

        return dists

    def _rgb_to_oklab(self, image: torch.Tensor) -> torch.Tensor:
        # Assumes image is (N, 3, H, W) in [0, 1] sRGB
        # 1. Inverse Gamma (sRGB to Linear RGB)
        mask = image > 0.04045
        linear_rgb = torch.zeros_like(image)
        linear_rgb[mask] = ((image[mask] + 0.055) / 1.055) ** 2.4
        linear_rgb[~mask] = image[~mask] / 12.92

        r, g, b = linear_rgb[:, 0], linear_rgb[:, 1], linear_rgb[:, 2]

        # 2. Linear RGB to LMS
        l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
        m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
        s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b

        # 3. Non-linearity (Cube root) + LMS to Oklab
        l_ = torch.pow(l.clamp(min=1e-12), 1 / 3)
        m_ = torch.pow(m.clamp(min=1e-12), 1 / 3)
        s_ = torch.pow(s.clamp(min=1e-12), 1 / 3)

        L = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_
        a = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_
        b = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_

        return torch.stack([L, a, b], dim=1)


class CosineMetric(MetricStrategy):
    def compute(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Computes Cosine Distance: 1 - Cosine Similarity.
        Range [0, 2]. 0 = Identical direction/pattern.
        Efficiently implemented via matrix multiplication.
        """
        # Flatten: (N, C, H, W) -> (N, D)
        flat = patches.reshape(patches.shape[0], -1)

        # Normalize rows (L2 norm) to create unit vectors
        norm = torch.norm(flat, p=2, dim=1, keepdim=True)
        flat_norm = flat / (norm + 1e-8)  # Avoid division by zero

        # Similarity = A . B^T (for unit vectors)
        similarity = torch.mm(flat_norm, flat_norm.t())

        # Distance = 1 - Similarity
        # Clamp ensures we don't get negative zeros or > 2 due to precision
        return 1.0 - similarity.clamp(-1.0, 1.0)
