from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F


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
        threshold: float = 0.0,
        multiplier: float = 1.0,
    ):
        self.k1 = k1
        self.k2 = k2
        self.alpha = alpha
        self.beta = beta
        self.threshold = threshold
        self.multiplier = multiplier

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

        dists = 1.0 - ssim_matrix
        if self.threshold > 0:
            # Range of 1-SSIM is [0, 2]
            actual_threshold = self.threshold * 2.0
            dists = torch.where(dists > actual_threshold, dists * self.multiplier, dists)

        return dists

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
        threshold: float = 0.0,
        multiplier: float = 1.0,
    ):
        self.weights = weights
        self.threshold = threshold
        self.multiplier = multiplier

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

        # 4. Flatten and Compute Euclidean Distance
        flat_vec = oklab.reshape(oklab.shape[0], -1)
        dists = torch.cdist(flat_vec, flat_vec, p=2.0)

        if self.threshold > 0:
            _, _, H, W = patches.shape
            # Max per-pixel Oklab L2 distance squared is approx 2.28
            max_dist = (2.28 * H * W) ** 0.5
            actual_threshold = self.threshold * max_dist
            dists = torch.where(dists > actual_threshold, dists * self.multiplier, dists)

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


class CIELABMetric(MetricStrategy):
    def __init__(
        self,
        kl: float = 1.0,
        kc: float = 1.0,
        kh: float = 1.0,
        threshold: float = 0.0,
        multiplier: float = 1.0,
    ):
        self.kl = kl
        self.kc = kc
        self.kh = kh
        self.threshold = threshold
        self.multiplier = multiplier

    def compute(self, patches: torch.Tensor) -> torch.Tensor:
        lab = self._rgb_to_lab(patches)
        # Compute pairwise distances where each cell (i, j) is the
        # mean of the Delta E 2000 map between patch i and patch j.
        dists = self._compute_pairwise_delta_e2000(lab)

        return self._apply_threshold(dists)

    def _apply_threshold(self, dists: torch.Tensor) -> torch.Tensor:
        if self.threshold <= 0:
            return dists
        # CIEDE2000 range is typically [0, 100]
        actual_threshold = self.threshold * 100.0
        return torch.where(dists > actual_threshold, dists * self.multiplier, dists)

    def _rgb_to_lab(self, image: torch.Tensor) -> torch.Tensor:
        mask = image > 0.04045
        linear_rgb = torch.zeros_like(image)
        linear_rgb[mask] = ((image[mask] + 0.055) / 1.055) ** 2.4
        linear_rgb[~mask] = image[~mask] / 12.92

        r, g, b = linear_rgb[:, 0], linear_rgb[:, 1], linear_rgb[:, 2]

        x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
        y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
        z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041

        # D65 White Point
        x /= 0.95047
        z /= 1.08883

        def f(t):
            m = t > 0.008856
            res = torch.zeros_like(t)
            res[m] = torch.pow(t[m], 1 / 3)
            res[~m] = (7.787 * t[~m]) + (16 / 116)
            return res

        fx, fy, fz = f(x), f(y), f(z)
        l = (116 * fy) - 16
        a = 500 * (fx - fy)
        b = 200 * (fy - fz)

        return torch.stack([l, a, b], dim=1)

    def _compute_pairwise_delta_e2000(self, lab: torch.Tensor) -> torch.Tensor:
        """
        Vectorized implementation of CIEDE2000.
        Input lab: (N, 3, H, W)
        Returns: (N, N) distance matrix where each entry is the mean pixel-wise Delta E.
        """
        N, _, H, W = lab.shape
        l, a, b = lab[:, 0], lab[:, 1], lab[:, 2]  # Each is (N, H, W)

        # Expand to (N, N, H, W) to compare all patches against each other
        l1, l2 = l.unsqueeze(1), l.unsqueeze(0)
        a1, a2 = a.unsqueeze(1), a.unsqueeze(0)
        b1, b2 = b.unsqueeze(1), b.unsqueeze(0)

        avg_lp = (l1 + l2) / 2.0
        c1 = torch.sqrt(a1**2 + b1**2)
        c2 = torch.sqrt(a2**2 + b2**2)
        avg_c1c2 = (c1 + c2) / 2.0

        g = 0.5 * (1 - torch.sqrt(avg_c1c2**7 / (avg_c1c2**7 + 25**7)))
        a1p, a2p = (1 + g) * a1, (1 + g) * a2
        c1p, c2p = torch.sqrt(a1p**2 + b1**2), torch.sqrt(a2p**2 + b2**2)
        avg_cp = (c1p + c2p) / 2.0

        h1p = torch.atan2(b1, a1p) * 180 / np.pi
        h1p = torch.where(h1p < 0, h1p + 360, h1p)
        h2p = torch.atan2(b2, a2p) * 180 / np.pi
        h2p = torch.where(h2p < 0, h2p + 360, h2p)

        h_diff = h2p - h1p
        h_diff = torch.where(
            torch.abs(h_diff) > 180, h_diff - 360 * torch.sign(h_diff), h_diff
        )
        delta_hp = 2 * torch.sqrt(c1p * c2p) * torch.sin(h_diff / 2.0 * np.pi / 180)

        avg_hp = torch.where(
            torch.abs(h1p - h2p) > 180, (h1p + h2p + 360) / 2.0, (h1p + h2p) / 2.0
        )

        t = (
            1
            - 0.17 * torch.cos((avg_hp - 30) * np.pi / 180)
            + 0.24 * torch.cos(2 * avg_hp * np.pi / 180)
            + 0.32 * torch.cos((3 * avg_hp + 6) * np.pi / 180)
            - 0.20 * torch.cos((4 * avg_hp - 63) * np.pi / 180)
        )

        delta_lp = l2 - l1
        delta_cp = c2p - c1p

        sl = 1 + (0.015 * (avg_lp - 50) ** 2) / torch.sqrt(20 + (avg_lp - 50) ** 2)
        sc = 1 + 0.045 * avg_cp
        sh = 1 + 0.015 * avg_cp * t

        delta_ro = 30 * torch.exp(-(((avg_hp - 275) / 25) ** 2))
        rc = 2 * torch.sqrt(avg_cp**7 / (avg_cp**7 + 25**7))
        rt = -torch.sin(2 * delta_ro * np.pi / 180) * rc

        # Calculate the Delta E 2000 map for every pair
        delta_e_map = torch.sqrt(
            (delta_lp / (self.kl * sl)) ** 2
            + (delta_cp / (self.kc * sc)) ** 2
            + (delta_hp / (self.kh * sh)) ** 2
            + rt * (delta_cp / (self.kc * sc)) * (delta_hp / (self.kh * sh))
        )

        # Return the mean difference per patch pair (N, N)
        return delta_e_map.mean(dim=(2, 3))
