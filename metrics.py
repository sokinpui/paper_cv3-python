import torch
import torch.nn.functional as F


class MetricStrategy:
    def compute(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Input: patches (N, C, H, W)
        Output: Distance/Similarity Matrix (N, N)
        """
        raise NotImplementedError

    def get_features(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Returns a feature vector representation for K-Means clustering.
        Default: Flattened raw patches (N, C*H*W).
        """
        return patches.reshape(patches.shape[0], -1)


class SSIMMetric(MetricStrategy):
    def compute(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Computes Mean Structural Similarity (MSSIM) between all pairs.
        Formula: SSIM(x,y) = l(x,y) * cs(x,y)
        Averaged over channels.
        """
        N, C, H, W = patches.shape
        P = H * W

        # Constants for SSIM (assuming range 0-1)
        L_dyn = 1.0
        K1 = 0.01
        K2 = 0.03
        C1 = (K1 * L_dyn) ** 2
        C2 = (K2 * L_dyn) ** 2

        # Reshape to (N, C, P)
        x = patches.reshape(N, C, -1)

        total_ssim = torch.zeros((N, N), device=patches.device)

        for c in range(C):
            # Extract channel: (N, P)
            xc = x[:, c, :]

            # 1. Means (N, 1)
            mu = xc.mean(dim=1, keepdim=True)
            mu_sq = mu**2

            # 2. Covariance (N, N)
            # Centered data
            xc_centered = xc - mu
            # E[(x-mux)(y-muy)]
            sigma_xy = (xc_centered @ xc_centered.T) / P

            # Variance (N, 1)
            sigma_sq = sigma_xy.diag().unsqueeze(1)

            # 3. Luminance Term (N, N)
            # (2 * mu_x * mu_y + C1) / (mu_x^2 + mu_y^2 + C1)
            mu_x_mu_y = mu @ mu.T
            mu_sq_sum = mu_sq + mu_sq.T
            l_term = (2 * mu_x_mu_y + C1) / (mu_sq_sum + C1)

            # 4. Contrast-Structure Term (N, N)
            # (2 * sigma_xy + C2) / (sigma_x^2 + sigma_y^2 + C2)
            sigma_sq_sum = sigma_sq + sigma_sq.T
            cs_term = (2 * sigma_xy + C2) / (sigma_sq_sum + C2)

            # Combine
            total_ssim += l_term * cs_term

        # Average over channels
        mean_ssim = total_ssim / C

        # Return Distance (1 - Similarity)
        return 1.0 - mean_ssim


class SSIMHalfMetric(MetricStrategy):
    def compute(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Computes Structural Similarity ignoring Luminance term (CS only).
        Formula: CS(x,y) = (2 * sigma_xy + C2) / (sigma_x^2 + sigma_y^2 + C2)
        Averaged over channels.
        """
        N, C, H, W = patches.shape
        P = H * W

        # Constants for SSIM (assuming range 0-1)
        L_dyn = 1.0
        K2 = 0.03
        C2 = (K2 * L_dyn) ** 2

        # Reshape to (N, C, P)
        x = patches.reshape(N, C, -1)

        total_cs = torch.zeros((N, N), device=patches.device)

        for c in range(C):
            # Extract channel: (N, P)
            xc = x[:, c, :]

            # 1. Means (N, 1) needed for covariance
            mu = xc.mean(dim=1, keepdim=True)

            # 2. Covariance (N, N)
            xc_centered = xc - mu
            sigma_xy = (xc_centered @ xc_centered.T) / P

            # Variance (N, 1)
            sigma_sq = sigma_xy.diag().unsqueeze(1)

            # 3. Contrast-Structure Term (N, N)
            sigma_sq_sum = sigma_sq + sigma_sq.T
            cs_term = (2 * sigma_xy + C2) / (sigma_sq_sum + C2)

            total_cs += cs_term

        # Average over channels
        mean_cs = total_cs / C

        # Return Distance (1 - Similarity)
        return 1.0 - mean_cs


class CIELabMetric(MetricStrategy):
    def compute(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Computes Delta E (Euclidean distance in Lab space).
        Input assumed to be normalized RGB [0, 1].
        """
        lab = self.get_features(patches)
        N = patches.shape[0]
        H, W = patches.shape[2], patches.shape[3]

        # Calculating pairwise distance for (N, C, H, W) is heavy if we do pixel-to-pixel exact match.
        # Assumption: We compare Unit X to Unit Y.
        # Distance = Mean Euclidean distance between corresponding pixels.

        # Reshape: (N, D) where D = C*H*W
        flat_vec = lab.reshape(
            N, -1
        )  # This is already flat if coming from get_features?

        # Euclidean Distance Matrix: ||A - B|| = sqrt(||A||^2 + ||B||^2 - 2<A,B>)
        # This computes distance between the flattened vectors.
        # To get Mean Delta E, we need to be careful.
        # Let's use the vector distance normalized by number of pixels.

        dists = torch.cdist(flat_vec, flat_vec, p=2)

        # Normalize by sqrt(pixels) because cdist sums squared differences
        # dist = sqrt(sum((a-b)^2))
        # mean_dist = dist / sqrt(H*W) is not quite right mathematically for Mean Delta E,
        # but it is a monotonic ranking equivalent.
        # For exact Mean Delta E, we would need element-wise averaging which is O(N^2 * H * W).
        # We will use Root Mean Square Error (RMSE) equivalent here.

        return dists / (H * W) ** 0.5

    def get_features(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Returns flattened Lab image data (channels a and b only).
        """
        lab = self._rgb_to_lab(patches)
        # Keep only a, b (indices 1, 2) as per original logic
        lab = lab[:, 1:, :, :]
        # Return flattened (N, -1)
        return lab.reshape(patches.shape[0], -1)

    def _rgb_to_lab(self, image: torch.Tensor) -> torch.Tensor:
        # RGB to XYZ
        # Assuming image is (N, 3, H, W) in [0, 1]
        r = image[:, 0, :, :]
        g = image[:, 1, :, :]
        b = image[:, 2, :, :]

        def _pivot_rgb(v):
            mask = v > 0.04045
            v[mask] = ((v[mask] + 0.055) / 1.055) ** 2.4
            v[~mask] = v[~mask] / 12.92
            return v * 100

        r = _pivot_rgb(r.clone())
        g = _pivot_rgb(g.clone())
        b = _pivot_rgb(b.clone())

        x = r * 0.4124 + g * 0.3576 + b * 0.1805
        y = r * 0.2126 + g * 0.7152 + b * 0.0722
        z = r * 0.0193 + g * 0.1192 + b * 0.9505

        # XYZ to Lab
        def _pivot_xyz(v):
            mask = v > 0.008856
            v[mask] = torch.pow(v[mask], 1 / 3)
            v[~mask] = (7.787 * v[~mask]) + (16 / 116)
            return v

        x = _pivot_xyz(x / 95.047)
        y = _pivot_xyz(y / 100.000)
        z = _pivot_xyz(z / 108.883)

        l_chan = (116 * y) - 16
        a_chan = 500 * (x - y)
        b_chan = 200 * (y - z)

        return torch.stack([l_chan, a_chan, b_chan], dim=1)


class MSEMetric(MetricStrategy):
    def compute(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Computes Mean Squared Error (MSE) between all pairs.
        Input: (N, C, H, W) normalized [0,1]
        Output: (N, N) matrix
        """
        N = patches.shape[0]
        # Flatten to (N, D)
        flat_vec = patches.reshape(N, -1)
        num_pixels = flat_vec.shape[1]

        # Pairwise Euclidean Distance (L2)
        dists = torch.cdist(flat_vec, flat_vec, p=2)

        # MSE = (L2^2) / D
        return (dists**2) / num_pixels


class PixelWiseColorMetric(CIELabMetric):
    def get_features(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Returns flattened Lab image data (L, a, b).
        """
        lab = self._rgb_to_lab(patches)
        return lab.reshape(patches.shape[0], -1)


class SSIMColorMixedMetric(MetricStrategy):
    def compute(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Computes a mixed metric: Structure (SSIM) * Color (Lab Similarity).
        Similarity = SSIM_Sim * (1 / (1 + Lab_Dist))
        Returns Distance = 1 - Similarity
        """
        ssim_sim = 1.0 - SSIMMetric().compute(patches)
        lab_dist = CIELabMetric().compute(patches)
        color_sim = 1.0 / (1.0 + lab_dist)
        return 1.0 - (ssim_sim * color_sim)


class LabMomentsMetric(CIELabMetric):
    def get_features(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Extracts (N, 6) feature vectors: [L_mu, a_mu, b_mu, L_std, a_std, b_std]
        Weighted for distance calculation.
        """
        # 1. Convert to Lab: (N, 3, H, W)
        lab = self._rgb_to_lab(patches)

        # 2. Compute Moments per channel -> (N, 3)
        # Mean: Average color
        means = lab.mean(dim=(2, 3))

        # Std: Color variation (Texture/Contrast)
        stds = lab.std(dim=(2, 3))

        # 3. Concatenate to form Feature Vector: (N, 6)
        # Vector: [L_mu, a_mu, b_mu, L_sigma, a_sigma, b_sigma]
        features = torch.cat([means, stds], dim=1)

        # 4. Weighting (Optional but recommended)
        # We want to penalize 'a' and 'b' (Color) differences more than 'L' (Lightness)
        # to ignore lighting gradients (shadows).
        # Indices: 0=L_mu, 1=a_mu, 2=b_mu, 3=L_std, 4=a_std, 5=b_std
        weights = torch.tensor([0.5, 2.0, 2.0, 0.5, 1.0, 1.0], device=patches.device)
        features = features * weights
        return features

    def compute(self, patches: torch.Tensor) -> torch.Tensor:
        features = self.get_features(patches)

        # 5. Compute Pairwise Euclidean Distance on the Feature Vectors
        # Input: (N, 6)
        # Output: (N, N)
        dists = torch.cdist(features, features, p=2)

        return dists


class TextureColorMetric(CIELabMetric):
    def get_features(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Extracts robust features for anomaly detection on uniform surfaces.
        1. Texture: Gradient Magnitude (Edges/Scratches) - Robust to smooth lighting.
        2. Color: 'a' and 'b' channels - Robust to shadows.
        3. Complexity: Std Dev of Gradient.
        """
        # 1. Convert to Lab (N, 3, H, W)
        lab = self._rgb_to_lab(patches)
        l_chan = lab[:, 0:1, :, :]  # (N, 1, H, W)
        ab_chan = lab[:, 1:, :, :]  # (N, 2, H, W)

        # 2. Compute Gradients on L channel (Sobel)
        kx = (
            torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=patches.device)
            .view(1, 1, 3, 3)
            .float()
        )
        ky = (
            torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=patches.device)
            .view(1, 1, 3, 3)
            .float()
        )

        # Padding 1 to keep size
        gx = F.conv2d(l_chan, kx, padding=1)
        gy = F.conv2d(l_chan, ky, padding=1)
        grad_mag = torch.sqrt(gx**2 + gy**2 + 1e-8)  # (N, 1, H, W)

        # 3. Pool Features
        # Texture Energy (Mean Gradient) - Detects lines/scratches
        feat_grad_mean = grad_mag.mean(dim=(2, 3))  # (N, 1)
        # Texture Complexity (Std Gradient)
        feat_grad_std = grad_mag.std(dim=(2, 3))  # (N, 1)

        # Color (Mean a, Mean b) - Detects stains/discoloration
        # We ignore L mean to be robust to shadows/vignetting
        feat_color_mean = ab_chan.mean(dim=(2, 3))  # (N, 2)
        feat_color_std = ab_chan.std(dim=(2, 3))  # (N, 2)

        # Concatenate: (N, 6)
        features = torch.cat(
            [feat_grad_mean, feat_grad_std, feat_color_mean, feat_color_std], dim=1
        )

        # 4. Z-Score Normalization
        # This ensures that "Edge Energy" and "Color Shift" are comparable,
        # preventing one from dominating due to arbitrary scale.
        f_mean = features.mean(dim=0, keepdim=True)
        f_std = features.std(dim=0, keepdim=True) + 1e-8

        return (features - f_mean) / f_std


class GradientColorMetric(CIELabMetric):
    def get_features(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Features:
        1. Texture Strength (Gradient Magnitude on L) - Captures lines/edges.
        2. Chrominance (Lab 'a' & 'b' Means) - Captures color shifts.
        3. Roughness (Luminance Std Dev) - Captures noise/texture variance.
        """
        # 1. Convert to Lab
        lab = self._rgb_to_lab(patches)
        l_chan = lab[:, 0:1, :, :]
        a_chan = lab[:, 1:2, :, :]
        b_chan = lab[:, 2:3, :, :]

        # 2. Texture Strength (Gradient Magnitude on L)
        kx = (
            torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=patches.device)
            .view(1, 1, 3, 3)
            .float()
        )
        ky = (
            torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=patches.device)
            .view(1, 1, 3, 3)
            .float()
        )

        gx = F.conv2d(l_chan, kx, padding=1)
        gy = F.conv2d(l_chan, ky, padding=1)
        grad_mag = torch.sqrt(gx**2 + gy**2 + 1e-8)

        feat_texture = grad_mag.mean(dim=(2, 3))  # (N, 1)

        # 3. Roughness (Luminance Std Dev)
        feat_roughness = l_chan.std(dim=(2, 3))  # (N, 1)

        # 4. Chrominance (Mean a, Mean b)
        feat_a = a_chan.mean(dim=(2, 3))  # (N, 1)
        feat_b = b_chan.mean(dim=(2, 3))  # (N, 1)

        # Concatenate: (N, 4)
        features = torch.cat([feat_texture, feat_roughness, feat_a, feat_b], dim=1)

        # 5. Z-Score Normalization
        f_mean = features.mean(dim=0, keepdim=True)
        f_std = features.std(dim=0, keepdim=True) + 1e-8

        return (features - f_mean) / f_std


class HistogramMetric(CIELabMetric):
    def get_features(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Computes marginal color histograms for L, a, b channels.
        Robust to rotation/translation of texture, focuses on color quantity.
        """
        # 1. Convert to Lab
        lab = self._rgb_to_lab(patches)  # (N, 3, H, W)

        N, C, H, W = lab.shape
        num_bins = 32

        # 2. Normalize channels to [0, 1] for binning
        # L: [0, 100] -> [0, 1]
        l = lab[:, 0].clamp(0, 100) / 100.0
        # a, b: [-128, 127] approx -> [0, 1]
        a = (lab[:, 1].clamp(-128, 127) + 128) / 255.0
        b = (lab[:, 2].clamp(-128, 127) + 128) / 255.0

        normalized = torch.stack([l, a, b], dim=1)  # (N, 3, H, W)

        # 3. Binning
        # Scale to integer indices [0, num_bins-1]
        indices = (normalized * num_bins).long().clamp(0, num_bins - 1)

        # Flatten spatial dimensions
        indices = indices.reshape(N, C, -1)  # (N, 3, H*W)

        # 4. Vectorized Batched Histogram (bincount)
        # Trick: Offset indices so every patch/channel has unique bins in a flat array
        # Global Index = patch_idx*(C*bins) + channel_idx*(bins) + bin_idx

        offset_patch = torch.arange(N, device=lab.device) * (C * num_bins)
        offset_channel = torch.arange(C, device=lab.device) * num_bins

        # Broadcasting to create offset map (N, C, 1)
        offsets = offset_patch.view(N, 1, 1) + offset_channel.view(1, C, 1)

        flat_indices = (indices + offsets).reshape(-1)  # Flatten everything

        total_bins = N * C * num_bins
        counts = torch.bincount(flat_indices, minlength=total_bins)

        # Reshape back to (N, Feature_Vector_Size)
        # Feature vector = C * num_bins = 3 * 32 = 96
        hist_features = counts.reshape(N, -1).float()

        # 5. Normalize (PDF)
        # Divide by number of pixels so sum is 1 per channel (roughly)
        hist_features = hist_features / (H * W)

        return hist_features


class HumanEyeColorMetric(MetricStrategy):
    def compute(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Uses Oklab color space (perceptually uniform) + Gaussian Blur.
        Oklab is currently the state-of-the-art for simple Euclidean perceptual color distance.
        The blur mimics the human eye's tendency to ignore high-frequency pixel noise
        and focus on regional color patches.
        """
        # 1. Convert to Oklab
        oklab = self._rgb_to_oklab(patches)  # (N, 3, H, W)

        # 2. Blur slightly (3x3 Gaussian) to simulate human visual area integration
        # This reduces false positives from single-pixel noise or slight texture shifts.
        kernel = (
            torch.tensor(
                [[1, 2, 1], [2, 4, 2], [1, 2, 1]], device=patches.device
            ).float()
            / 16.0
        )
        kernel = kernel.view(1, 1, 3, 3).repeat(3, 1, 1, 1)
        # groups=3 applies kernel to each channel independently
        oklab_blurred = F.conv2d(oklab, kernel, padding=1, groups=3)

        # 3. Flatten and Compute Euclidean Distance
        flat_vec = oklab_blurred.reshape(patches.shape[0], -1)
        dists = torch.cdist(flat_vec, flat_vec, p=2)

        # Normalize (approx RMSE)
        H, W = patches.shape[2], patches.shape[3]
        return dists / (H * W) ** 0.5

    def get_features(self, patches: torch.Tensor) -> torch.Tensor:
        oklab = self._rgb_to_oklab(patches)
        return oklab.reshape(patches.shape[0], -1)

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
        l_ = torch.pow(l.clamp(min=1e-8), 1 / 3)
        m_ = torch.pow(m.clamp(min=1e-8), 1 / 3)
        s_ = torch.pow(s.clamp(min=1e-8), 1 / 3)

        L = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_
        a = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_
        b = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_

        return torch.stack([L, a, b], dim=1)


class CIEDE2000Metric(MetricStrategy):
    def get_features(self, patches: torch.Tensor) -> torch.Tensor:
        # Convert RGB to Lab first
        # We reuse the conversion logic from CIELabMetric
        lab = CIELabMetric()._rgb_to_lab(patches)

        # Take the MEAN of the unit colors to represent the unit
        # (N, 3, H, W) -> (N, 3)
        return lab.mean(dim=(2, 3))

    def compute(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Computes the Pairwise CIEDE2000 Color Difference.
        Input: (N, C, H, W)
        Returns: (N, N) distance matrix
        """
        # (N, 3) features: L, a, b
        lab = self.get_features(patches)
        N = lab.shape[0]

        # We need pairwise comparisons.
        # Expand lab to (N, N, 3) for broadcasting
        # lab1: (N, 1, 3), lab2: (1, N, 3)
        lab1 = lab.unsqueeze(1)
        lab2 = lab.unsqueeze(0)

        L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
        L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]

        # --- CIEDE2000 Implementation ---
        # Constants
        kL = 1.0
        kC = 1.0
        kH = 1.0

        # 1. Calculate C' and h'
        C1 = torch.sqrt(a1**2 + b1**2)
        C2 = torch.sqrt(a2**2 + b2**2)
        C_bar = (C1 + C2) / 2.0

        G = 0.5 * (1 - torch.sqrt(C_bar**7 / (C_bar**7 + 25**7)))

        a1_prime = (1 + G) * a1
        a2_prime = (1 + G) * a2

        C1_prime = torch.sqrt(a1_prime**2 + b1**2)
        C2_prime = torch.sqrt(a2_prime**2 + b2**2)

        # Compute h_prime (Hue angle)
        # atan2 returns radians, we need degrees [0, 360]
        h1_prime = torch.rad2deg(torch.atan2(b1, a1_prime)) % 360
        h2_prime = torch.rad2deg(torch.atan2(b2, a2_prime)) % 360

        # 2. Calculate Delta L', Delta C', Delta H'
        delta_L_prime = L2 - L1
        delta_C_prime = C2_prime - C1_prime

        # Delta h_prime logic
        h_diff = h2_prime - h1_prime
        delta_h_prime = torch.zeros_like(h_diff)

        mask_abs_leq_180 = torch.abs(h_diff) <= 180
        mask_gt_180_pos = (torch.abs(h_diff) > 180) & (h2_prime <= h1_prime)
        mask_gt_180_neg = (torch.abs(h_diff) > 180) & (h2_prime > h1_prime)

        delta_h_prime[mask_abs_leq_180] = h_diff[mask_abs_leq_180]
        delta_h_prime[mask_gt_180_pos] = h_diff[mask_gt_180_pos] + 360
        delta_h_prime[mask_gt_180_neg] = h_diff[mask_gt_180_neg] - 360

        # delta H' (Big H)
        delta_H_prime = (
            2
            * torch.sqrt(C1_prime * C2_prime)
            * torch.sin(torch.deg2rad(delta_h_prime / 2.0))
        )

        # 3. Calculate CIEDE2000 Terms
        L_bar_prime = (L1 + L2) / 2.0
        C_bar_prime = (C1_prime + C2_prime) / 2.0

        # h_bar_prime logic
        h_sum = h1_prime + h2_prime
        h_bar_prime = torch.zeros_like(h_sum)

        mask_c_zero = (C1_prime * C2_prime) == 0
        mask_diff_leq_180 = (~mask_c_zero) & (torch.abs(h_diff) <= 180)
        mask_diff_gt_180_sum_lt_360 = (
            (~mask_c_zero) & (torch.abs(h_diff) > 180) & (h_sum < 360)
        )
        mask_diff_gt_180_sum_ge_360 = (
            (~mask_c_zero) & (torch.abs(h_diff) > 180) & (h_sum >= 360)
        )

        h_bar_prime[mask_c_zero] = h_sum[
            mask_c_zero
        ]  # Should technically be sum, logic handles it
        h_bar_prime[mask_diff_leq_180] = h_sum[mask_diff_leq_180] / 2.0
        h_bar_prime[mask_diff_gt_180_sum_lt_360] = (
            h_sum[mask_diff_gt_180_sum_lt_360] + 360
        ) / 2.0
        h_bar_prime[mask_diff_gt_180_sum_ge_360] = (
            h_sum[mask_diff_gt_180_sum_ge_360] - 360
        ) / 2.0

        T = (
            1
            - 0.17 * torch.cos(torch.deg2rad(h_bar_prime - 30))
            + 0.24 * torch.cos(torch.deg2rad(2 * h_bar_prime))
            + 0.32 * torch.cos(torch.deg2rad(3 * h_bar_prime + 6))
            - 0.20 * torch.cos(torch.deg2rad(4 * h_bar_prime - 63))
        )

        S_L = 1 + (0.015 * (L_bar_prime - 50) ** 2) / torch.sqrt(
            20 + (L_bar_prime - 50) ** 2
        )
        S_C = 1 + 0.045 * C_bar_prime
        S_H = 1 + 0.015 * C_bar_prime * T

        R_T = (
            -2
            * torch.sqrt(C_bar_prime**7 / (C_bar_prime**7 + 25**7))
            * torch.sin(
                torch.deg2rad(60 * torch.exp(-(((h_bar_prime - 275) / 25) ** 2)))
            )
        )

        # Final Calculation
        delta_E = torch.sqrt(
            (delta_L_prime / (kL * S_L)) ** 2
            + (delta_C_prime / (kC * S_C)) ** 2
            + (delta_H_prime / (kH * S_H)) ** 2
            + R_T * (delta_C_prime / (kC * S_C)) * (delta_H_prime / (kH * S_H))
        )

        return delta_E


class GradientStructureMetric(CIELabMetric):
    def get_features(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Features:
        1. Texture Strength (Gradient Magnitude on L) - Captures lines/edges.
        2. Roughness (Luminance Std Dev) - Captures noise/texture variance.

        Explicitly IGNORES mean color (a and b channels) so that
        different background colors (e.g. Green vs White) are treated as identical
        if they have the same surface smoothness.
        """
        # 1. Convert to Lab
        lab = self._rgb_to_lab(patches)
        l_chan = lab[:, 0:1, :, :]

        # 2. Texture Strength (Gradient Magnitude on L)
        kx = (
            torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=patches.device)
            .view(1, 1, 3, 3)
            .float()
        )
        ky = (
            torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=patches.device)
            .view(1, 1, 3, 3)
            .float()
        )

        gx = F.conv2d(l_chan, kx, padding=1)
        gy = F.conv2d(l_chan, ky, padding=1)
        grad_mag = torch.sqrt(gx**2 + gy**2 + 1e-8)

        feat_texture = grad_mag.mean(dim=(2, 3))  # (N, 1)

        # 3. Roughness (Luminance Std Dev)
        feat_roughness = l_chan.std(dim=(2, 3))  # (N, 1)

        # Concatenate: (N, 2) - NO Color Channels here
        features = torch.cat([feat_texture, feat_roughness], dim=1)

        # 4. Z-Score Normalization
        f_mean = features.mean(dim=0, keepdim=True)
        f_std = features.std(dim=0, keepdim=True) + 1e-8

        return (features - f_mean) / f_std
