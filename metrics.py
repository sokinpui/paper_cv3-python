import torch
import torch.nn.functional as F


class MetricStrategy:
    def compute(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Input: patches (N, C, H, W)
        Output: Distance/Similarity Matrix (N, N)
        """
        raise NotImplementedError

    def compute_pairs(self, patches_a: torch.Tensor, patches_b: torch.Tensor) -> torch.Tensor:
        """
        Computes distance between corresponding pairs in two batches.
        Input: patches_a (B, ...), patches_b (B, ...)
        Output: Distances (B,)
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
        Computes structural dissimilarity (1 - SSIM) between all pairs.
        Simplified SSIM for patch-wise comparison.
        """
        # Flatten spatial dims: (N, C, H*W)
        N, C, H, W = patches.shape
        x = patches.reshape(N, C, -1)

        # --- 1. Calculate Statistics ---
        # Mean per channel: (N, C)
        mu = x.mean(dim=2)

        # Centered data: (N, C, H*W)
        x_centered = x - mu.unsqueeze(2)

        # Flatten C into the vector for Structure correlation (N, C*H*W)
        # This ensures we check spatial structure across all channels
        flat_centered = x_centered.reshape(N, -1)

        # Covariance Matrix (N, N)
        # sig_xy = E[(x-mux)(y-muy)]
        sig_xy = (flat_centered @ flat_centered.T) / (H * W * C)

        # Variance (Diagonal of Covariance)
        sig2 = sig_xy.diag().unsqueeze(1)  # (N, 1)
        sig2_x = sig2
        sig2_y = sig2.T

        # --- 2. Structure & Contrast Term ---
        C2 = 0.03**2
        contrast_structure = (2 * sig_xy + C2) / (sig2_x + sig2_y + C2)

        # --- 3. Color Term (Cosine Similarity) ---
        # Compares the "Direction" of the Mean Vector (Color ratios).
        # Ignores "Magnitude" (Brightness/Shadows).
        # If RGB is used, this distinguishes R vs G vs B while treating Dark Red == Bright Red.

        # Normalize mean vectors (N, C)
        mu_norm = torch.linalg.norm(mu, dim=1, keepdim=True) + 1e-8
        mu_dir = mu / mu_norm

        # Cosine Similarity (N, N) -> Range [0, 1] for non-negative RGB
        color_sim = mu_dir @ mu_dir.T

        # --- 4. Combine ---
        # Combined Similarity = Structure * Color
        # Both terms are roughly [0, 1].
        # Distance = 1 - Similarity
        return 1.0 - (contrast_structure * color_sim)

    def compute_pairs(self, patches_a: torch.Tensor, patches_b: torch.Tensor) -> torch.Tensor:
        """
        Pairwise SSIM Distance (1 - SSIM) for corresponding elements.
        """
        N, C, H, W = patches_a.shape

        # Flatten spatial dims: (N, C, H*W)
        x = patches_a.reshape(N, C, -1)
        y = patches_b.reshape(N, C, -1)

        # Means
        mu_x = x.mean(dim=2)  # (N, C)
        mu_y = y.mean(dim=2)

        # Centered
        x_centered = x - mu_x.unsqueeze(2)
        y_centered = y - mu_y.unsqueeze(2)

        # Variances / Covariance
        # Flatten C: (N, C*H*W) to treat structure globally across channels
        flat_x = x_centered.reshape(N, -1)
        flat_y = y_centered.reshape(N, -1)

        # Element-wise dot product for covariance terms
        D = H * W * C
        sig_xy = (flat_x * flat_y).sum(dim=1) / D
        sig_xx = (flat_x * flat_x).sum(dim=1) / D
        sig_yy = (flat_y * flat_y).sum(dim=1) / D

        # Structure term
        C2 = 0.03**2
        structure = (2 * sig_xy + C2) / (sig_xx + sig_yy + C2)

        # Color term (Cosine Similarity of means)
        mu_x_norm = torch.linalg.norm(mu_x, dim=1) + 1e-8
        mu_y_norm = torch.linalg.norm(mu_y, dim=1) + 1e-8
        color_sim = (mu_x * mu_y).sum(dim=1) / (mu_x_norm * mu_y_norm)

        return 1.0 - (structure * color_sim)


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

    def compute_pairs(self, patches_a: torch.Tensor, patches_b: torch.Tensor) -> torch.Tensor:
        # Feature extraction
        fa = self.get_features(patches_a)  # (B, D)
        fb = self.get_features(patches_b)  # (B, D)
        H, W = patches_a.shape[2], patches_a.shape[3]
        # Euclidean distance row-wise + Normalize
        return torch.linalg.norm(fa - fb, dim=1) / (H * W) ** 0.5

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

    def compute_pairs(self, patches_a: torch.Tensor, patches_b: torch.Tensor) -> torch.Tensor:
        fa = self.get_features(patches_a)
        fb = self.get_features(patches_b)
        return torch.linalg.norm(fa - fb, dim=1)


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

    def compute_pairs(self, patches_a: torch.Tensor, patches_b: torch.Tensor) -> torch.Tensor:
        # Compute features on combined batch to ensure consistent statistics if Z-score used
        N = patches_a.shape[0]
        combined = torch.cat([patches_a, patches_b], dim=0)
        feats = self.get_features(combined)
        # Split back
        return torch.linalg.norm(feats[:N] - feats[N:], dim=1)


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

    def compute_pairs(self, patches_a: torch.Tensor, patches_b: torch.Tensor) -> torch.Tensor:
        N = patches_a.shape[0]
        combined = torch.cat([patches_a, patches_b], dim=0)
        feats = self.get_features(combined)
        return torch.linalg.norm(feats[:N] - feats[N:], dim=1)


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

    def compute_pairs(self, patches_a: torch.Tensor, patches_b: torch.Tensor) -> torch.Tensor:
        fa = self.get_features(patches_a)
        fb = self.get_features(patches_b)
        return torch.linalg.norm(fa - fb, dim=1)
