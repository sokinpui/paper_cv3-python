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
        return patches.view(patches.shape[0], -1)

class SSIMMetric(MetricStrategy):
    def compute(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Computes structural dissimilarity (1 - SSIM) between all pairs.
        Simplified SSIM for patch-wise comparison.
        """
        # Flatten spatial dims: (N, C, H*W)
        N, C, H, W = patches.shape
        x = patches.view(N, C, -1)
        
        # --- 1. Calculate Statistics ---
        # Mean per channel: (N, C)
        mu = x.mean(dim=2)
        
        # Centered data: (N, C, H*W)
        x_centered = x - mu.unsqueeze(2)
        
        # Flatten C into the vector for Structure correlation (N, C*H*W)
        # This ensures we check spatial structure across all channels
        flat_centered = x_centered.view(N, -1)
        
        # Covariance Matrix (N, N)
        # sig_xy = E[(x-mux)(y-muy)]
        sig_xy = (flat_centered @ flat_centered.T) / (H * W * C)
        
        # Variance (Diagonal of Covariance)
        sig2 = sig_xy.diag().unsqueeze(1) # (N, 1)
        sig2_x = sig2
        sig2_y = sig2.T

        # --- 2. Structure & Contrast Term ---
        C2 = 0.03 ** 2
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
        flat_vec = lab.view(N, -1) # This is already flat if coming from get_features?

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

        return dists / (H * W)**0.5

    def get_features(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Returns flattened Lab image data (channels a and b only).
        """
        lab = self._rgb_to_lab(patches)
        # Keep only a, b (indices 1, 2) as per original logic
        lab = lab[:, 1:, :, :]
        # Return flattened (N, -1)
        return lab.view(patches.shape[0], -1)

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
            v[mask] = torch.pow(v[mask], 1/3)
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
        l_chan = lab[:, 0:1, :, :] # (N, 1, H, W)
        ab_chan = lab[:, 1:, :, :] # (N, 2, H, W)

        # 2. Compute Gradients on L channel (Sobel)
        kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=patches.device).view(1, 1, 3, 3).float()
        ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=patches.device).view(1, 1, 3, 3).float()

        # Padding 1 to keep size
        gx = F.conv2d(l_chan, kx, padding=1)
        gy = F.conv2d(l_chan, ky, padding=1)
        grad_mag = torch.sqrt(gx**2 + gy**2 + 1e-8) # (N, 1, H, W)

        # 3. Pool Features
        # Texture Energy (Mean Gradient) - Detects lines/scratches
        feat_grad_mean = grad_mag.mean(dim=(2, 3)) # (N, 1)
        # Texture Complexity (Std Gradient)
        feat_grad_std = grad_mag.std(dim=(2, 3))   # (N, 1)
        
        # Color (Mean a, Mean b) - Detects stains/discoloration
        # We ignore L mean to be robust to shadows/vignetting
        feat_color_mean = ab_chan.mean(dim=(2, 3)) # (N, 2)
        feat_color_std = ab_chan.std(dim=(2, 3))   # (N, 2)

        # Concatenate: (N, 6)
        features = torch.cat([feat_grad_mean, feat_grad_std, feat_color_mean, feat_color_std], dim=1)

        # 4. Z-Score Normalization
        # This ensures that "Edge Energy" and "Color Shift" are comparable, 
        # preventing one from dominating due to arbitrary scale.
        f_mean = features.mean(dim=0, keepdim=True)
        f_std = features.std(dim=0, keepdim=True) + 1e-8
        
        return (features - f_mean) / f_std
