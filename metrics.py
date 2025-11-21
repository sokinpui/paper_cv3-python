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
    def compute(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Computes structural dissimilarity (1 - SSIM) between all pairs.
        Simplified SSIM for patch-wise comparison.
        """
        # Flatten spatial dims: (N, C, H*W)
        N, C, H, W = patches.shape
        x = patches.view(N, C, -1)

        # Mean: (N, C, 1)
        mu = x.mean(dim=2, keepdim=True)
        # Centered: (N, C, H*W)
        x_centered = x - mu
        # Variance: (N, C, 1)
        sigma2 = (x_centered ** 2).mean(dim=2, keepdim=True)
        sigma = torch.sqrt(sigma2)

        # Prepare for broadcasting
        # A: (N, 1, C, 1), B: (1, N, C, 1)
        mu_A = mu.unsqueeze(1)
        mu_B = mu.unsqueeze(0)
        sigma2_A = sigma2.unsqueeze(1)
        sigma2_B = sigma2.unsqueeze(0)
        sigma_A = sigma.unsqueeze(1)
        sigma_B = sigma.unsqueeze(0)

        # Covariance calculation via matrix multiplication
        # (N, C, L) @ (N, C, L).T is too big if we do naive.
        # Instead, we compute covariance pairwise.
        # Cov(A, B) = Mean((A-muA)(B-muB))

        # Optimization: Use simplified SSIM constants
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        # Luminance term
        l_num = (2 * mu_A * mu_B + C1)
        l_den = (mu_A ** 2 + mu_B ** 2 + C1)

        # Contrast term (partially combined with structure in standard formula)
        # We need covariance for the full formula.
        # For efficiency in N*N, we approximate or compute dot product.

        # Reshape for dot product: (N, C*H*W)
        flat = x_centered.view(N, -1)
        # Covariance matrix (N, N) scaled by 1/L
        cov_matrix = (flat @ flat.T) / (H * W)
        # Reshape to (N, N, 1) to match C dimension logic if needed,
        # but SSIM is usually per channel then averaged.

        # Let's stick to the standard structure term using the precomputed variances
        # This implementation assumes we average over channels at the end

        # Re-calculating Covariance properly for broadcasting
        # This is memory intensive for huge N.
        # We use the property: Cov(X,Y) = E[XY] - E[X]E[Y]

        # E[XY]
        flat_raw = x.view(N, -1)
        exy = (flat_raw @ flat_raw.T) / (H * W) # (N, N)

        # E[X]E[Y]
        # mu is (N, C, 1). Average over C for global patch mean or keep C?
        # Standard SSIM is per channel.

        # To keep it fast and robust on GPU for N^2:
        # We will treat the patch as a vector for structural comparison.

        # Vectorized SSIM (approximate for speed on N^2)
        # SSIM(x, y) = (2*mu_x*mu_y + C1)(2*sig_xy + C2) / ((mu_x^2+mu_y^2+C1)(sig_x^2+sig_y^2+C2))

        # We need to average over channels (C)
        mu_flat = mu.mean(dim=1).squeeze() # (N)
        sigma2_flat = sigma2.mean(dim=1).squeeze() # (N)

        mu_x = mu_flat.unsqueeze(1) # (N, 1)
        mu_y = mu_flat.unsqueeze(0) # (1, N)
        sig2_x = sigma2_flat.unsqueeze(1)
        sig2_y = sigma2_flat.unsqueeze(0)

        # Covariance (averaged over channels)
        # (N, C*H*W)
        flat_centered = x_centered.view(N, -1)
        sig_xy = (flat_centered @ flat_centered.T) / (H * W * C)

        luminance = (2 * mu_x * mu_y + C1) / (mu_x**2 + mu_y**2 + C1)
        contrast_structure = (2 * sig_xy + C2) / (sig2_x + sig2_y + C2)

        ssim_map = luminance * contrast_structure
        return 1.0 - ssim_map

class CIELabMetric(MetricStrategy):
    def compute(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Computes Delta E (Euclidean distance in Lab space).
        Input assumed to be normalized RGB [0, 1].
        """
        lab = self._rgb_to_lab(patches)

        # Discard the 'L' (Lightness) channel. 
        # lab is (N, 3, H, W). Channel 0 is L, 1 is a, 2 is b.
        # We only keep channels 1 and 2 (a and b).
        lab = lab[:, 1:, :, :]


        # Flatten to (N, Features)
        # We want the average color difference per pixel or total difference?
        # Usually mean delta E per pixel.
        N, C, H, W = lab.shape
        flat = lab.view(N, C, -1).permute(0, 2, 1).reshape(N * H * W, C)

        # Calculating pairwise distance for (N, C, H, W) is heavy if we do pixel-to-pixel exact match.
        # Assumption: We compare Unit X to Unit Y.
        # Distance = Mean Euclidean distance between corresponding pixels.

        # Reshape: (N, D) where D = C*H*W
        flat_vec = lab.view(N, -1)

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
