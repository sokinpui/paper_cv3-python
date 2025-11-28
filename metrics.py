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


class HumanEyeColorMetric(MetricStrategy):
    def __init__(self, multiplier: float = 10.0, exponent: float = 2.5):
        self.multiplier = multiplier
        self.exponent = exponent

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

        H, W = patches.shape[2], patches.shape[3]
        return (self.multiplier * dists) ** self.exponent

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
