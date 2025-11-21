from dataclasses import dataclass
from typing import Dict, List

import torch


@dataclass
class UnitStats:
    index: int
    row: int
    col: int
    mean: float
    median: float
    std_dev: float
    min_score: float
    max_score: float

    def to_dict(self):
        return self.__dict__


class PatchAnalyzer:
    def __init__(self, metric_strategy):
        self.metric = metric_strategy

    def analyze(
        self,
        patches: torch.Tensor,
        grid_shape: tuple,
        top_n: int,
        sort_by: str = "mean",
        ascending: bool = True,
    ) -> List[UnitStats]:
        """
        patches: (N, C, H, W)
        """
        N = patches.shape[0]
        if N < 2:
            raise ValueError("Need at least 2 units to compare.")

        # 1. Compute Similarity/Distance Matrix (N, N)
        # This is the heavy GPU operation
        matrix = self.metric.compute(patches)

        # 2. Mask diagonal (self-comparison) to avoid skewing stats
        # We set diagonal to NaN so we can ignore it in stats
        mask = torch.eye(N, device=patches.device).bool()
        matrix.masked_fill_(mask, float("nan"))

        # 3. Calculate Statistics per Unit (Row-wise)
        # Clone to avoid modifying matrix for subsequent steps if needed
        data = matrix.clone()

        # Handle NaNs for stats
        # Note: nanmean, nanmedian are available in newer pytorch versions.
        # If not, we mask. Assuming modern pytorch here.

        # Count valid comparisons per unit (row)
        # When using local radius, this varies (corners < center).
        valid_counts = (~torch.isnan(data)).sum(dim=1)

        # 1. Mean
        means = torch.nanmean(data, dim=1)

        # 2. Median
        # Sort puts NaNs at the end (ascending).
        sorted_vals, _ = torch.sort(data, dim=1)
        # Index of the middle valid element
        mid_indices = ((valid_counts - 1) // 2).clamp(min=0)
        medians = torch.gather(sorted_vals, 1, mid_indices.unsqueeze(1)).squeeze(1)

        # 3. Std Dev (Sample)
        # var = sum((x - mean)^2) / (n - 1)
        centered = data - means.unsqueeze(1)
        sum_sq_diff = torch.nansum(centered**2, dim=1)
        # Avoid division by zero if count <= 1
        divisor = (valid_counts - 1).clamp(min=1)
        stds = torch.sqrt(sum_sq_diff / divisor)

        # 4. Min / Max
        # Fill NaNs with inf/-inf to ignore them in min/max reduction
        mins = torch.nan_to_num(data, nan=float("inf")).min(dim=1).values
        maxs = torch.nan_to_num(data, nan=float("-inf")).max(dim=1).values

        # 4. Aggregate results
        results = []
        rows, cols = grid_shape

        for i in range(N):
            r, c = divmod(i, cols)
            stats = UnitStats(
                index=i,
                row=r,
                col=c,
                mean=means[i].item(),
                median=medians[i].item(),
                std_dev=stds[i].item(),
                min_score=mins[i].item(),
                max_score=maxs[i].item(),
            )
            results.append(stats)

        # 5. Rank
        results.sort(key=lambda x: getattr(x, sort_by), reverse=not ascending)

        return results[:top_n]

    def cluster(
        self, patches: torch.Tensor, n_clusters: int, method: str = "hierarchical"
    ) -> List[int]:
        """
        Groups patches into n_clusters based on the metric distance.
        Returns a list of cluster labels corresponding to patches.
        method: 'hierarchical' (uses metric matrix) or 'kmeans' (uses metric features)
        """
        try:
            from sklearn.cluster import AgglomerativeClustering, KMeans
        except ImportError:
            raise ImportError("scikit-learn is required for clustering.")

        if method == "kmeans":
            # K-Means uses features (N, Features)
            # Note: For SSIM, this falls back to flattened raw pixels which ignores structure.
            features = self.metric.get_features(patches)
            X = features.detach().cpu().numpy()

            model = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
            labels = model.fit_predict(X)
        else:
            # Hierarchical uses the Pairwise Matrix (N, N)
            matrix = self.metric.compute(patches)
            matrix.fill_diagonal_(0)
            dist_matrix = matrix.detach().cpu().numpy()

            try:
                model = AgglomerativeClustering(
                    n_clusters=n_clusters, metric="precomputed", linkage="average"
                )
            except TypeError:
                model = AgglomerativeClustering(
                    n_clusters=n_clusters, affinity="precomputed", linkage="average"
                )

            labels = model.fit_predict(dist_matrix)

        return labels.tolist()
