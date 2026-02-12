from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch

from clustering import dbscan, dbscan2, find_dbscan_eps


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
    cluster_id: int = -1
    l2_norm: float = 0.0
    neighbor_dist: float = 0.0
    nn_dist: float = 0.0

    def to_dict(self):
        return self.__dict__


class PatchAnalyzer:
    def __init__(self, metric_strategy):
        self.metric = metric_strategy

    def compute_distance_matrix(
        self, patches: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the distance matrix.
        """
        return self.metric.compute(patches)

    def analyze(
        self,
        patches: torch.Tensor,
        grid_shape: tuple,
        top_n: int,
        sort_by: str = "mean",
        ascending: bool = True,
        cluster_on_matrix: bool = False,
        clustering_algorithm: str = "kmeans",
        eps: float = 0.0,
        min_samples: int = 2,
    ) -> Tuple[List[UnitStats], torch.Tensor, float]:
        """
        patches: (N, C, H, W)
        """
        N = patches.shape[0]
        if N < 2:
            raise ValueError("Need at least 2 units to compare.")
        calculated_eps = eps

        # 1. Compute Similarity/Distance Matrix (N, N)
        matrix = self.compute_distance_matrix(patches)

        # Calculate k-Distance for debugging DBSCAN (Distance to the (min_samples-1)-th neighbor)
        # Index 0 is self, so index (min_samples-1) corresponds to the k-th neighbor count.
        k_idx = max(1, min_samples - 1)
        sorted_dists, _ = torch.sort(matrix, dim=1)
        # k-distance (for global density/debug)
        k_distances = sorted_dists[:, min(k_idx, N - 1)]
        # 1-NN distance (distance to closest neighbor, for connectivity check)
        # Index 0 is self (0.0), Index 1 is the nearest neighbor
        nn_distances = sorted_dists[:, 1] if N > 1 else torch.zeros(N, device=matrix.device)

        # Optional: Cluster on the distance matrix (rows as features)
        matrix_labels = None
        # Allow k < 2 for hierarchical (auto mode), but require k > 1 for kmeans/spectral

        # Auto-determine eps for DBSCAN/DBSCAN2 if selected and eps <= 0.0
        if (
            cluster_on_matrix
            and clustering_algorithm.startswith("dbscan")
            and eps <= 0.0
        ):
            calculated_eps = find_dbscan_eps(matrix, min_samples)

        if cluster_on_matrix:
            if clustering_algorithm == "dbscan":
                if calculated_eps <= 0.0:
                    calculated_eps = 1e-6
                matrix_labels = dbscan(matrix, calculated_eps, min_samples)
            elif clustering_algorithm == "dbscan2":
                if calculated_eps <= 0.0:
                    calculated_eps = 1e-6
                matrix_labels = dbscan2(matrix, calculated_eps, min_samples)

        # 2. Mask diagonal (self-comparison) to avoid skewing stats
        # We set diagonal to NaN so we can ignore it in stats
        mask = torch.eye(N, device=matrix.device).bool()
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

        l2_norms = torch.sqrt(torch.nansum(matrix**2, dim=1))

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
                cluster_id=matrix_labels[i].item() if matrix_labels is not None else -1,
                l2_norm=l2_norms[i].item(),
                neighbor_dist=k_distances[i].item(),
                nn_dist=nn_distances[i].item(),
            )
            results.append(stats)

        # 5. Rank
        results.sort(key=lambda x: getattr(x, sort_by), reverse=not ascending)

        return results[:top_n], matrix, calculated_eps

