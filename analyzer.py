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
    cluster_id: int = -1

    def to_dict(self):
        return self.__dict__


class PatchAnalyzer:
    def __init__(self, metric_strategy):
        self.metric = metric_strategy

    def _hierarchical(self, matrix: torch.Tensor, k: int) -> torch.Tensor:
        """
        Agglomerative Hierarchical Clustering (Ward linkage).
        matrix: (N, N) distance matrix.
        """
        try:
            import numpy as np
            from scipy.cluster.hierarchy import fcluster, linkage
            from scipy.spatial.distance import squareform
        except ImportError:
            print("Error: Scipy is required for hierarchical clustering.")
            return torch.zeros(matrix.shape[0], dtype=torch.long, device=matrix.device)

        dist_matrix = matrix.detach().cpu().numpy()
        # Ensure symmetry and zero diagonal
        dist_matrix = (dist_matrix + dist_matrix.T) / 2.0
        np.fill_diagonal(dist_matrix, 0)

        condensed = squareform(dist_matrix, checks=False)
        Z = linkage(condensed, method="ward")

        # Automatic k determination if k < 2 (Auto mode)
        if k < 2:
            distances = Z[:, 2]
            num_merges = len(distances)
            if num_merges > 1:
                # Heuristic: Largest jump in the last 15 merges (distances)
                window = min(15, num_merges)
                last_dists = distances[-window:]
                acceleration = np.diff(last_dists)
                # k = window - index_of_max_jump
                k = window - np.argmax(acceleration)
            else:
                k = 2

        labels = fcluster(Z, t=k, criterion="maxclust")

        return torch.tensor(labels - 1, dtype=torch.long, device=matrix.device)

    def _kmeans(self, data: torch.Tensor, k: int, max_iter: int = 20) -> torch.Tensor:
        """
        Generic K-Means implementation.
        data: (N, D)
        Returns: labels (N,)
        """
        N = data.shape[0]
        if N < k:
            return torch.zeros(N, dtype=torch.long, device=data.device)

        # 1. Initialize centroids deterministically (Maximin / Farthest Point Sampling)
        # This ensures stability and covers extremes (like min/max for 1D).
        centroids = []

        # 1.1 Start with the point having the minimum value (1D) or norm (ND)
        if data.shape[1] == 1:
            first_idx = torch.argmin(data).item()
        else:
            first_idx = torch.argmin(torch.sum(data**2, dim=1)).item()
        centroids.append(data[first_idx])

        # 1.2 Select subsequent centroids based on maximum distance
        dist_sq = torch.sum((data - centroids[0]) ** 2, dim=1)
        for _ in range(1, k):
            next_idx = torch.argmax(dist_sq).item()
            centroids.append(data[next_idx])
            new_dist_sq = torch.sum((data - centroids[-1]) ** 2, dim=1)
            dist_sq = torch.min(dist_sq, new_dist_sq)

        centroids = torch.stack(centroids)

        labels = torch.zeros(N, dtype=torch.long, device=data.device)

        for _ in range(max_iter):
            # 2. Assign labels: |x - c|
            # data (N, D), centroids (K, D) -> dists (N, K)
            dists = torch.cdist(data, centroids)
            new_labels = torch.argmin(dists, dim=1)

            if torch.equal(labels, new_labels):
                break
            labels = new_labels

            # 3. Update centroids
            for i in range(k):
                mask = labels == i
                if mask.any():
                    centroids[i] = data[mask].mean(dim=0)
                # else keep old

        return labels

    def analyze(
        self,
        patches: torch.Tensor,
        grid_shape: tuple,
        top_n: int,
        sort_by: str = "mean",
        ascending: bool = True,
        cluster_on_matrix: bool = False,
        k: int = 2,
        clustering_algorithm: str = "kmeans",
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

        # Optional: Cluster on the distance matrix (rows as features)
        matrix_labels = None
        # Allow k < 2 for hierarchical (auto mode), but require k > 1 for kmeans
        if cluster_on_matrix and (k > 1 or clustering_algorithm == "hierarchical"):
            if clustering_algorithm == "hierarchical":
                matrix_labels = self._hierarchical(matrix, k)
            else:
                matrix_labels = self._kmeans(matrix, k)

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
                cluster_id=matrix_labels[i].item() if matrix_labels is not None else -1,
            )
            results.append(stats)

        # 5. Rank
        results.sort(key=lambda x: getattr(x, sort_by), reverse=not ascending)

        return results[:top_n]

    def cluster_stats(
        self,
        stats: List[UnitStats],
        k: int,
        metric: str = "mean",
        threshold_n: float = 1.0,
    ) -> List[UnitStats]:
        """
        Performs 1D K-Means clustering on the specified score of the units.
        Updates the cluster_id in the UnitStats objects.
        """
        if not stats or k < 2:
            return stats

        # Extract data (N, 1) based on metric
        values = []
        for s in stats:
            if metric == "std_dev":
                values.append(s.std_dev)
            elif metric == "threshold":
                values.append(s.mean + threshold_n * s.std_dev)
            else:
                values.append(s.mean)

        data = torch.tensor(values, dtype=torch.float32).view(-1, 1)
        labels = self._kmeans(data, k)

        # Assign back to stats
        for i, s in enumerate(stats):
            s.cluster_id = labels[i].item()

        return stats
