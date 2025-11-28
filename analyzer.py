from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
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

    def _hierarchical(
        self, matrix: torch.Tensor, k: int, method: str = "ward"
    ) -> torch.Tensor:
        """
        Agglomerative Hierarchical Clustering (Ward linkage).
        matrix: (N, N) distance matrix.
        """
        try:
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
        Z = linkage(condensed, method=method)

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

    def _spectral(self, matrix: torch.Tensor, k: int) -> torch.Tensor:
        """
        Spectral Clustering using Normalized Cuts (Ng, Jordan, Weiss).
        """
        N = matrix.shape[0]
        if N < k:
            return torch.zeros(N, dtype=torch.long, device=matrix.device)

        # 1. Affinity Matrix from Distance Matrix
        # Heuristic: sigma = median distance
        flat_dists = matrix.view(-1)
        sigma = torch.median(flat_dists)
        if sigma < 1e-6:
            sigma = 1.0

        # Gaussian Kernel
        affinity = torch.exp(-(matrix**2) / (2 * sigma**2))
        affinity.fill_diagonal_(0)

        # 2. Normalized Laplacian: D^(-1/2) * A * D^(-1/2)
        degrees = affinity.sum(dim=1)
        d_inv_sqrt = torch.pow(degrees + 1e-8, -0.5)

        # Symmetric normalized adjacency
        dad = d_inv_sqrt.unsqueeze(1) * affinity * d_inv_sqrt.unsqueeze(0)

        # 3. Eigen Decomposition
        # torch.linalg.eigh returns eigenvalues in ascending order
        eigenvalues, eigenvectors = torch.linalg.eigh(dad)

        # 4. Top k eigenvectors (corresponding to k largest eigenvalues)
        features = eigenvectors[:, -k:]

        # 5. Normalize rows to unit length
        norm = torch.norm(features, p=2, dim=1, keepdim=True)
        features = features / (norm + 1e-8)

        # 6. K-Means on embeddings
        return self._kmeans(features, k)

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

    def _dbscan(
        self, matrix: torch.Tensor, eps: float, min_samples: int
    ) -> torch.Tensor:
        """
        DBSCAN density-based clustering on distance matrix.
        """
        # Heuristic for eps if Auto (<= 0)
        if eps <= 0:
            # User suggestion: "stddev of vector length".
            # Approximated by std dev of the distance matrix values.
            eps = matrix.std().item()
            if eps < 1e-6:
                eps = 0.5

        device = matrix.device
        N = matrix.shape[0]

        # Convert to CPU Numpy for logic
        dist_mat = matrix.detach().cpu().numpy()

        # 1. Adjacency: dist <= eps
        adj = dist_mat <= eps

        # 2. Identify Core Points
        # count neighbors (includes self because dist(i,i)=0 <= eps)
        degrees = np.sum(adj, axis=1)
        core_mask = degrees >= min_samples

        labels = -1 * np.ones(N, dtype=np.int64)
        cluster_id = 0

        # 3. Cluster Expansion
        # Iterate over all points, but we only start expansion from unvisited core points
        for i in range(N):
            if labels[i] != -1 or not core_mask[i]:
                continue

            # Start new cluster
            labels[i] = cluster_id
            stack = [i]

            while stack:
                curr = stack.pop()
                neighbors = np.where(adj[curr])[0]

                for neighbor in neighbors:
                    if labels[neighbor] == -1:
                        # Was Noise or Unvisited -> assign to cluster
                        labels[neighbor] = cluster_id
                        if core_mask[neighbor]:
                            stack.append(neighbor)

            cluster_id += 1

        return torch.from_numpy(labels).to(device)

    def _merge_spatial_clusters(
        self, stats: List[UnitStats], grid_shape: Tuple[int, int]
    ) -> List[UnitStats]:
        if not stats:
            return stats

        rows, cols = grid_shape
        unit_map = {(s.row, s.col): s for s in stats}

        # 1. Find background cluster (most frequent, non-noise)
        cluster_counts = {}
        for s in stats:
            if s.cluster_id >= 0:
                cluster_counts[s.cluster_id] = cluster_counts.get(s.cluster_id, 0) + 1

        if not cluster_counts:
            return stats  # No non-noise clusters to merge

        background_cluster_id = max(cluster_counts, key=cluster_counts.get)
        anomalous_clusters = {
            cid for cid in cluster_counts if cid != background_cluster_id
        }

        if not anomalous_clusters:
            return stats  # Nothing to merge

        # 2. Build adjacency graph of anomalous clusters
        adj = {cid: set() for cid in anomalous_clusters}
        for r in range(rows):
            for c in range(cols):
                current_unit = unit_map.get((r, c))
                if (
                    not current_unit
                    or current_unit.cluster_id not in anomalous_clusters
                ):
                    continue

                current_cid = current_unit.cluster_id

                # Check neighbors (4-connectivity)
                for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nr, nc = r + dr, c + dc
                    neighbor_unit = unit_map.get((nr, nc))

                    if (
                        not neighbor_unit
                        or neighbor_unit.cluster_id not in anomalous_clusters
                    ):
                        continue

                    neighbor_cid = neighbor_unit.cluster_id
                    if current_cid != neighbor_cid:
                        adj[current_cid].add(neighbor_cid)
                        adj[neighbor_cid].add(current_cid)

        # 3. Find connected components (groups of touching clusters)
        visited = set()
        components = []
        for cid in anomalous_clusters:
            if cid in visited:
                continue

            component = []
            q = [cid]
            visited.add(cid)
            while q:
                curr_cid = q.pop(0)
                component.append(curr_cid)
                for neighbor_cid in adj[curr_cid]:
                    if neighbor_cid not in visited:
                        visited.add(neighbor_cid)
                        q.append(neighbor_cid)
            components.append(component)

        # 4. Re-label clusters based on components
        max_id = max(s.cluster_id for s in stats) if stats else -1
        relabel_map = {}
        for i, component in enumerate(components):
            new_cid = max_id + 1 + i
            for old_cid in component:
                relabel_map[old_cid] = new_cid

        # 5. Apply new labels
        for s in stats:
            s.cluster_id = relabel_map.get(s.cluster_id, s.cluster_id)

        return stats

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
        hierarchical_method: str = "ward",
        eps: float = 0.0,
        min_samples: int = 1,
        power_transform_degree: float = 1.0,
    ) -> Tuple[List[UnitStats], torch.Tensor]:
        """
        patches: (N, C, H, W)
        """
        N = patches.shape[0]
        if N < 2:
            raise ValueError("Need at least 2 units to compare.")

        # 1. Compute Similarity/Distance Matrix (N, N)
        # This is the heavy GPU operation
        matrix = self.metric.compute(patches)

        # Optional: Apply Power Transformation to exaggerate/flatten distances
        if power_transform_degree != 1.0:
            matrix = torch.pow(matrix.clamp(min=0.0), power_transform_degree)

        # Optional: Cluster on the distance matrix (rows as features)
        matrix_labels = None
        # Allow k < 2 for hierarchical (auto mode), but require k > 1 for kmeans/spectral
        if cluster_on_matrix:
            if clustering_algorithm == "hierarchical":
                matrix_labels = self._hierarchical(
                    matrix, k, method=hierarchical_method
                )
            elif clustering_algorithm == "spectral":
                matrix_labels = self._spectral(matrix, k)
            elif clustering_algorithm in ["dbscan", "dbscan_spatial_merge"]:
                matrix_labels = self._dbscan(matrix, eps, min_samples)
            else:
                # Default to K-Means if k > 1
                if k > 1:
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

        # Post-process for DBSCAN2: merge spatially connected anomalous clusters
        if cluster_on_matrix and clustering_algorithm == "dbscan_spatial_merge":
            results = self._merge_spatial_clusters(results, grid_shape)

        # 5. Rank
        results.sort(key=lambda x: getattr(x, sort_by), reverse=not ascending)

        return results[:top_n], matrix

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
