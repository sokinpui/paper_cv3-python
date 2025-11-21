from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn.functional as F


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


class LocalAnomalyAnalyzer:
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
        Performs Local Continuity Check (Bottom-Up).
        1. Compare each unit with its 8 neighbors.
        2. Create an Anomaly Map.
        3. Apply secondary aggregation kernel.
        """
        rows, cols = grid_shape
        N, C, H, W = patches.shape

        # 1. Reshape to Grid Indices
        grid_indices = torch.arange(N, device=patches.device).view(rows, cols)

        # Accumulators for the anomaly map
        score_map = torch.zeros((rows, cols), device=patches.device)
        count_map = torch.zeros((rows, cols), device=patches.device)

        # 8 Neighbors (dy, dx)
        neighbors = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ]

        for dy, dx in neighbors:
            # Define valid ranges for shifting
            r_start, r_end = max(0, -dy), min(rows, rows - dy)
            c_start, c_end = max(0, -dx), min(cols, cols - dx)
            
            nr_start, nr_end = max(0, dy), min(rows, rows + dy)
            nc_start, nc_end = max(0, dx), min(cols, cols + dx)

            # Extract valid indices
            src_idx = grid_indices[r_start:r_end, c_start:c_end].flatten()
            nbr_idx = grid_indices[nr_start:nr_end, nc_start:nc_end].flatten()

            if src_idx.numel() == 0:
                continue

            # Gather patches
            src_patches = patches[src_idx]
            nbr_patches = patches[nbr_idx]

            # Compute pairwise distance efficiently
            dists = self.metric.compute_pairs(src_patches, nbr_patches)

            # Accumulate scores back to the grid
            # index_add_ sums values into the specific indices
            score_map.view(-1).index_add_(0, src_idx, dists)
            count_map.view(-1).index_add_(0, src_idx, torch.ones_like(dists))

        # 2. Raw Anomaly Map (Mean distance to neighbors)
        # Units with no neighbors (single 1x1 grid) would be NaN, handle clamping
        raw_map = score_map / count_map.clamp(min=1)

        # 3. Secondary Pass (Kernel Sliding)
        # "Perform another kernel sliding on the units"
        # We use a 3x3 Average Pool to aggregate local error density.
        # Padding=1 maintains the grid size for mapping back to UnitStats.
        input_map = raw_map.unsqueeze(0).unsqueeze(0)  # (1, 1, R, C)
        smooth_map = F.avg_pool2d(input_map, kernel_size=3, stride=1, padding=1)
        
        final_scores = smooth_map.squeeze().view(-1)  # (N,)

        # 4. Generate Stats
        results = []
        for i in range(N):
            val = final_scores[i].item()
            r, c = divmod(i, cols)
            # We populate mean/max with the final aggregated score
            results.append(UnitStats(
                index=i,
                row=r,
                col=c,
                mean=val,
                median=val,
                std_dev=0.0,
                min_score=val,
                max_score=val
            ))

        results.sort(key=lambda x: getattr(x, sort_by), reverse=not ascending)
        return results[:top_n]
