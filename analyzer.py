import torch
from dataclasses import dataclass
from typing import List, Dict

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

    def analyze(self, patches: torch.Tensor, grid_shape: tuple, top_n: int, sort_by: str = 'mean', ascending: bool = True) -> List[UnitStats]:
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
        matrix.masked_fill_(mask, float('nan'))

        # 3. Calculate Statistics per Unit (Row-wise)
        # Clone to avoid modifying matrix for subsequent steps if needed
        data = matrix.clone()

        # Handle NaNs for stats
        # Note: nanmean, nanmedian are available in newer pytorch versions.
        # If not, we mask. Assuming modern pytorch here.

        means = torch.nanmean(data, dim=1)

        # Median: torch.nanmedian is strictly available in very recent versions.
        # Fallback: sort and pick middle ignoring nans
        sorted_vals, _ = torch.sort(data, dim=1)
        # The last column is NaN (since we pushed NaNs to end or beginning depending on sort).
        # Actually NaNs are usually at the end.
        # Valid elements are N-1. Median index is (N-1)//2
        medians = sorted_vals[:, (N-1)//2]

        # Std Dev
        # torch.std does not support nan ignore natively in older versions easily without a loop or masking
        # We use a mask approach
        not_nan_mask = ~torch.isnan(data)
        # We can't easily vectorize std with variable lengths if N is constant,
        # but here N is constant (N-1 valid items).
        # So we can just compute std on the N-1 items.
        # Let's gather valid items.
        valid_data = data[not_nan_mask].view(N, N-1)
        stds = torch.std(valid_data, dim=1)
        mins = torch.min(valid_data, dim=1).values
        maxs = torch.max(valid_data, dim=1).values

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
                max_score=maxs[i].item()
            )
            results.append(stats)

        # 5. Rank
        results.sort(key=lambda x: getattr(x, sort_by), reverse=not ascending)

        return results[:top_n]
