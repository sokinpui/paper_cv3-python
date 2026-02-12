import argparse
import sys

import numpy as np
import torch

from analyzer import PatchAnalyzer
from globals import DEVICE
from metrics import CosineMetric, OklabMetric, SSIMMetric
from processor import ImageProcessor
from utils import calculate_oklab_range


class ImageInspector:
    def __init__(
        self,
        image_path: str,
        unit_size: int = 512,
        metric_name: str = "oklab",
        threshold: float = float("inf"),
    ):
        # Configure numpy to print full vectors on one line without truncation
        np.set_printoptions(
            linewidth=np.inf, threshold=sys.maxsize, suppress=True, precision=4
        )

        self.processor = ImageProcessor(DEVICE)
        self.metric_name = metric_name
        self.oklab_threshold = threshold
        self._init_metric(metric_name)
        self.image_path = image_path
        self.unit_size = unit_size
        self.image_tensor = None
        self.patches = None
        self.grid_shape = None
        self.stats = []
        self.matrix = None

        self._load_and_process()

    def _init_metric(self, name: str):
        name = name.lower()
        if name == "oklab":
            self.metric = OklabMetric(threshold=self.oklab_threshold)
        elif name == "ssim":
            self.metric = SSIMMetric()
        elif name == "cosine":
            self.metric = CosineMetric()
        else:
            raise ValueError(f"Unknown metric: {name}")
        self.analyzer = PatchAnalyzer(self.metric)

    def set_metric(self, name: str):
        try:
            self._init_metric(name)
            self.metric_name = name.lower()
            self._load_and_process()
            print(f"Metric updated to {self.metric_name}. Recomputed matrix.")
        except ValueError as e:
            print(f"Error: {e}")

    def set_threshold(self, n: float):
        self.oklab_threshold = n
        if self.metric_name == "oklab":
            self._init_metric("oklab")
            self._load_and_process()
            print(f"Oklab threshold updated to {n}. Recomputed matrix.")
        else:
            print(
                f"Threshold stored as {n}, but current metric is {self.metric_name} (Threshold only affects Oklab)."
            )

    def _load_and_process(self):
        if self.image_tensor is None:
            self.image_tensor = self.processor.load_image(self.image_path)

        self.patches, self.grid_shape, _ = self.processor.extract_patches(
            self.image_tensor, self.unit_size, self.unit_size, overlap_ratio=0.0
        )

        self.stats, self.matrix, _ = self.analyzer.analyze(
            self.patches,
            self.grid_shape,
            top_n=self.patches.shape[0],
            cluster_on_matrix=True,
            clustering_algorithm="dbscan",
            eps=0.0,  # Auto-detect
            min_samples=4,
        )

    def set_size(self, n: int):
        h, w = self.image_tensor.shape[2], self.image_tensor.shape[3]
        if n > h or n > w:
            print(f"Error: Unit size {n} exceeds image dimensions ({w}x{h}).")
            return

        if n <= 1:
            print("Error: Size must be positive.")
            return
        self.unit_size = n
        self._load_and_process()
        print(f"Unit size updated to {n}x{n}. Recomputed matrix.")

    def show_info(self):
        rows, cols = self.grid_shape
        n_units = self.patches.shape[0]
        print(f"Image: {self.image_path}")
        print(f"Metric: {self.metric_name}")
        print(f"Unit Size: {self.unit_size}x{self.unit_size}")
        print(f"Grid: {rows} rows x {cols} columns ({n_units} total units)")
        if self.metric_name == "oklab":
            multiplier = calculate_oklab_range(self.unit_size, self.unit_size)
            print(f"Oklab Threshold: {self.oklab_threshold}")
            print(f"Oklab Multiplier: {multiplier:.4f}")
        print(f"Matrix Shape: {self.matrix.shape}")

    def print_vector(self, row: int = None, col: int = None):
        if row is None or col is None:
            print(f"--- Distance Matrix (All {self.patches.shape[0]} Units) ---")
            matrix_np = self.matrix.cpu().numpy()
            for i, vector in enumerate(matrix_np):
                r, c = divmod(i, self.grid_shape[1])
                print(f"Unit Index {i} [Row {r}, Col {c}]:")
                print(f"Sum: {np.nansum(vector):.4f}")
                print(vector)
                print("-" * 40)
            return

        rows, cols = self.grid_shape
        if not (0 <= row < rows and 0 <= col < cols):
            print(
                f"Error: Coordinates ({row}, {col}) out of range for grid {rows}x{cols}"
            )
            return

        idx = row * cols + col
        vector = self.matrix[idx].cpu().numpy()

        print(f"--- Distance Vector for Unit ({row}, {col}) [Index {idx}] ---")
        print(f"Shape: {vector.shape}")
        print(f"Sum: {np.nansum(vector):.4f}")
        print(vector)

    def print_clusters(self):
        if not self.stats:
            print("No analysis data available.")
            return

        clusters = {}
        for u in self.stats:
            cid = u.cluster_id
            if cid not in clusters:
                clusters[cid] = []
            clusters[cid].append(u)

        print(f"--- Clustering Summary (Total Units: {len(self.stats)}) ---")
        for cid in sorted(clusters.keys()):
            label = f"Cluster {cid}" if cid != -1 else "Noise (Cluster -1)"
            units = clusters[cid]
            indices = [f"{u.index}({u.row},{u.col})" for u in units]
            print(f"\n{label} [Count: {len(units)}]")
            print(f"Units: {', '.join(indices)}")

    def print_noise_units(self):
        noise = [u for u in self.stats if u.cluster_id == -1]

        if not noise:
            print("No noise units detected (all units belong to clusters).")
            return

        print(f"--- Noise Units (Total: {len(noise)}) ---")
        for u in noise:
            print(f"Unit Index {u.index} [Row {u.row}, Col {u.col}]")


def print_help():
    print("\nAvailable Commands:")
    print("  info              : Show unit size and grid dimensions")
    print("  size <n>          : Set unit size to n x n (default 512)")
    print("  m / metric <name> : Set metric (oklab, ssim, cosine)")
    print("  c / cluster       : Show units grouped by cluster")
    print("  set <n>           : Set Oklab distance threshold")
    print("  p / print         : Print distance matrix of all units")
    print("  w / which         : Print indices of noise units (Cluster -1)")
    print("  p <row> <col>     : Print distance vector of a specific unit")
    print("  help              : Show this help")
    print("  exit / quit       : Close the inspector")


def main():
    parser = argparse.ArgumentParser(description="Interactive Image Unit Inspector")
    parser.add_argument("image_path", type=str, help="Path to the image file")
    args = parser.parse_args()

    try:
        inspector = ImageInspector(args.image_path)
    except Exception as e:
        print(f"Failed to initialize: {e}")
        return

    print(f"Inspecting: {args.image_path}")
    print_help()

    while True:
        try:
            cmd_input = input("\n> ").strip().lower()
            if not cmd_input:
                continue

            parts = cmd_input.split()
            cmd = parts[0]

            if cmd in ["exit", "quit"]:
                break

            if cmd == "help":
                print_help()
                continue

            if cmd == "info":
                inspector.show_info()
                continue

            if cmd == "size" and len(parts) > 1:
                inspector.set_size(int(parts[1]))
                continue

            if cmd in ["m", "metric"] and len(parts) > 1:
                inspector.set_metric(parts[1])
                continue

            if cmd == "set" and len(parts) > 1:
                inspector.set_threshold(float(parts[1]))
                continue

            if cmd in ["p", "print"]:
                if len(parts) == 3:
                    inspector.print_vector(int(parts[1]), int(parts[2]))
                else:
                    inspector.print_vector()
                continue

            if cmd in ["c", "cluster"]:
                inspector.print_clusters()
                continue

            if cmd in ["w", "which"]:
                inspector.print_noise_units()
                continue

            print(f"Unknown command: {cmd}. Type 'help' for info.")

        except ValueError:
            print("Error: Invalid numeric argument.")
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
