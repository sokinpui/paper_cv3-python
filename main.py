import argparse
import json
import sys

import torch

from analyzer import PatchAnalyzer
from metrics import CIELabMetric, SSIMMetric
from processor import ImageProcessor


def main():
    parser = argparse.ArgumentParser(description="GPU Image Unit Detection")
    parser.add_argument("image_path", type=str, help="Path to input image")
    parser.add_argument("--height", type=int, required=True, help="Unit Height")
    parser.add_argument("--width", type=int, required=True, help="Unit Width")
    parser.add_argument(
        "--metric",
        "-m",
        type=str,
        choices=["ssim", "cielab"],
        default="ssim",
        help="Comparison metric",
    )
    parser.add_argument(
        "--top_n", type=int, default=5, help="Number of significant units to output"
    )
    parser.add_argument(
        "--sort_by",
        "-s",
        type=str,
        default="mean",
        choices=["mean", "median", "std_dev", "min_score", "max_score"],
        help="Stat to rank by",
    )
    parser.add_argument(
        "--descending",
        "-r",
        action="store_true",
        help="Sort descending (default ascending)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Path to save annotated image (e.g., output.png)",
    )

    args = parser.parse_args()

    # 1. Setup Device
    if not torch.cuda.is_available():
        print(
            "Error: CUDA is not available. This program requires an NVIDIA GPU.",
            file=sys.stderr,
        )
        sys.exit(1)

    device = torch.device("cuda")
    print(f"Using Device: {torch.cuda.get_device_name(0)}")

    # 2. Initialize Components
    processor = ImageProcessor(device)

    if args.metric == "ssim":
        # SSIM: Higher is more similar.
        # Significant = Low Mean SSIM (Unique) -> Ascending sort
        metric = SSIMMetric()
        default_asc = True
    else:
        # CIELab: Higher is more different (Distance).
        # Significant = High Mean Distance (Unique) -> Descending sort
        metric = CIELabMetric()
        default_asc = False

    # Override sort order if user specified
    ascending = args.descending is False if args.descending else default_asc

    analyzer = PatchAnalyzer(metric)

    # 3. Execution Pipeline
    try:
        print("Loading and tiling image...")
        image_tensor = processor.load_image(args.image_path)
        patches, grid_shape = processor.extract_patches(
            image_tensor, args.height, args.width
        )

        print(f"Extracted {patches.shape[0]} units. Grid: {grid_shape}")
        print("Computing pairwise matrix and statistics...")

        top_units = analyzer.analyze(
            patches,
            grid_shape,
            top_n=args.top_n,
            sort_by=args.sort_by,
            ascending=ascending,
        )

        # 4. Output
        print(f"\nTop {args.top_n} Significant Units (Ranked by {args.sort_by}):")
        print(json.dumps([u.to_dict() for u in top_units], indent=4))

        # 5. Visualization
        if args.output:
            processor.save_annotated_image(
                image_tensor, top_units, args.height, args.width, args.output
            )

    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
