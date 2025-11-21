import argparse
import json
import sys

import torch

from analyzer import PatchAnalyzer
from metrics import (
    CIELabMetric,
    GradientColorMetric,
    HistogramMetric,
    LabMomentsMetric,
    SSIMMetric,
    TextureColorMetric,
)
from processor import ImageProcessor


def main():
    parser = argparse.ArgumentParser(description="GPU Image Unit Detection")
    parser.add_argument("image_path", type=str, help="Path to input image")
    parser.add_argument("--height", type=int, required=True, help="Unit Height")
    parser.add_argument("--width", type=int, required=True, help="Unit Width")
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.0,
        help="Overlap ratio between patches (0.0 to 0.9)",
    )
    parser.add_argument(
        "--metric",
        "-m",
        type=str,
        choices=["ssim", "cielab", "moments", "texture", "grad_color", "hist"],
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
        "--ascending",
        "-a",
        action="store_true",
        help="Sort ascending (default descending)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Path to save annotated image (e.g., output.png)",
    )
    parser.add_argument(
        "--brightness",
        type=float,
        default=0.0,
        help="Adjust brightness (-1.0 to 1.0)",
    )
    parser.add_argument(
        "--contrast",
        type=float,
        default=1.0,
        help="Adjust contrast (0.0 to 3.0)",
    )
    parser.add_argument(
        "--blur",
        type=float,
        default=0.0,
        help="Gaussian Blur radius (0 to 5) to reduce grain noise",
    )
    parser.add_argument(
        "--sharpen",
        type=float,
        default=0.0,
        help="Sharpen strength (0.0 to 2.0) to highlight edges",
    )
    parser.add_argument(
        "--clahe",
        type=float,
        default=0.0,
        help="CLAHE limit (0.0 to 5.0) for local contrast",
    )
    parser.add_argument(
        "--grayscale",
        action="store_true",
        help="Convert image to grayscale before processing",
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
        metric = SSIMMetric()
    elif args.metric == "moments":
        metric = LabMomentsMetric()
    elif args.metric == "texture":
        metric = TextureColorMetric()
    elif args.metric == "grad_color":
        metric = GradientColorMetric()
    elif args.metric == "hist":
        metric = HistogramMetric()
    else:
        metric = CIELabMetric()

    # Both metrics now use High Score = Different. Default sort is Descending (False).
    ascending = args.ascending

    analyzer = PatchAnalyzer(metric)

    # 3. Execution Pipeline
    try:
        print("Loading and tiling image...")
        image_tensor = processor.load_image(args.image_path)

        # Global Adjustments
        if args.brightness != 0.0 or args.contrast != 1.0:
            print(
                f"Adjusting image: Brightness={args.brightness}, Contrast={args.contrast}"
            )
            image_tensor = processor.adjust_image(
                image_tensor, args.brightness, args.contrast
            )

        # Advanced Texture Preprocessing
        image_tensor = processor.apply_preprocessing(
            image_tensor, args.blur, args.sharpen, args.clahe, args.grayscale
        )

        patches, grid_shape, strides = processor.extract_patches(
            image_tensor, args.height, args.width, args.overlap
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
                image_tensor, top_units, args.height, args.width, grid_shape, strides, args.output
            )

    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
