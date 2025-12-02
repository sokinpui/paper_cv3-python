import argparse
import sys

import torch

from ui import create_ui

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPU Image Anomaly Detection Web UI")
    parser.add_argument(
        "-i",
        "--input_dir",
        type=str,
        help="Directory containing images to list in UI",
        default=None,
    )

    args = parser.parse_args()

    # server_name="0.0.0.0" makes it accessible from external IP (SSH tunnel/remote)
    demo = create_ui(args.input_dir)
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
