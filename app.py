import argparse
import glob
import json
import os
import subprocess
import sys
import tempfile
import time

import gradio as gr
import torch

from analyzer import PatchAnalyzer
from metrics import CIELabMetric, SSIMMetric
from processor import ImageProcessor

# 1. Initialize CUDA Device once
try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"Web UI running on: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}"
    )
except Exception as e:
    print(f"Warning: {e}")
    device = torch.device("cpu")


def run_paddleseg(image_path, paddle_args):
    """
    Executes PaddleSeg prediction in a subprocess.
    """
    if (
        not paddle_args.get("root")
        or not paddle_args.get("config")
        or not paddle_args.get("model")
    ):
        return None, "PaddleSeg arguments not provided."

    # Create a temporary directory for this specific run's output
    output_dir = tempfile.mkdtemp()
    # We need absolute path for the image because we change CWD to paddle root
    abs_img_path = os.path.abspath(image_path)

    python_exec = paddle_args.get("python") or sys.executable

    # Construct command
    cmd = [
        python_exec,
        "tools/predict.py",
        "--config",
        paddle_args["config"],
        "--model_path",
        paddle_args["model"],
        "--image_path",
        abs_img_path,
        "--is_slide",
        "--crop_size",
        "512",
        "512",
        "--stride",
        "256",
        "256",
        "--custom_color",
        "255",
        "255",
        "255",
        "0",
        "0",
        "255",
        "--save_dir",
        output_dir,
    ]

    try:
        # Run predict.py inside the PaddleSeg root directory
        subprocess.run(cmd, cwd=paddle_args["root"], check=True, capture_output=True)

        # Look for the result in 'added_prediction' (overlay)
        # structure: output_dir/added_prediction/filename.bmp
        target_dir = os.path.join(output_dir, "added_prediction")
        if os.path.exists(target_dir):
            files = os.listdir(target_dir)
            if files:
                # Return full path to the result image
                return os.path.join(target_dir, files[0]), "Success"
        return None, "No output image found in PaddleSeg results."
    except Exception as e:
        print(f"PaddleSeg Error: {e}")
        return None, f"PaddleSeg Failed: {str(e)}"


def run_analysis(
    image_path, height, width, metric_name, top_n, sort_by, descending, paddle_args=None
):
    """
    The core function called when user clicks 'Run Detection'
    """
    if image_path is None:
        return None, None, "Please upload an image.", ""

    try:
        # Setup Components
        processor = ImageProcessor(device)

        if metric_name == "SSIM (Structure)":
            metric = SSIMMetric()
        else:
            metric = CIELabMetric()

        analyzer = PatchAnalyzer(metric)

        # Pipeline
        # 1. Load
        image_tensor = processor.load_image(image_path)

        # 2. Tile
        patches, grid_shape = processor.extract_patches(
            image_tensor, int(height), int(width)
        )

        # 3. Analyze
        t0 = time.time()
        stats = analyzer.analyze(
            patches,
            grid_shape,
            top_n=int(top_n),
            sort_by=sort_by,
            ascending=not descending,
        )
        t1 = time.time()

        # 4. Run PaddleSeg (Optional)
        paddle_img = None
        paddle_status = "Not Configured"
        if paddle_args and paddle_args.get("root"):
            paddle_img, paddle_status = run_paddleseg(image_path, paddle_args)

        # Performance Stats
        duration = t1 - t0
        N = patches.shape[0]
        total_pairs = N * N
        cps = total_pairs / duration if duration > 0 else 0

        perf_text = (
            f"### ⚡ Performance Metrics\n"
            f"- **Analysis Time:** {duration:.4f} seconds\n"
            f"- **Total Units:** {N} (Grid: {grid_shape})\n"
            f"- **Total Comparisons:** {total_pairs:,}\n"
            f"- **Throughput:** {cps:,.0f} comparisons/sec\n"
            f"- **PaddleSeg Status:** {paddle_status}"
        )

        # 5. Visualize (My Detection)
        result_image = processor.get_annotated_rgb(
            image_tensor, stats, int(height), int(width)
        )

        # 6. Format JSON result
        json_result = json.dumps([u.to_dict() for u in stats], indent=4)

        return result_image, paddle_img, json_result, perf_text

    except Exception as e:
        import traceback

        traceback.print_exc()
        return None, None, f"Error: {str(e)}", ""


# --- Build the UI ---


def create_ui(input_dir=None, paddle_args=None):
    with gr.Blocks(title="GPU Image Anomaly Detection") as demo:
        gr.Markdown("# 🔍 GPU Image Unit Detection")
        gr.Markdown(
            "Upload an image to find significant/unique blocks using CUDA acceleration."
        )

        with gr.Row():
            with gr.Column(scale=1):
                # Input Controls
                img_input = gr.Image(type="filepath", label="Input Image")

                # Display examples from server directory if provided
                if input_dir and os.path.isdir(input_dir):
                    exts = [
                        "*.bmp",
                        "*.png",
                        "*.jpg",
                        "*.jpeg",
                        "*.tiff",
                        "*.BMP",
                        "*.PNG",
                        "*.JPG",
                        "*.JPEG",
                    ]
                    server_images = []
                    for ext in exts:
                        server_images.extend(glob.glob(os.path.join(input_dir, ext)))
                    server_images = sorted(server_images)

                    if server_images:
                        gr.Examples(
                            examples=server_images,
                            inputs=img_input,
                            label=f"Select from Server Directory: {input_dir}",
                        )

                with gr.Row():
                    h_input = gr.Number(value=32, label="Unit Height", precision=0)
                    w_input = gr.Number(value=32, label="Unit Width", precision=0)

                metric_input = gr.Radio(
                    choices=["SSIM (Structure)", "CIELAB (Color)"],
                    value="SSIM (Structure)",
                    label="Comparison Metric",
                )

                with gr.Row():
                    top_n_input = gr.Number(value=5, label="Top N Units", precision=0)
                    sort_input = gr.Dropdown(
                        choices=["mean", "median", "std_dev", "min_score", "max_score"],
                        value="mean",
                        label="Sort By Stat",
                    )

                desc_input = gr.Checkbox(
                    value=False, label="Sort Descending (High Score = Significant)"
                )

                btn_run = gr.Button("🚀 Run Detection", variant="primary")

            with gr.Column(scale=2):
                # Outputs
                with gr.Row():
                    img_output = gr.Image(label="My Detection (Unit Stats)")
                    paddle_output = gr.Image(label="PaddleSeg Prediction")

                perf_output = gr.Markdown()
                json_output = gr.Code(language="json", label="Statistics")

        # Event Binding
        # Auto-adjust sort direction: CIELAB -> Descending (True), SSIM -> Ascending (False)
        metric_input.change(
            fn=lambda x: x == "CIELAB (Color)", inputs=metric_input, outputs=desc_input
        )

        # Wrapper to pass paddle_args
        def analysis_wrapper(img, h, w, metric, top_n, sort, desc):
            return run_analysis(img, h, w, metric, top_n, sort, desc, paddle_args)

        btn_run.click(
            fn=analysis_wrapper,
            inputs=[
                img_input,
                h_input,
                w_input,
                metric_input,
                top_n_input,
                sort_input,
                desc_input,
            ],
            outputs=[img_output, paddle_output, json_output, perf_output],
        )
    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPU Image Anomaly Detection Web UI")
    parser.add_argument(
        "-i",
        "--input_dir",
        type=str,
        help="Directory containing images to list in UI",
        default=None,
    )

    # PaddleSeg Arguments
    parser.add_argument(
        "--paddle_root", type=str, help="Path to PaddleSeg root directory"
    )
    parser.add_argument(
        "--paddle_config", type=str, help="Path to PaddleSeg config file"
    )
    parser.add_argument(
        "--paddle_model", type=str, help="Path to PaddleSeg model params"
    )
    parser.add_argument(
        "--paddle_python",
        type=str,
        help="Absolute path to Python executable for PaddleSeg",
        default=None,
    )

    args = parser.parse_args()

    paddle_settings = {
        "root": args.paddle_root,
        "config": args.paddle_config,
        "model": args.paddle_model,
        "python": args.paddle_python,
    }

    # server_name="0.0.0.0" makes it accessible from external IP (SSH tunnel/remote)
    demo = create_ui(args.input_dir, paddle_settings)
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
