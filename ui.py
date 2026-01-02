import glob
import os

import gradio as gr

from config import METRICS_CONFIG
from event_handlers import (
    create_click_handler,
    run_analysis,
    toggle_annotations,
    update_annotation_settings,
)
from ui_helpers import (
    calculate_vector_distance,
    clear_vector_inputs,
)

# Compatibility for older Gradio versions (Pre-5.0)
if not hasattr(gr, "Modal"):
    print(
        "Warning: Gradio version does not support Modals. Falling back to inline Group."
    )
    gr.Modal = gr.Group


def create_ui(input_dir=None):
    with gr.Blocks(title="GPU Image Anomaly Detection") as demo:
        gr.Markdown("# 🔍 GPU Image Unit Detection")
        gr.Markdown(
            "Upload an image to find significant/unique blocks using CUDA acceleration."
        )

        with gr.Row():
            with gr.Column(scale=1):
                # Action Buttons
                mode_input = gr.Radio(
                    choices=[
                        "Clustering (DBSCAN)",
                        "Clustering (DBSCAN2)",
                    ],
                    value="Clustering (DBSCAN)",
                    label="Analysis Mode",
                )
                with gr.Row():
                    btn_run = gr.Button("🚀 Run Analysis", variant="primary")
                    btn_toggle_annotations = gr.Button("🎨 Toggle Annotations")

                # Distance Function Selection
                metric_names = [m[0] for m in METRICS_CONFIG]
                distance_funcs_input = gr.CheckboxGroup(
                    choices=metric_names,
                    value=["Oklab"],
                    label="Distance Functions",
                )

                with gr.Group(visible=True) as oklab_options:
                    oklab_p_norm_input = gr.Slider(
                        minimum=1.0,
                        maximum=10.0,
                        value=2.0,
                        step=1.0,
                        label="Norm Degree (P-Value)",
                        info="2.0=Euclidean. Increase to >2.0 to suppress small distributed noise and highlight sharp deviations.",
                    )
                    oklab_explosion_k_input = gr.Slider(
                        minimum=0.0,
                        maximum=20.0,
                        value=0.0,
                        step=0.5,
                        label="Pixel Explosion (k)",
                        info="Exponential sensitivity. 0=Off. High values make the distance explode for even single pixel differences.",
                    )

                with gr.Group(visible=False) as ssim_options:
                    ssim_k1_input = gr.Slider(
                        minimum=0.001,
                        maximum=0.1,
                        value=0.01,
                        step=0.005,
                        label="SSIM K1",
                        info="Controls sensitivity to brightness. Higher = less sensitive.",
                    )
                    ssim_k2_input = gr.Slider(
                        minimum=0.001,
                        maximum=0.10,
                        value=0.03,
                        step=0.005,
                        label="SSIM K2",
                        info="Controls sensitivity to contrast/structure. Higher = less sensitive.",
                    )
                    with gr.Row():
                        ssim_alpha_input = gr.Slider(
                            minimum=0.05,
                            maximum=5.0,
                            value=1.0,
                            step=0.05,
                            label="Alpha (Lum Power)",
                            info="Power applied to Luminance similarity. >1 suppresses small brightness matches.",
                        )
                        ssim_beta_input = gr.Slider(
                            minimum=0.05,
                            maximum=5.0,
                            value=1.0,
                            step=0.05,
                            label="Beta (Struct Power)",
                            info="Power applied to Structure similarity. >1 suppresses small texture matches.",
                        )

                def update_metric_options_visibility(selected_metrics):
                    return gr.update(visible="Oklab" in selected_metrics), gr.update(
                        visible="SSIM" in selected_metrics
                    )

                distance_funcs_input.change(
                    fn=update_metric_options_visibility,
                    inputs=distance_funcs_input,
                    outputs=[oklab_options, ssim_options],
                )

                gr.Markdown("### Settings")

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
                            examples_per_page=20,
                        )

                unit_size_presets = {
                    "30x30": (30, 30),
                    "50x50": (50, 50),
                    "100x100": (100, 100),
                    "150x150": (150, 150),
                    "200x200": (200, 200),
                    "250x250": (250, 250),
                    "512x512": (512, 512),
                }
                unit_preset_input = gr.Radio(
                    choices=list(unit_size_presets.keys()),
                    value="512x512",
                    label="Unit Size Presets",
                )

                with gr.Row():
                    h_input = gr.Number(value=512, label="Unit Height", precision=0)
                    w_input = gr.Number(value=512, label="Unit Width", precision=0)

                with gr.Row():
                    overlap_input = gr.Slider(
                        minimum=0.0,
                        maximum=0.9,
                        value=0.0,
                        step=0.05,
                        label="Overlap Ratio",
                    )

                with gr.Row():
                    power_transform_input = gr.Slider(
                        minimum=0.1,
                        maximum=5.0,
                        value=1.0,
                        step=0.1,
                        label="Power Transformation",
                        info="distance^n. >1 exaggerates differences.",
                    )
                    sigmoid_k_input = gr.Slider(
                        minimum=0.0,
                        maximum=20.0,
                        value=0.0,
                        step=0.5,
                        label="Sigmoid Contrast (k)",
                        info="Sigmoid Stretch. 0=Off, 10=Strong Contrast. Pushes similar units to 0, different to 1.",
                    )

                def update_unit_size(preset_key):
                    h, w = unit_size_presets[preset_key]
                    return h, w

                unit_preset_input.change(
                    fn=update_unit_size,
                    inputs=unit_preset_input,
                    outputs=[h_input, w_input],
                )

                # DBSCAN Settings
                dbscan_eps_input = gr.Number(
                    value=0.0,
                    label="DBSCAN Eps",
                    info="Distance threshold. Set to 0.0 for auto-detection (K-needle).",
                    visible=True,
                )
                dbscan_min_input = gr.Number(
                    value=4,
                    label="DBSCAN Min Samples",
                    precision=0,
                    visible=True,
                )

                cluster_label_mode = gr.Dropdown(
                    choices=["1-NN Distance", "k-Distance", "Mean Score", "Max Score"],
                    value="1-NN Distance",
                    label="Map Label Value",
                    visible=True,
                )

                cluster_show_scores = gr.Checkbox(
                    value=False, label="Show Units' Stat", visible=True
                )

            with gr.Column(scale=3):
                gr.Markdown("### 📊 Analysis Results (By Distance Function)")

                # Dynamically create output rows for each metric
                metric_outputs = []
                metric_images = []  # Keep track of image components to bind events

                for name, _ in METRICS_CONFIG:
                    # Using a group to manage visibility of the whole block for a metric
                    with gr.Group(visible=False) as metric_group:
                        gr.Markdown(f"**{name}**")
                        m_img = gr.Image(label=f"Result ({name})", type="numpy")
                        m_perf = gr.Markdown(value="Waiting...")

                    # These are the components that run_analysis needs to update
                    metric_outputs.extend([metric_group, m_img, m_perf])
                    metric_images.append((name, m_img))

                gr.Markdown("### 📐 Vector Calculator")
                with gr.Group():
                    with gr.Row():
                        vc_a = gr.Textbox(label="Vector A", placeholder="1.0, 2.0, ...")
                        vc_b = gr.Textbox(label="Vector B", placeholder="3.0, 4.0, ...")
                    with gr.Row():
                        vc_btn = gr.Button("Calculate Distance", variant="primary")
                        vc_clear = gr.Button("Clear")
                    vc_res = gr.Textbox(label="Results", lines=8)
                    vc_btn.click(
                        calculate_vector_distance,
                        inputs=[vc_a, vc_b],
                        outputs=vc_res,
                    )
                    vc_clear.click(
                        clear_vector_inputs, inputs=None, outputs=[vc_a, vc_b, vc_res]
                    )

                gr.Markdown("### 🔬 Unit Inspector")
                gr.Markdown(
                    "Click on a unit in the result image to see it and its neighbors here."
                )
                unit_inspector_gallery = gr.Gallery(
                    label="Clicked Unit (Center) and Neighbors",
                    show_label=False,
                    columns=3,
                    rows=3,
                    object_fit="contain",
                    height="auto",
                )

                # perf_output = gr.Markdown() # Removed global perf
                json_output = gr.Code(language="json", label="Statistics")
                analysis_state = gr.State({})  # Store matrix data per session

        # Common inputs for all buttons
        common_inputs = [
            img_input,
            h_input,
            w_input,
            overlap_input,
            mode_input,
            cluster_show_scores,
            cluster_label_mode,
            distance_funcs_input,
            dbscan_eps_input,
            dbscan_min_input,
            power_transform_input,
            sigmoid_k_input,
            ssim_k1_input,
            ssim_k2_input,
            oklab_p_norm_input,
            oklab_explosion_k_input,
            ssim_alpha_input,
            ssim_beta_input,
            analysis_state,
        ]
        common_outputs = metric_outputs + [json_output, analysis_state]

        btn_run.click(
            fn=run_analysis,
            inputs=common_inputs,
            outputs=common_outputs,
        )

        btn_toggle_annotations.click(
            fn=toggle_annotations,
            inputs=[analysis_state, cluster_show_scores, cluster_label_mode],
            outputs=[m[1] for m in metric_images] + [analysis_state],
        )

        # Enable dynamic updating of visualization settings
        for comp in [cluster_show_scores, cluster_label_mode]:
            comp.change(
                fn=update_annotation_settings,
                inputs=[analysis_state, cluster_show_scores, cluster_label_mode],
                outputs=[m[1] for m in metric_images] + [analysis_state],
            )

        # Wire Select/Click Events for Result Images
        for name, img_comp in metric_images:
            img_comp.select(
                fn=create_click_handler(name),
                inputs=[analysis_state, vc_a, vc_b],
                outputs=[vc_a, vc_b, vc_res, unit_inspector_gallery, img_comp],
            )

    return demo
