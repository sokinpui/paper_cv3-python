import glob
import os

import gradio as gr

from config import METRICS_CONFIG
from event_handlers import (
    create_click_handler,
    run_analysis,
    run_and_plot_k_distance,
    toggle_annotations,
)
from ui_helpers import calculate_vector_distance, clear_vector_inputs

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
                        # "Clustering (K-means)",
                        # "Clustering (Spectral)",
                        # "Clustering (Hierarchical)",
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
                    oklab_blur_sigma_input = gr.Slider(
                        minimum=0.0,
                        maximum=3.0,
                        value=0.8,
                        step=0.1,
                        label="Oklab Blur Sigma",
                        info="Strength of Gaussian blur to reduce noise. 0 = disabled.",
                    )

                with gr.Row():
                    oklab_w_l = gr.Slider(
                        minimum=0.0,
                        maximum=5.0,
                        value=1.0,
                        step=0.1,
                        label="L Weight (Lightness)",
                        info="Sensitivity to brightness changes.",
                    )
                    oklab_w_a = gr.Slider(
                        minimum=0.0,
                        maximum=5.0,
                        value=1.0,
                        step=0.1,
                        label="a Weight (Green-Red)",
                        info="Sensitivity to Green/Red shifts.",
                    )
                    oklab_w_b = gr.Slider(
                        minimum=0.0,
                        maximum=5.0,
                        value=1.0,
                        step=0.1,
                        label="b Weight (Blue-Yellow)",
                        info="Sensitivity to Blue/Yellow shifts.",
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
                    "30x30 (Tiny)": (30, 30),
                    "50x50 (Small)": (50, 50),
                    "100x100 (Medium)": (100, 100),
                    "150x150 (Large)": (150, 150),
                    "200x200 (X-Large)": (200, 200),
                    "250x250 (Huge)": (250, 250),
                }
                unit_preset_input = gr.Radio(
                    choices=list(unit_size_presets.keys()),
                    value="50x50 (Small)",
                    label="Unit Size Presets",
                )

                with gr.Row():
                    h_input = gr.Number(value=50, label="Unit Height", precision=0)
                    w_input = gr.Number(value=50, label="Unit Width", precision=0)

                with gr.Row():
                    overlap_input = gr.Slider(
                        minimum=0.0,
                        maximum=0.9,
                        value=0.0,
                        step=0.05,
                        label="Overlap Ratio",
                    )

                power_transform_input = gr.Slider(
                    minimum=0.1,
                    maximum=5.0,
                    value=1.0,
                    step=0.1,
                    label="Power Transformation",
                    info="Raise distance to a power (distance^n). >1 exaggerates large distances, <1 flattens them.",
                )

                def update_unit_size(preset_key):
                    h, w = unit_size_presets[preset_key]
                    return h, w

                unit_preset_input.change(
                    fn=update_unit_size,
                    inputs=unit_preset_input,
                    outputs=[h_input, w_input],
                )

                # Dynamic Settings
                top_n_input = gr.Number(
                    value=5, label="Top N Units", precision=0, visible=False
                )
                sort_input = gr.Dropdown(
                    choices=[
                        "mean",
                        "median",
                        "std_dev",
                        "min_score",
                        "max_score",
                        "l2_norm",
                    ],
                    value="mean",
                    label="Sort By Stat",
                    visible=False,
                )

                desc_input = gr.Checkbox(
                    value=True,
                    label="Sort Descending (High Score = Significant)",
                    visible=False,
                )

                k_input = gr.Slider(
                    minimum=2,
                    maximum=8,
                    value=2,
                    step=1,
                    label="K Clusters (for Clustering)",
                    visible=True,
                )

                h_method_input = gr.Dropdown(
                    choices=["ward", "single", "complete", "average"],
                    value="ward",
                    label="Linkage Method (Hierarchical Only)",
                    visible=False,
                )

                # DBSCAN Settings
                dbscan_eps_input = gr.Number(
                    value=0.0,
                    label="DBSCAN Eps",
                    info="Distance threshold. Set to 0.0 for auto-detection (K-needle).",
                    visible=False,
                )
                dbscan_min_input = gr.Number(
                    value=4,
                    label="DBSCAN Min Samples",
                    precision=0,
                    visible=False,
                )

                cluster_label_mode = gr.Dropdown(
                    choices=["1-NN Distance", "k-Distance", "Mean Score", "Max Score"],
                    value="1-NN Distance",
                    label="Map Label Value",
                    visible=False,
                )

                cluster_metric_input = gr.Dropdown(
                    choices=["mean", "std_dev", "threshold"],
                    value="mean",
                    label="Clustering Score",
                    visible=False,
                )

                cluster_threshold_n_input = gr.Number(
                    value=1.0,
                    label="Threshold N (Mean + N * Std)",
                    visible=False,
                )

                cluster_show_scores = gr.Checkbox(
                    value=False, label="Show Units' Stat", visible=True
                )

                # Visibility Logic
                def update_visibility(mode, metric, dbscan_eps):
                    is_top_n = mode == "Top N"
                    is_all = mode == "All Units"
                    is_heatmap = mode == "Heatmap"
                    is_cluster = mode == "Clustering"
                    is_cluster2 = mode == "Clustering (K-means)"
                    is_cluster_h = mode == "Clustering (Hierarchical)"
                    is_cluster_s = mode == "Clustering (Spectral)"
                    is_cluster_d = mode == "Clustering (DBSCAN)"
                    is_cluster_d2 = mode == "Clustering (DBSCAN2)"

                    is_threshold = is_cluster and (metric == "threshold")
                    is_dbscan_mode = is_cluster_d or is_cluster_d2
                    is_clustering_any = (
                        is_cluster
                        or is_cluster2
                        or is_cluster_h
                        or is_cluster_s
                        or is_cluster_d
                        or is_cluster_d2
                    )

                    return (
                        gr.update(visible=is_top_n),  # top_n
                        gr.update(visible=(is_top_n or is_all or is_heatmap)),  # sort
                        gr.update(visible=(is_top_n or is_all)),  # desc
                        gr.update(
                            visible=(is_cluster or is_cluster2 or is_cluster_s)
                            and not is_dbscan_mode
                        ),  # k (Hidden for Hierarchical)
                        gr.update(visible=is_clustering_any),  # show_scores
                        gr.update(
                            visible=is_clustering_any
                        ),  # cluster_label_mode (show whenever show_scores is relevant)
                        gr.update(
                            visible=is_cluster
                        ),  # cluster_metric (only for stats clustering)
                        gr.update(visible=is_cluster_h),  # linkage method
                        gr.update(visible=is_threshold),  # threshold_n
                        gr.update(visible=is_dbscan_mode),  # dbscan eps
                        gr.update(visible=is_dbscan_mode),  # dbscan min
                    )

                mode_input.change(
                    fn=update_visibility,
                    inputs=[mode_input, cluster_metric_input, dbscan_eps_input],
                    outputs=[
                        top_n_input,
                        sort_input,
                        desc_input,
                        k_input,
                        cluster_show_scores,
                        cluster_label_mode,
                        cluster_metric_input,
                        h_method_input,
                        cluster_threshold_n_input,
                        dbscan_eps_input,
                        dbscan_min_input,
                    ],
                )

                for comp in [cluster_metric_input, dbscan_eps_input]:
                    comp.change(
                        fn=update_visibility,
                        inputs=[mode_input, cluster_metric_input, dbscan_eps_input],
                        outputs=[
                            top_n_input,
                            sort_input,
                            desc_input,
                            k_input,
                            cluster_show_scores,
                            cluster_label_mode,
                            cluster_metric_input,
                            h_method_input,
                            cluster_threshold_n_input,
                            dbscan_eps_input,
                            dbscan_min_input,
                        ],
                    )

                # Trigger visibility update on load to match default mode
                demo.load(
                    fn=update_visibility,
                    inputs=[mode_input, cluster_metric_input, dbscan_eps_input],
                    outputs=[
                        top_n_input,
                        sort_input,
                        desc_input,
                        k_input,
                        cluster_show_scores,
                        cluster_label_mode,
                        cluster_metric_input,
                        h_method_input,
                        cluster_threshold_n_input,
                        dbscan_eps_input,
                        dbscan_min_input,
                    ],
                )

            with gr.Column(scale=3):
                gr.Markdown("### 📊 Analysis Results (By Distance Function)")

                # Dynamically create output rows for each metric
                metric_outputs = []
                metric_images = []  # Keep track of image components to bind events

                for name, _ in METRICS_CONFIG:
                    m_header = gr.Markdown(f"**{name}**", visible=False)
                    m_img = gr.Image(
                        label=f"Result ({name})", type="numpy", visible=False
                    )
                    m_perf = gr.Markdown(value="Waiting...", visible=False)
                    metric_outputs.extend([m_header, m_img, m_perf])
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

                gr.Markdown("### 📈 K-Distance Graph (Global Analysis)")
                with gr.Group():
                    gr.Markdown(
                        "Analyze the distance matrix to find the optimal DBSCAN Eps (the 'Elbow' in the graph). This runs a separate, dedicated analysis."
                    )
                    with gr.Row():
                        plot_metric_select = gr.Dropdown(
                            choices=[m[0] for m in METRICS_CONFIG],
                            value="Oklab",
                            label="Select Metric to Plot",
                        )
                        plot_min_samples = gr.Number(
                            value=4, label="K-Dist. Min Samples", precision=0
                        )
                        plot_eps_input = gr.Number(
                            value=0.0,
                            label="Plot Eps Threshold",
                            info="If > 0.0, draws a horizontal line. If <= 0.0, auto-detects and plots (K-needle).",
                        )
                    plot_run_btn = gr.Button(
                        "📊 Generate K-Distance Plot", variant="primary"
                    )
                    score_dist_plot = gr.Plot(label="Score Distribution")

                    plot_run_btn.click(
                        fn=run_and_plot_k_distance,
                        inputs=[
                            img_input,
                            h_input,
                            w_input,
                            overlap_input,
                            plot_metric_select,
                            power_transform_input,
                            plot_min_samples,
                            plot_eps_input,
                            ssim_k1_input,
                            ssim_k2_input,
                            oklab_blur_sigma_input,
                            oklab_w_l,
                            oklab_w_a,
                            oklab_w_b,
                        ],
                        outputs=[score_dist_plot],
                    )

        # Common inputs for all buttons
        common_inputs = [
            img_input,
            h_input,
            w_input,
            top_n_input,
            sort_input,
            desc_input,
            overlap_input,
            mode_input,
            k_input,
            cluster_show_scores,
            cluster_label_mode,
            cluster_metric_input,
            cluster_threshold_n_input,
            distance_funcs_input,
            h_method_input,
            dbscan_eps_input,
            dbscan_min_input,
            power_transform_input,
            ssim_k1_input,
            ssim_k2_input,
            oklab_w_l,
            oklab_w_a,
            oklab_w_b,
            oklab_blur_sigma_input,
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
            inputs=[analysis_state],
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
