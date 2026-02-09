import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F

from globals import DEVICE
from metrics import HumanEyeColorMetric

# Configuration for the test bench
CANVAS_SIZE = 64  # Small size to keep vector representations readable/computable

def rgb_to_oklab_vec(img_tensor):
    """Reuses the Oklab conversion logic from metrics.py"""
    metric = HumanEyeColorMetric()
    # Ensure (1, 3, H, W)
    if img_tensor.dim() == 3:
        img_tensor = img_tensor.unsqueeze(0)
    return metric._rgb_to_oklab(img_tensor)

def process_image(img_np, pooling_n):
    """Applies preprocessing steps to get the final comparison vector."""
    # Convert to tensor (1, 3, H, W)
    img_t = torch.from_numpy(img_np).permute(2, 0, 1).float().unsqueeze(0).to(DEVICE) / 255.0
    
    # 1. Convert to Oklab
    oklab = rgb_to_oklab_vec(img_t)
    
    # 2. Pooling
    if pooling_n > 1:
        oklab = F.avg_pool2d(oklab, kernel_size=pooling_n, stride=pooling_n)
        
    # 3. Flatten
    return oklab.reshape(-1)

def calculate_distance(vA, vB, p, k):
    """Calculates the distance using Minkowski or Explosion logic."""
    diff = torch.abs(vA - vB)
    if k > 0:
        # Pixel Explosion: sum(exp(k * |a - b|) - 1)
        return torch.sum(torch.exp(k * diff) - 1)
    return torch.pow(torch.sum(torch.pow(diff, p)), 1/p)

class TestState:
    def __init__(self):
        self.imgA = np.full((CANVAS_SIZE, CANVAS_SIZE, 3), 255, dtype=np.uint8)
        self.imgB = np.full((CANVAS_SIZE, CANVAS_SIZE, 3), 255, dtype=np.uint8)

state = TestState()

def on_click(img_idx, evt: gr.SelectData, pooling_n, p_norm, explosion_k):
    y, x = evt.index[1], evt.index[0]
    
    # Update the correct image in state
    target_img = state.imgA if img_idx == 0 else state.imgB
    # Draw a 3x3 black dot
    target_img[max(0, y-1):y+2, max(0, x-1):x+2] = 0
    
    # Process vectors
    vA = process_image(state.imgA, pooling_n)
    vB = process_image(state.imgB, pooling_n)
    
    # Calculate Distance
    dist = calculate_distance(vA, vB, p_norm, explosion_k)
    
    # Format vector strings (showing snippet if too long)
    def fmt_vec(v):
        v_np = v.detach().cpu().numpy()
        if len(v_np) > 12:
            return f"Shape: {v_np.shape} | [{v_np[0]:.3f}, {v_np[1]:.3f}, ... , {v_np[-1]:.3f}]"
        return str(v_np)

    vD = vA - vB
    
    if explosion_k > 0:
        res_text = f"Final Distance (Explosion k={explosion_k}): {dist.item():.6f}\n"
    else:
        res_text = f"Final Distance (Minkowski p={p_norm}): {dist.item():.6f}\n"
        
    res_text += f"L2 Norm of Diff: {torch.norm(vD).item():.6f}"

    return (
        state.imgA, 
        state.imgB, 
        fmt_vec(vA), 
        fmt_vec(vB), 
        fmt_vec(vD), 
        res_text
    )

def reset_test():
    state.imgA = np.full((CANVAS_SIZE, CANVAS_SIZE, 3), 255, dtype=np.uint8)
    state.imgB = np.full((CANVAS_SIZE, CANVAS_SIZE, 3), 255, dtype=np.uint8)
    return state.imgA, state.imgB, "", "", "", ""

with gr.Blocks(title="Oklab Distance Test Bench") as test_ui:
    gr.Markdown("# 🧪 Oklab Distance Test Bench")
    gr.Markdown("Click on the white images to add black dots and observe the vector changes.")
    
    with gr.Row():
        with gr.Column():
            p_norm_input = gr.Slider(1, 10, value=2, step=1, label="P-Norm (Minkowski)")
            pool_input = gr.Slider(1, 8, value=1, step=1, label="Area Pooling (n)")
            explosion_k_input = gr.Slider(0, 100, value=0, step=0.5, label="Pixel Explosion (k)")
            btn_reset = gr.Button("Clear Canvases")

    with gr.Row():
        img_a_comp = gr.Image(value=state.imgA, label="Image A", width=256, height=256)
        img_b_comp = gr.Image(value=state.imgB, label="Image B", width=256, height=256)

    with gr.Group():
        vA_out = gr.Textbox(label="vA: vector representation of image A")
        vB_out = gr.Textbox(label="vB: vector representation of image B")
        vD_out = gr.Textbox(label="vD: vector representation of vA - vB")
        dist_out = gr.Textbox(label="Distance Analysis", lines=3)

    # Event Handlers
    img_a_comp.select(
        fn=lambda evt, n, p, k: on_click(0, evt, n, p, k),
        inputs=[pool_input, p_norm_input, explosion_k_input],
        outputs=[img_a_comp, img_b_comp, vA_out, vB_out, vD_out, dist_out]
    )
    
    img_b_comp.select(
        fn=lambda evt, n, p, k: on_click(1, evt, n, p, k),
        inputs=[pool_input, p_norm_input, explosion_k_input],
        outputs=[img_a_comp, img_b_comp, vA_out, vB_out, vD_out, dist_out]
    )

    btn_reset.click(fn=reset_test, outputs=[img_a_comp, img_b_comp, vA_out, vB_out, vD_out, dist_out])

if __name__ == "__main__":
    test_ui.launch(server_name="0.0.0.0", server_port=7861)
