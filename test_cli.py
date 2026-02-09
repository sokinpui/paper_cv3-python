import sys

import numpy as np
import torch
import torch.nn.functional as F

from globals import DEVICE
from metrics import HumanEyeColorMetric


class OklabVectorEngine:
    def __init__(self, size=512):
        self.size = size
        self.metric = HumanEyeColorMetric()
        self.pooling_n = 1
        self.explosion_k = 0.0
        self.use_dim_scale = False
        self.reset()

    def reset(self):
        self.canvas_a = np.full((self.size, self.size, 3), 255, dtype=np.uint8)
        self.canvas_b = np.full((self.size, self.size, 3), 255, dtype=np.uint8)

    def draw(self, target, x, y):
        canvas = self.canvas_a if target.lower() == "a" else self.canvas_b

        # Bounds check
        if not (0 <= x < self.size and 0 <= y < self.size):
            print(f"Error: Coordinates ({x}, {y}) out of range [0-{self.size-1}]")
            return

        # Draw 3x3 dot
        canvas[y, x] = 0
        print(f"Drew dot on {target.upper()} at ({x}, {y})")

    def _process(self, canvas_np):
        img_t = (
            torch.from_numpy(canvas_np).permute(2, 0, 1).float().unsqueeze(0).to(DEVICE)
            / 255.0
        )

        oklab = self.metric._rgb_to_oklab(img_t)

        if self.pooling_n > 1:
            oklab = F.avg_pool2d(
                oklab, kernel_size=self.pooling_n, stride=self.pooling_n
            )

        return oklab.reshape(-1)

    def get_vector_summary(self, target):
        canvas = self.canvas_a if target.lower() == "a" else self.canvas_b
        vec = self._process(canvas)
        vec_np = vec.detach().cpu().numpy()

        summary = [
            f"--- Vector {target.upper()} ---",
            f"Shape: {vec_np.shape}",
            f"Mean:  {vec_np.mean():.6f}",
            f"Std:   {vec_np.std():.6f}",
            f"Min:   {vec_np.min():.6f}",
            f"Max:   {vec_np.max():.6f}",
            f"Snippet: {vec_np[:5]} ... {vec_np[-5:]}",
        ]
        return "\n".join(summary), vec

    def get_final_distance(self):
        _, vA = self.get_vector_summary("a")
        _, vB = self.get_vector_summary("b")

        diff = torch.abs(vA - vB)

        # Coefficient based on total pixels (Area: Width x Height)
        area_coefficient = self.size * self.size
        scale = area_coefficient if self.use_dim_scale else 1.0

        if self.explosion_k > 0:
            dist = torch.sum(torch.exp(self.explosion_k * diff) - 1)
            return (dist * scale).item()

        # Euclidean Distance scaled by resolution to maintain resolution-independent magnitude
        sum_pow = torch.sum(torch.pow(diff, 2.0))
        dist = torch.pow(sum_pow, 0.5) * scale

        return dist.item()

    def analyze(self):
        _, vA = self.get_vector_summary("a")
        _, vB = self.get_vector_summary("b")

        dist = self.get_final_distance()
        vD = vA - vB
        vD_np = vD.detach().cpu().numpy()

        res = [
            "--- Distance Analysis ---",
            f"Scale Coefficient: {self.size * self.size if self.use_dim_scale else 1.0}",
            f"Final Distance: {dist:.6f}",
            f"L2 Norm of Diff (vD): {torch.norm(vD).item():.6f}",
            f"vD Snippet: {vD_np[:5]} ... {vD_np[-5:]}",
        ]
        return "\n".join(res)


def print_help():
    print("\nAvailable Commands:")
    print("  p a / print a      : Print vector summary of Canvas A")
    print("  p b / print b      : Print vector summary of Canvas B")
    print("  draw a <x> <y>     : Draw a dot on Canvas A (0-511)")
    print("  draw b <x> <y>     : Draw a dot on Canvas B (0-511)")
    print("  run                : Run distance analysis and show vD")
    print("  d / distance       : Print the final distance value")
    print("  set k <val>        : Set Pixel Explosion k (default 0.0)")
    print("  set scale <on|off> : Apply image dimension coefficient (512x512)")
    print("  reset              : Clear both canvases")
    print("  help               : Show this help")
    print("  exit / quit        : Close the tester")


def main():
    engine = OklabVectorEngine(size=512)
    print("Oklab CLI Tester - Canvas Size 512x512")
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

            if cmd == "reset":
                engine.reset()
                print("Canvases reset.")
                continue

            if cmd == "run" or (cmd == "p" and len(parts) > 1 and parts[1] == "run"):
                print(engine.analyze())
                continue

            if cmd in ["d", "distance"]:
                print(f"Final Distance: {engine.get_final_distance():.6f}")
                continue

            if cmd == "set" and len(parts) > 2:
                try:
                    val = float(parts[2])
                    if parts[1] == "k":
                        engine.explosion_k = val
                        print(f"explosion_k set to {val}")
                    continue
                except ValueError:
                    pass

                if parts[1] == "scale" and len(parts) > 2:
                    engine.use_dim_scale = parts[2] == "on"
                    print(
                        f"Dimension scaling {'Enabled' if engine.use_dim_scale else 'Disabled'}"
                    )
                    continue

            if cmd in ["p", "print"]:
                if len(parts) < 2:
                    print("Usage: print <a|b>")
                    continue
                summary, _ = engine.get_vector_summary(parts[1])
                print(summary)
                continue

            if cmd == "draw":
                if len(parts) < 4:
                    print("Usage: draw <a|b> <x> <y>")
                    continue
                target = parts[1]
                try:
                    x, y = int(parts[2]), int(parts[3])
                    engine.draw(target, x, y)
                except ValueError:
                    print("Error: x and y must be integers.")
                continue

            print(f"Unknown command: {cmd}. Type 'help' for info.")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
