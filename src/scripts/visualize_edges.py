"""
可视化 EARB 模块的 Sobel 边缘图。

展示不同深度的 EARB 块提取的边缘特征。

用法：
    python src/scripts/visualize_edges.py --checkpoint checkpoints/edgesr_standard_best.pt \
        --baseline_checkpoint checkpoints/baseline_best.pt \
        --image ./data/benchmark/Set5/butterfly.png --output results/
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
import torchvision.transforms.functional as TF

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.edgesr import EdgeSR
from src.models.baseline import EDSRBaseline
from src.models.modules import SobelEdgeConv, LCAP


@torch.no_grad()
def visualize_edges(checkpoint_path, baseline_checkpoint, image_path, output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)

    # 加载模型
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = EdgeSR(n_resblocks=16, n_feats=64, n_earb=8, scale=2, lcap_threshold=0.01)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # 加载图像
    hr_img = Image.open(image_path).convert("RGB")
    hr_w, hr_h = hr_img.size
    hr_w, hr_h = hr_w - hr_w % 2, hr_h - hr_h % 2
    hr_img = hr_img.crop((0, 0, hr_w, hr_h))
    lr_img = hr_img.resize((hr_w // 2, hr_h // 2), Image.BICUBIC)
    lr_tensor = TF.to_tensor(lr_img).unsqueeze(0)

    # 前向传播，提取边缘图
    activations = {}

    def hook_fn(name):
        def hook(module, input, output):
            activations[name] = output
        return hook

    hooks = []
    for name, mod in model.named_modules():
        if isinstance(mod, LCAP):
            parts = name.split(".")
            idx = int(parts[-1])
            # LCAP at body.N gates EARB at body.(N-1); indices 17,19..31 → EARB 16,18..30
            if 17 <= idx <= 31 and idx % 2 == 1:
                hooks.append(mod.register_forward_hook(hook_fn(name)))

    _ = model(lr_tensor)

    for h in hooks:
        h.remove()

    # 保存 Sobel 边缘图
    sobel_keys = sorted(activations.keys())
    n = len(sobel_keys)
    if n == 0:
        print("No SobelEdgeConv modules found. Check model structure.")
        return
    n = len(sobel_keys)
    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(4 * ((n + 1) // 2), 8))
    axes = axes.flatten()

    for i, key in enumerate(sobel_keys):
        edge = activations[key]
        edge_mag = edge.pow(2).sum(dim=1, keepdim=True).sqrt()
        edge_img = edge_mag[0].mean(dim=0).cpu().numpy()
        axes[i].imshow(edge_img, cmap="inferno")
        lcap_idx = int(key.split(".")[1])
        earb_idx = lcap_idx - 1  # LCAP 在 body.N 处门控 EARB 在 body.(N-1)
        axes[i].set_title(f"EARB {earb_idx}")
        axes[i].axis("off")

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.suptitle("EARB Features (post-LCAP gating)", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_path = os.path.join(output_dir, "sobel_edges.png")
    fig.savefig(save_path, dpi=150)
    print(f"Edge maps saved to {save_path}")

    # 并排对比：LR vs EdgeSR vs Baseline vs HR
    sr_tensor = model(lr_tensor)

    # 加载 Baseline 模型
    base_ckpt = torch.load(baseline_checkpoint, map_location="cpu", weights_only=True)
    baseline = EDSRBaseline(n_resblocks=16, n_feats=64, scale=2)
    baseline.load_state_dict(base_ckpt["model"])
    baseline.eval()
    base_sr_tensor = baseline(lr_tensor)

    sr_img = TF.to_pil_image(sr_tensor[0].cpu().clamp(0, 1))
    base_img = TF.to_pil_image(base_sr_tensor[0].cpu().clamp(0, 1))

    # 差值图（放大 10 倍以便观察）
    diff = (sr_tensor[0].cpu() - base_sr_tensor[0].cpu()).abs().mean(dim=0)
    diff_amp = (diff * 10).clamp(0, 1).numpy()

    fig, axes = plt.subplots(1, 5, figsize=(24, 5))
    axes[0].imshow(lr_img)
    axes[0].set_title(f"LR Input ({lr_img.size[0]}x{lr_img.size[1]})")
    axes[0].axis("off")
    axes[1].imshow(sr_img)
    axes[1].set_title(f"EdgeSR SR")
    axes[1].axis("off")
    axes[2].imshow(base_img)
    axes[2].set_title(f"Baseline SR")
    axes[2].axis("off")
    axes[3].imshow(diff_amp, cmap="hot")
    axes[3].set_title("|EdgeSR − Baseline| ×10")
    axes[3].axis("off")
    axes[4].imshow(hr_img)
    axes[4].set_title(f"HR Ground Truth ({hr_img.size[0]}x{hr_img.size[1]})")
    axes[4].axis("off")
    fig.tight_layout()
    save_path = os.path.join(output_dir, "sr_comparison.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Comparison saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--baseline_checkpoint", type=str, default="../../checkpoints/baseline_best.pt")
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--output", type=str, default="results")
    args = parser.parse_args()
    visualize_edges(args.checkpoint, args.baseline_checkpoint, args.image, args.output)
