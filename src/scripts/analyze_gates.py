"""
分析训练好的 EdgeSR 模型中 LCAP 门控分布。

用法：
    python src/scripts/analyze_gates.py --checkpoint checkpoints/edgesr_standard_best.pt

输出：
    - 打印每个 LCAP 层的门控统计
    - 保存门控分布图到 results/gate_distribution.png
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.edgesr import EdgeSR
from src.models.modules import LCAP


def analyze_gates(checkpoint_path, output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = EdgeSR(n_resblocks=16, n_feats=64, n_earb=8, scale=2, lcap_threshold=0.01)
    model.load_state_dict(ckpt["model"])

    layers_info = []
    all_gates = []

    for name, mod in model.named_modules():
        if isinstance(mod, LCAP):
            g = torch.sigmoid(mod.gate).detach().numpy()
            layers_info.append({
                "name": name,
                "mean": g.mean(),
                "min": g.min(),
                "max": g.max(),
                "gates": g,
            })
            all_gates.extend(g.tolist())

    all_gates = np.array(all_gates)
    print(f"Total LCAP gates: {len(all_gates)}")
    print(f"  Mean: {all_gates.mean():.4f}")
    print(f"  Std:  {all_gates.std():.4f}")
    print(f"  Min:  {all_gates.min():.4f}")
    print(f"  Max:  {all_gates.max():.4f}")

    # 打印每层在不同阈值下的剪枝率
    print(f"\n{'Layer':<15} {'Mean':>6} {'Min':>6} {'Max':>6} {'<0.3':>6} {'<0.5':>6}")
    print("-" * 55)
    for info in layers_info:
        lt03 = int((info["gates"] < 0.3).sum())
        lt05 = int((info["gates"] < 0.5).sum())
        print(f"{info['name']:<15} {info['mean']:>6.3f} {info['min']:>6.3f} {info['max']:>6.3f} {lt03:>6d} {lt05:>6d}")

    for thresh in [0.1, 0.2, 0.3, 0.4, 0.5]:
        pruned = (all_gates < thresh).sum()
        print(f"  threshold={thresh:.1f}: {pruned}/{len(all_gates)} ({pruned/len(all_gates)*100:.0f}% pruned)")

    # 绘图
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # 所有门控的直方图
    axes[0].hist(all_gates, bins=50, color="steelblue", edgecolor="white")
    axes[0].axvline(0.3, color="red", linestyle="--", alpha=0.7, label="threshold 0.3")
    axes[0].axvline(0.5, color="orange", linestyle="--", alpha=0.7, label="threshold 0.5")
    axes[0].set_xlabel("Gate value (sigmoid)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("LCAP Gate Distribution (All Layers)")
    axes[0].legend()

    # 每层平均门控值
    names = [info["name"] for info in layers_info]
    means = [info["mean"] for info in layers_info]
    colors = ["coral" if "sobel" in n or int(n.split(".")[1]) >= (16 - 8) * 2 else "steelblue" for n in names]
    axes[1].bar(range(len(names)), means, color=colors)
    axes[1].axhline(0.5, color="orange", linestyle="--", alpha=0.7, label="threshold 0.5")
    axes[1].set_xticks(range(len(names)))
    axes[1].set_xticklabels(names, rotation=45, fontsize=8)
    axes[1].set_ylabel("Mean gate value")
    axes[1].set_title("Per-Layer Mean Gate (blue=ResBlock, red=EARB)")
    axes[1].legend()
    plt.tight_layout()

    save_path = os.path.join(output_dir, "gate_distribution.png")
    fig.savefig(save_path, dpi=150)
    print(f"\nPlot saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default="results")
    args = parser.parse_args()
    analyze_gates(args.checkpoint, args.output)
