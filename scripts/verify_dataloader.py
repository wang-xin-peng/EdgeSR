"""
Verify DataLoader correctness.

Loads a batch and checks:
    - LR and HR tensor shapes match expected dimensions
    - Value ranges are in [0, 1]
    - Spatial relationship (LR * scale = HR)
    - Visual inspection: save a sample pair

Usage:
    python scripts/verify_dataloader.py --config configs/default.yaml
"""

import os
import sys
import yaml
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.dataset import create_train_dataloader


def verify():
    with open("configs/default.yaml") as f:
        config = yaml.safe_load(f)

    dataloader = create_train_dataloader(config)
    lr_imgs, hr_imgs = next(iter(dataloader))

    print(f"Batch size: {lr_imgs.shape[0]}")
    print(f"LR shape: {lr_imgs.shape} (expected: [B, 3, H, W])")
    print(f"HR shape: {hr_imgs.shape} (expected: [B, 3, H*scale, W*scale])")
    print(f"LR range: [{lr_imgs.min().item():.4f}, {lr_imgs.max().item():.4f}] (expected: [0, 1])")
    print(f"HR range: [{hr_imgs.min().item():.4f}, {hr_imgs.max().item():.4f}] (expected: [0, 1])")

    scale = config["data"]["scale"]
    expected_hr_size = lr_imgs.shape[-1] * scale
    actual_hr_size = hr_imgs.shape[-1]
    print(f"HR spatial size: {actual_hr_size} (expected LR*scale = {expected_hr_size})")
    assert actual_hr_size == expected_hr_size, "HR size mismatch!"
    assert lr_imgs.min() >= 0 and lr_imgs.max() <= 1, "LR out of [0,1] range"
    assert hr_imgs.min() >= 0 and hr_imgs.max() <= 1, "HR out of [0,1] range"

    lr_np = lr_imgs[0].permute(1, 2, 0).numpy()
    hr_np = hr_imgs[0].permute(1, 2, 0).numpy()

    os.makedirs("verify_output", exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    ax1.imshow(lr_np)
    ax1.set_title(f"LR ({lr_np.shape[1]}x{lr_np.shape[0]})")
    ax1.axis("off")
    ax2.imshow(hr_np)
    ax2.set_title(f"HR ({hr_np.shape[1]}x{hr_np.shape[0]})")
    ax2.axis("off")
    plt.tight_layout()
    plt.savefig("verify_output/dataloader_check.png", dpi=150)
    plt.close()
    print(f"Visual check saved to verify_output/dataloader_check.png")
    print("All checks passed!")


if __name__ == "__main__":
    verify()
