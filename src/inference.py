"""
Inference and visualization script.

Generates SR results for images and creates comparison grids.

Usage:
    python src/inference.py --config configs/edgesr_standard.yaml --model edgesr \
        --checkpoint checkpoints/edgesr_standard_best.pt --input ./data/benchmark/Set5 --output ./results
"""

import os
import argparse
import yaml
import torch
import torchvision.utils as vutils
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from tqdm import tqdm

from src.models import EDSRBaseline, EdgeSR, EdgeSRNoLCAP
from src.models.edgesr_pruned import prune_model


def get_model(config, checkpoint_path, device):
    model_name = config["model"]["name"]
    if model_name == "baseline":
        model = EDSRBaseline(
            n_resblocks=config["model"]["n_resblocks"],
            n_feats=config["model"]["n_feats"],
            scale=config["data"]["scale"],
        )
    elif model_name == "edgesr":
        model = EdgeSR(
            n_resblocks=config["model"]["n_resblocks"],
            n_feats=config["model"]["n_feats"],
            n_earb=config["model"]["n_earb"],
            scale=config["data"]["scale"],
            lcap_threshold=config["model"]["lcap_threshold"],
        )
    elif model_name == "edgesr_nolcap":
        model = EdgeSRNoLCAP(
            n_resblocks=config["model"]["n_resblocks"],
            n_feats=config["model"]["n_feats"],
            n_earb=config["model"]["n_earb"],
            scale=config["data"]["scale"],
        )
    elif model_name == "edgesr_pruned":
        base = EdgeSR(
            n_resblocks=config["model"]["n_resblocks"],
            n_feats=config["model"]["n_feats"],
            n_earb=config["model"]["n_earb"],
            scale=config["data"]["scale"],
            lcap_threshold=0.01,
        )
        model = prune_model(base, threshold=config["model"].get("prune_threshold", 0.5))
    else:
        raise ValueError(f"Unknown model: {model_name}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def super_resolve(model, lr_tensor, device):
    """Super-resolve a single LR tensor [1, 3, H, W]."""
    lr_tensor = lr_tensor.to(device)
    sr_tensor = model(lr_tensor)
    return sr_tensor.clamp(0, 1).cpu()


def load_image(path):
    """Load image as tensor [1, 3, H, W] in [0, 1]."""
    img = Image.open(path).convert("RGB")
    return torch.from_numpy(np.array(img)).float().permute(2, 0, 1).unsqueeze(0) / 255.0


def save_comparison_grid(lr_path, sr_path, hr_path, save_path, scale=2):
    """Save a 2x2 comparison grid: LR (bicubic up), SR (model), LR original, HR."""
    lr_img = Image.open(lr_path).convert("RGB")
    sr_img = Image.open(sr_path).convert("RGB")
    hr_img = Image.open(hr_path).convert("RGB")
    lr_up = lr_img.resize(sr_img.size, Image.BICUBIC)

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes[0, 0].imshow(lr_up)
    axes[0, 0].set_title("LR (Bicubic up)", fontsize=10)
    axes[0, 0].axis("off")
    axes[0, 1].imshow(sr_img)
    axes[0, 1].set_title("SR (Model)", fontsize=10)
    axes[0, 1].axis("off")
    axes[1, 0].imshow(lr_img.resize(sr_img.size, Image.NEAREST))
    axes[1, 0].set_title(f"LR ({lr_img.size[0]}x{lr_img.size[1]})", fontsize=10)
    axes[1, 0].axis("off")
    axes[1, 1].imshow(hr_img)
    axes[1, 1].set_title("HR (Ground Truth)", fontsize=10)
    axes[1, 1].axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def save_side_by_side(lr_path, sr_path, hr_path, save_path):
    """Save a side-by-side comparison image (LR | SR | HR)."""
    lr_img = Image.open(lr_path).convert("RGB")
    sr_img = Image.open(sr_path).convert("RGB")
    hr_img = Image.open(hr_path).convert("RGB")
    lr_resized = lr_img.resize(sr_img.size, Image.BICUBIC)
    hr_resized = hr_img.resize(sr_img.size, Image.BICUBIC)
    total_w = lr_resized.width * 3
    total_h = lr_resized.height
    canvas = Image.new("RGB", (total_w, total_h))
    canvas.paste(lr_resized, (0, 0))
    canvas.paste(sr_img, (lr_resized.width, 0))
    canvas.paste(hr_resized, (lr_resized.width * 2, 0))
    canvas.save(save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/edgesr_standard.yaml")
    parser.add_argument("--model", type=str, default="edgesr", choices=["baseline", "edgesr", "edgesr_nolcap", "edgesr_pruned"])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--input", type=str, required=True, help="Input image or directory")
    parser.add_argument("--output", type=str, default="./results", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--grid", action="store_true", help="Save comparison grid instead of side-by-side")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    config["model"]["name"] = args.model
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model = get_model(config, args.checkpoint, device)
    print(f"Loaded {args.model} model from {args.checkpoint}")

    if os.path.isfile(args.input):
        input_paths = [args.input]
    else:
        input_paths = sorted([
            os.path.join(args.input, f)
            for f in os.listdir(args.input)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
        ])

    os.makedirs(args.output, exist_ok=True)

    for img_path in tqdm(input_paths, desc="Processing"):
        name = os.path.splitext(os.path.basename(img_path))[0]
        lr_tensor = load_image(img_path)
        _, _, h, w = lr_tensor.shape
        scale = config["data"]["scale"]
        lr_tensor = lr_tensor[:, :, :h - h % scale, :w - w % scale]
        sr_tensor = super_resolve(model, lr_tensor, device)
        sr_save_path = os.path.join(args.output, f"{name}_sr.png")
        vutils.save_image(sr_tensor, sr_save_path)
        lr_np = lr_tensor.squeeze(0).permute(1, 2, 0).numpy()
        lr_pil = Image.fromarray((lr_np * 255).astype(np.uint8))
        lr_save_path = os.path.join(args.output, f"{name}_lr.png")
        lr_pil.save(lr_save_path)

        if args.grid:
            cmp_path = os.path.join(args.output, f"{name}_comparison.png")
            save_comparison_grid(lr_save_path, sr_save_path, img_path, cmp_path, scale)
        else:
            cmp_path = os.path.join(args.output, f"{name}_sidebyside.png")
            save_side_by_side(lr_save_path, sr_save_path, img_path, cmp_path)

    print(f"Results saved to {args.output}")
