"""
Evaluation script for EdgeSR.

Computes PSNR, SSIM, and LPIPS on Set5/Set14/BSD100.
Usage:
    python src/test.py --config configs/default.yaml --model baseline --checkpoint checkpoints/best.pt
"""

import os
import argparse
import yaml
import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim_func
from tqdm import tqdm

from src.data.dataset import create_test_dataloader
from src.models import EDSRBaseline, EdgeSR, EdgeSRNoLCAP
from src.models.edgesr_pruned import prune_model
from src.ensemble import self_ensemble


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


def compute_psnr_np(img1, img2):
    """PSNR in dB for numpy arrays in [0, 1]."""
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
    return 10 * np.log10(1.0 / mse)


def compute_ssim_np(img1, img2):
    """SSIM for numpy arrays in [0, 1], multichannel."""
    return ssim_func(img1, img2, channel_axis=-1, data_range=1.0)


def evaluate(model, dataloader, device, lpips_model=None, use_ensemble=False):
    """Evaluate model on a dataset, return dict of metrics."""
    psnr_list = []
    ssim_list = []
    lpips_list = []

    for lr_imgs, hr_imgs in tqdm(dataloader, desc="Evaluating"):
        lr_imgs = lr_imgs.to(device)
        hr_imgs = hr_imgs.to(device)
        with torch.no_grad():
            if use_ensemble:
                sr_imgs = self_ensemble(model, lr_imgs.cpu(), device).to(device)
            else:
                sr_imgs = model(lr_imgs)
        scale = hr_imgs.shape[-1] // lr_imgs.shape[-1]
        border = scale
        sr_cropped = sr_imgs[..., border:-border, border:-border]
        hr_cropped = hr_imgs[..., border:-border, border:-border]
        sr_np = sr_cropped.cpu().clamp(0, 1).numpy().transpose(0, 2, 3, 1)
        hr_np = hr_cropped.cpu().numpy().transpose(0, 2, 3, 1)
        for i in range(sr_np.shape[0]):
            psnr_list.append(compute_psnr_np(sr_np[i], hr_np[i]))
            ssim_list.append(compute_ssim_np(sr_np[i], hr_np[i]))
        if lpips_model is not None:
            sr_lpips = sr_cropped * 2 - 1
            hr_lpips = hr_cropped * 2 - 1
            lpips_val = lpips_model(sr_lpips, hr_lpips)
            lpips_list.extend(lpips_val.cpu().numpy().flatten().tolist())

    results = {
        "PSNR": float(np.mean(psnr_list)),
        "SSIM": float(np.mean(ssim_list)),
    }
    if lpips_list:
        results["LPIPS"] = float(np.mean(lpips_list))
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/edgesr_standard.yaml")
    parser.add_argument("--model", type=str, default="edgesr", choices=["baseline", "edgesr", "edgesr_nolcap", "edgesr_pruned"])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--lpips", action="store_true", help="Compute LPIPS (requires lpips package)")
    parser.add_argument("--self-ensemble", action="store_true", help="Use geometric self-ensemble (8x inference)")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    config["model"]["name"] = args.model
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model = get_model(config, args.checkpoint, device)
    print(f"Loaded {args.model} model from {args.checkpoint}")

    lpips_model = None
    if args.lpips:
        import lpips

        lpips_model = lpips.LPIPS(net="alex").to(device)

    for dataset_name in ["Set5", "Set14", "BSD100"]:
        dataloader, _ = create_test_dataloader(config, dataset_name)
        results = evaluate(model, dataloader, device, lpips_model, use_ensemble=args.self_ensemble)
        print(f"\nDataset: {dataset_name}")
        for metric, value in results.items():
            print(f"  {metric}: {value:.6f}")
        print()
