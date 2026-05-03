"""
Training script for EdgeSR.

Usage:
    python src/train.py --config configs/default.yaml --model baseline
    python src/train.py --config configs/default.yaml --model edgesr
"""

import os
import argparse
import yaml
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from src.data.dataset import create_train_dataloader
from src.models import EDSRBaseline, EdgeSR


def get_model(config):
    model_name = config["model"]["name"]
    if model_name == "baseline":
        return EDSRBaseline(
            n_resblocks=config["model"]["n_resblocks"],
            n_feats=config["model"]["n_feats"],
            scale=config["data"]["scale"],
        )
    elif model_name == "edgesr":
        return EdgeSR(
            n_resblocks=config["model"]["n_resblocks"],
            n_feats=config["model"]["n_feats"],
            n_earb=config["model"]["n_earb"],
            scale=config["data"]["scale"],
            lcap_threshold=config["model"]["lcap_threshold"],
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")


def compute_psnr(img1, img2):
    """Compute PSNR between two image tensors (batch, C, H, W) in [0,1]."""
    mse = torch.mean((img1 - img2) ** 2, dim=(1, 2, 3))
    return 10 * torch.log10(1.0 / (mse + 1e-8))


def train_epoch(model, dataloader, optimizer, scaler, loss_fn, config, device, use_amp=False):
    model.train()
    total_loss = 0
    for lr_imgs, hr_imgs in tqdm(dataloader, desc="Training", leave=False):
        lr_imgs = lr_imgs.to(device)
        hr_imgs = hr_imgs.to(device)

        optimizer.zero_grad()
        if use_amp:
            with autocast():
                sr_imgs = model(lr_imgs)
                loss = loss_fn(sr_imgs, hr_imgs)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["train"]["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
        else:
            sr_imgs = model(lr_imgs)
            loss = loss_fn(sr_imgs, hr_imgs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["train"]["grad_clip"])
            optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


@torch.no_grad()
def validate(model, dataloader, device):
    model.eval()
    total_psnr = 0
    for lr_imgs, hr_imgs in dataloader:
        lr_imgs = lr_imgs.to(device)
        hr_imgs = hr_imgs.to(device)
        sr_imgs = model(lr_imgs)
        scale = hr_imgs.shape[-1] // lr_imgs.shape[-1]
        border = scale
        sr_cropped = sr_imgs[..., border:-border, border:-border]
        hr_cropped = hr_imgs[..., border:-border, border:-border]
        psnr = compute_psnr(sr_cropped, hr_cropped)
        total_psnr += psnr.mean().item()
    return total_psnr / len(dataloader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--model", type=str, default="edgesr", choices=["baseline", "edgesr"])
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    config["model"]["name"] = args.model

    # Device selection
    if args.device:
        device_str = args.device
    else:
        device_str = config.get("device", "cuda")
    device = torch.device(device_str if (device_str == "cpu" or torch.cuda.is_available()) else "cpu")
    use_amp = device.type == "cuda"
    print(f"Using device: {device}, AMP: {use_amp}")

    # Model
    model = get_model(config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.model}, Parameters: {total_params:,}")

    # Data
    train_loader = create_train_dataloader(config)
    print(f"Training samples: {len(train_loader.dataset)}")

    # Optimizer & scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=config["train"]["lr"],
        weight_decay=config["train"]["weight_decay"],
    )
    scheduler = StepLR(
        optimizer,
        step_size=config["train"]["lr_decay_step"],
        gamma=config["train"]["lr_decay_factor"],
    )
    scaler = GradScaler(enabled=use_amp)
    loss_fn = nn.L1Loss()

    # Resume
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resumed from epoch {start_epoch}")

    # Training loop
    os.makedirs(config["log"]["save_dir"], exist_ok=True)
    best_psnr = 0

    for epoch in range(start_epoch, config["train"]["epochs"]):
        train_loss = train_epoch(model, train_loader, optimizer, scaler, loss_fn, config, device, use_amp)

        if (epoch + 1) % config["log"]["eval_every"] == 0:
            val_psnr = validate(model, train_loader, device)
            print(f"Epoch [{epoch+1}/{config['train']['epochs']}] "
                  f"Loss: {train_loss:.6f} | Val PSNR: {val_psnr:.4f}")
            if val_psnr > best_psnr:
                best_psnr = val_psnr
                torch.save({
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "psnr": val_psnr,
                }, os.path.join(config["log"]["save_dir"], "best.pt"))
                print(f"  New best model saved (PSNR: {val_psnr:.4f})")
        else:
            print(f"Epoch [{epoch+1}/{config['train']['epochs']}] Loss: {train_loss:.6f}")

        if (epoch + 1) % config["log"]["save_every"] == 0:
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "psnr": val_psnr if (epoch + 1) % config["log"]["eval_every"] == 0 else 0,
            }, os.path.join(config["log"]["save_dir"], f"checkpoint_{epoch+1}.pt"))

        scheduler.step()
