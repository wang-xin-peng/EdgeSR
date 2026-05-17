"""
EdgeSR 训练脚本。

用法：
    python -m src.train --config configs/edgesr_standard.yaml --model baseline
    python -m src.train --config configs/edgesr_standard.yaml --model edgesr
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
from src.models import EDSRBaseline, EdgeSR, EdgeSRNoLCAP
from src.models.edgesr_pruned import prune_model
from src.loss import SSIMLoss


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
    elif model_name == "edgesr_nolcap":
        return EdgeSRNoLCAP(
            n_resblocks=config["model"]["n_resblocks"],
            n_feats=config["model"]["n_feats"],
            n_earb=config["model"]["n_earb"],
            scale=config["data"]["scale"],
        )
    elif model_name == "edgesr_pruned":
        pretrained_path = config["model"].get("pretrained")
        prune_threshold = config["model"].get("prune_threshold", 0.5)
        model = EdgeSR(
            n_resblocks=config["model"]["n_resblocks"],
            n_feats=config["model"]["n_feats"],
            n_earb=config["model"]["n_earb"],
            scale=config["data"]["scale"],
            lcap_threshold=0.01,
        )
        if pretrained_path:
            ckpt = torch.load(pretrained_path, map_location="cpu")
            model.load_state_dict(ckpt["model"])
            print(f"Loaded pretrained EdgeSR from {pretrained_path}")
        model = prune_model(model, threshold=prune_threshold)
        active = sum(m.mask.sum().item() for m in model.modules() if hasattr(m, 'mask'))
        total = sum(64 for m in model.modules() if hasattr(m, 'mask'))
        print(f"Pruned: {int(total-active)}/{int(total)} channels removed at threshold={prune_threshold}")
        return model
    else:
        raise ValueError(f"Unknown model: {model_name}")


def compute_psnr(img1, img2):
    """计算两个图像张量之间的 PSNR（batch, C, H, W），值域 [0,1]。"""
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
    parser.add_argument("--config", type=str, default="configs/edgesr_standard.yaml")
    parser.add_argument("--model", type=str, default="edgesr", choices=["baseline", "edgesr", "edgesr_nolcap", "edgesr_pruned"])
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    config["model"]["name"] = args.model

    # 设备选择
    if args.device:
        device_str = args.device
    else:
        device_str = config.get("device", "cuda")
    device = torch.device(device_str if (device_str == "cpu" or torch.cuda.is_available()) else "cpu")
    use_amp = device.type == "cuda"
    print(f"Using device: {device}, AMP: {use_amp}")

    # 实验名称（用于生成检查点文件名）
    exp_name = config.get("experiment", config["model"]["name"])

    # 模型
    model = get_model(config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.model}, Parameters: {total_params:,}")

    # 数据
    train_loader = create_train_dataloader(config)
    print(f"Training samples: {len(train_loader.dataset)}")

    # 优化器与调度器
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
    loss_type = config["train"].get("loss", "L1")
    if loss_type == "L1+SSIM":
        ssim_fn = SSIMLoss()
        loss_fn = lambda sr, hr: nn.L1Loss()(sr, hr) + 0.1 * ssim_fn(sr, hr)
    else:
        loss_fn = nn.L1Loss()

    # 恢复训练
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resumed from epoch {start_epoch}")

    # 训练循环
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
                }, os.path.join(config["log"]["save_dir"], f"{exp_name}_best.pt"))
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
            }, os.path.join(config["log"]["save_dir"], f"{exp_name}_checkpoint_{epoch+1}.pt"))

        scheduler.step()
