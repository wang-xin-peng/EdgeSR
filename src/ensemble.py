"""
EdgeSR 自集成推理。

在推理时应用几何自集成（8 种变换），
无需重新训练即可提升 SR 质量。
"""

import torch


def _transform(x, t):
    if t == 0:
        return x
    if t == 1:
        return torch.flip(x, [-1])
    if t == 2:
        return torch.flip(x, [-2])
    if t == 3:
        return torch.rot90(x, 1, [-2, -1])
    if t == 4:
        return torch.rot90(x, -1, [-2, -1])
    if t == 5:
        return torch.rot90(x, 2, [-2, -1])
    if t == 6:
        return torch.flip(torch.rot90(x, 1, [-2, -1]), [-1])
    if t == 7:
        return torch.flip(torch.rot90(x, -1, [-2, -1]), [-1])
    raise ValueError(f"Unknown transform {t}")


def _inverse(x, t):
    if t == 0:
        return x
    if t == 1:
        return torch.flip(x, [-1])
    if t == 2:
        return torch.flip(x, [-2])
    if t == 3:
        return torch.rot90(x, -1, [-2, -1])
    if t == 4:
        return torch.rot90(x, 1, [-2, -1])
    if t == 5:
        return torch.rot90(x, 2, [-2, -1])
    if t == 6:
        return torch.rot90(torch.flip(x, [-1]), -1, [-2, -1])
    if t == 7:
        return torch.rot90(torch.flip(x, [-1]), 1, [-2, -1])
    raise ValueError(f"Unknown transform {t}")


@torch.no_grad()
def self_ensemble(model, lr_tensor, device):
    """
    应用几何自集成：8 种变换 → 模型推理 → 平均 → 反变换。

    参数：
        model：PyTorch SR 模型（评估模式）
        lr_tensor：LR 图像 [B, C, H, W]
        device：torch 设备

    返回：
        集成后的 SR 输出 [B, C, H, W]
    """
    results = []
    for t in range(8):
        x = _transform(lr_tensor, t).to(device)
        sr = model(x)
        results.append(_inverse(sr, t).cpu())
    return torch.stack(results).mean(dim=0)
