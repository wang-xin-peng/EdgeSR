"""
Self-ensemble inference for EdgeSR.

Applies geometric self-ensemble (8 transformations) during inference
to improve SR quality without retraining.
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
    Apply geometric self-ensemble: 8 transforms → model → average → inverse.

    Args:
        model: torch SR model (eval mode)
        lr_tensor: LR image [B, C, H, W]
        device: torch device

    Returns:
        Ensemble SR output [B, C, H, W]
    """
    results = []
    for t in range(8):
        x = _transform(lr_tensor, t).to(device)
        sr = model(x)
        results.append(_inverse(sr, t).cpu())
    return torch.stack(results).mean(dim=0)
