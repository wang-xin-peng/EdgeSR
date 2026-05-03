"""
Simplified EDSR baseline model.

Architecture:
    Input -> Conv3x3 -> [ResBlock x 16] -> Conv3x3 -> PixelShuffle -> Conv3x3 -> Output

Reference: Enhanced Deep Residual Networks for Single Image Super-Resolution (EDSR, CVPR 2017)
Simplified: removed BatchNorm, used fixed residual scaling.
"""

import torch
import torch.nn as nn


class ResBlock(nn.Module):
    """Residual block with pre-activation Conv-ReLU-Conv."""
    def __init__(self, n_feats=64, res_scale=0.1):
        super().__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(n_feats, n_feats, 3, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(n_feats, n_feats, 3, padding=1, bias=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return residual + out * self.res_scale


class UpsampleBlock(nn.Module):
    """PixelShuffle upsampling block (2x)."""
    def __init__(self, n_feats=64):
        super().__init__()
        self.conv = nn.Conv2d(n_feats, n_feats * 4, 3, padding=1, bias=True)
        self.pixel_shuffle = nn.PixelShuffle(2)

    def forward(self, x):
        return self.pixel_shuffle(self.conv(x))


class EDSRBaseline(nn.Module):
    """Simplified EDSR baseline for super-resolution.

    Args:
        n_resblocks: number of residual blocks (default: 16)
        n_feats: number of feature channels (default: 64)
        scale: upsampling factor (2 or 4)
    """
    def __init__(self, n_resblocks=16, n_feats=64, scale=2):
        super().__init__()
        self.head = nn.Conv2d(3, n_feats, 3, padding=1, bias=True)
        body = [ResBlock(n_feats) for _ in range(n_resblocks)]
        body.append(nn.Conv2d(n_feats, n_feats, 3, padding=1, bias=True))
        self.body = nn.Sequential(*body)
        upsample_blocks = []
        for _ in range(scale // 2):
            upsample_blocks.append(UpsampleBlock(n_feats))
        self.upsampler = nn.Sequential(*upsample_blocks) if upsample_blocks else nn.Identity()
        self.tail = nn.Conv2d(n_feats, 3, 3, padding=1, bias=True)

    def forward(self, x):
        x = self.head(x)
        residual = self.body(x)
        x = x + residual
        x = self.upsampler(x)
        x = self.tail(x)
        return x
