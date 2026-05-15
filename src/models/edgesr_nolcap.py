"""
EdgeSR ablation: EARB blocks without LCAP channel gating.

Same as EdgeSR but removes LCAP layers after each block.
Used to measure LCAP's contribution independently.
"""

import torch.nn as nn
from .baseline import ResBlock, UpsampleBlock
from .modules import EARB


class EdgeSRNoLCAP(nn.Module):
    """
    EdgeSR without Lightweight Channel Attention Pruning.

    Args:
        n_resblocks: total number of residual blocks (EARB + standard)
        n_feats: feature channels
        n_earb: number of EARB blocks (replaces the last n_earb ResBlocks)
        scale: upsampling factor
    """
    def __init__(self, n_resblocks=16, n_feats=64, n_earb=8, scale=2):
        super().__init__()
        assert n_earb <= n_resblocks, "n_earb must be <= n_resblocks"

        self.head = nn.Conv2d(3, n_feats, 3, padding=1, bias=True)

        body = []
        n_standard = n_resblocks - n_earb
        for i in range(n_resblocks):
            if i < n_standard:
                body.append(ResBlock(n_feats))
            else:
                body.append(EARB(n_feats))
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
