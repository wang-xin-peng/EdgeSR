"""
EdgeSR 消融实验：只有 EARB，无 LCAP 通道门控。

与 EdgeSR 相同，但去掉了每个块后面的 LCAP 层。
用于单独测量 LCAP 的贡献。
"""

import torch.nn as nn
from .baseline import ResBlock, UpsampleBlock
from .modules import EARB


class EdgeSRNoLCAP(nn.Module):
    """
    EdgeSR 去除 LCAP 通道剪枝的版本。

    参数：
        n_resblocks：残差块总数（EARB + 标准）
        n_feats：特征通道数
        n_earb：EARB 块数量
        scale：上采样因子
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
