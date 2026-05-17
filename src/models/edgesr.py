"""
EdgeSR：改进的边缘感知超分辨率模型。

架构：EDSR 基线骨架 + EARB 模块 + LCAP 通道门控。
"""

import torch.nn as nn
from .baseline import ResBlock, UpsampleBlock
from .modules import EARB, LCAP


class EdgeSR(nn.Module):
    """
    EdgeSR 模型，包含边缘感知残差块和通道剪枝。

    参数：
        n_resblocks：残差块总数（EARB + 标准）
        n_feats：特征通道数
        n_earb：EARB 块数量（替换最后 n_earb 个 ResBlock）
        scale：上采样因子
        lcap_threshold：LCAP 推理时剪枝阈值
    """
    def __init__(self, n_resblocks=16, n_feats=64, n_earb=8, scale=2, lcap_threshold=0.01):
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
            body.append(LCAP(n_feats, lcap_threshold))
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
