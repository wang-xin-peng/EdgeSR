"""
EdgeSR 二值化 LCAP 门控剪枝版本。

从训练好的 EdgeSR 检查点加载，将 LCAP 门控按阈值转为二值掩码，
然后微调剩余权重。

二值掩码冻结——微调时仅更新卷积层。
"""

import torch
import torch.nn as nn
from .edgesr import EdgeSR
from .modules import LCAP


class BinaryLCAP(nn.Module):
    """二值化 LCAP，冻结掩码——低于阈值的通道被永久置零。"""

    def __init__(self, gate_values, threshold):
        super().__init__()
        mask = (gate_values >= threshold).float()
        self.register_buffer("mask", mask.view(1, -1, 1, 1))

    def forward(self, x):
        return x * self.mask


def prune_model(model, threshold=0.5):
    """
    将 LCAP 模块替换为二值门控的 BinaryLCAP。

    参数：
        model：训练好的 EdgeSR 实例
        threshold：门控剪枝阈值

    返回：
        替换了 BinaryLCAP 的模型（原地替换）
    """
    for name, module in model.named_modules():
        if isinstance(module, LCAP):
            g = torch.sigmoid(module.gate).detach()
            binary_lcap = BinaryLCAP(g, threshold)
            # Replace in the parent module
            parent = model
            parts = name.split(".")
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], binary_lcap)

    return model
