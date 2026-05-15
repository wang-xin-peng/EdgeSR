"""
EdgeSR with binary LCAP gating for channel pruning.

Loads from a trained EdgeSR checkpoint, converts LCAP gates to binary
masks at a given threshold, then finetunes the remaining weights.

The binary masks are frozen — only conv layers are updated during finetune.
"""

import torch
import torch.nn as nn
from .edgesr import EdgeSR
from .modules import LCAP


class BinaryLCAP(nn.Module):
    """LCAP with frozen binary mask — channels below threshold are permanently zeroed."""

    def __init__(self, gate_values, threshold):
        super().__init__()
        mask = (gate_values >= threshold).float()
        self.register_buffer("mask", mask.view(1, -1, 1, 1))

    def forward(self, x):
        return x * self.mask


def prune_model(model, threshold=0.5):
    """
    Replace LCAP modules with BinaryLCAP using gate masks at the given threshold.

    Args:
        model: trained EdgeSR instance
        threshold: pruning threshold for gates

    Returns:
        model with BinaryLCAP modules (in-place replacement)
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
