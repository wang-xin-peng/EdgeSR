"""
EdgeSR custom modules.

EARB (Edge-Aware Residual Block):
    Adds a frozen Sobel edge branch to a standard residual block,
    fusing edge features with texture features.

L-CAP (Lightweight Channel Attention Pruning):
    Learnable channel gating for soft suppression during training
    and hard pruning during inference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SobelEdgeConv(nn.Module):
    """
    Fixed Sobel edge extraction conv (frozen, not trained).

    Applies horizontal and vertical Sobel filters to each input channel
    and concatenates the resulting gradient maps.
    Output has 2x input channels (Gx and Gy per input channel).
    """
    def __init__(self, in_channels):
        super().__init__()
        kernel = torch.zeros(2 * in_channels, in_channels, 3, 3)
        gx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        gy = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        for i in range(in_channels):
            kernel[i, i, :, :] = gx
            kernel[in_channels + i, i, :, :] = gy
        self.register_buffer("weight", kernel)
        self.bias = None

    def forward(self, x):
        return F.conv2d(x, self.weight, bias=self.bias, padding=1)


class EARB(nn.Module):
    """
    Edge-Aware Residual Block.

    Main branch: Conv-ReLU-Conv (standard residual path)
    Edge branch: SobelEdgeConv -> ReLU -> Conv1x1 (adjust channels)
    Output: residual + edge_features + skip_connection

    Args:
        n_feats: number of input/output feature channels
        res_scale: residual scaling factor
    """
    def __init__(self, n_feats=64, res_scale=0.1):
        super().__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(n_feats, n_feats, 3, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(n_feats, n_feats, 3, padding=1, bias=True)
        self.sobel = SobelEdgeConv(n_feats)
        self.edge_conv = nn.Conv2d(2 * n_feats, n_feats, 1, padding=0, bias=True)
        self.edge_relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        edge = self.sobel(x)
        edge = self.edge_relu(edge)
        edge = self.edge_conv(edge)
        out = out + edge
        return residual + out * self.res_scale


class LCAP(nn.Module):
    """
    Lightweight Channel Attention Pruning.

    Learns a gating vector g in (0,1) per channel.
    During training: g softly gates channels (suppression).
    During inference: channels with g < threshold are pruned.

    Args:
        n_feats: number of channels
        threshold: pruning threshold (inference only)
    """
    def __init__(self, n_feats=64, threshold=0.01):
        super().__init__()
        self.threshold = threshold
        self.gate = nn.Parameter(torch.zeros(n_feats))

    def forward(self, x):
        g = torch.sigmoid(self.gate)
        g = g.view(1, -1, 1, 1)
        return x * g

    def get_pruned_channels(self):
        g = torch.sigmoid(self.gate)
        return torch.where(g >= self.threshold)[0]
