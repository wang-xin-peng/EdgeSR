"""
EdgeSR 自定义模块。

EARB（边缘感知残差块）：
    在标准残差块中加入固定 Sobel 边缘分支，融合边缘特征与纹理特征。

LCAP（轻量通道注意力剪枝）：
    可学习的通道门控，训练时软抑制，推理时可硬剪枝。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SobelEdgeConv(nn.Module):
    """
    固定 Sobel 边缘提取卷积（冻结，不训练）。

    对每个输入通道分别应用水平和垂直 Sobel 滤波器，
    拼接后的输出通道数为输入通道数的 2 倍（Gx 和 Gy）。
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
    边缘感知残差块。

    主支路：Conv-ReLU-Conv（标准残差路径）
    边缘支路：SobelEdgeConv -> ReLU -> Conv1x1（通道融合）
    输出：残差 + 边缘特征 + 跳跃连接

    参数：
        n_feats：输入/输出特征通道数
        res_scale：残差缩放因子
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
    轻量通道注意力剪枝。

    为每个通道学习一个门控值 g ∈ (0,1)。
    训练时：g 软门控通道（抑制）。
    推理时：门控值低于阈值的通道被剪枝。

    参数：
        n_feats：通道数
        threshold：剪枝阈值（仅推理时使用）
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
