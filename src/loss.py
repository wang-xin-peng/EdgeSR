"""
Loss functions for EdgeSR training.
"""

import torch
import torch.nn.functional as F


class SSIMLoss(torch.nn.Module):
    """Differentiable SSIM loss (1 - SSIM)."""

    def __init__(self, window_size=11, size_average=True):
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average
        # Gaussian window
        gauss = torch.tensor(
            [0.0000, 0.0000, 0.0002, 0.0011, 0.0038, 0.0098, 0.0195, 0.0303,
             0.0369, 0.0352, 0.0262, 0.0152, 0.0069, 0.0024, 0.0007, 0.0001,
             0.0000],
            dtype=torch.float32,
        )
        # 1D → 2D separable window
        kernel_1d = gauss[:window_size] / gauss[:window_size].sum()
        kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
        self.register_buffer("kernel", kernel_2d[None, None, :, :])

    def forward(self, img1, img2):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        pad = self.window_size // 2
        kernel = self.kernel.to(img1.device).expand(3, 1, -1, -1)

        mu1 = F.conv2d(F.pad(img1, (pad, pad, pad, pad), mode="reflect"), kernel, groups=3)
        mu2 = F.conv2d(F.pad(img2, (pad, pad, pad, pad), mode="reflect"), kernel, groups=3)

        # Compute variance as conv((x - mu)^2) — always non-negative
        sigma1_sq = F.conv2d(F.pad((img1 - mu1) ** 2, (pad, pad, pad, pad), mode="reflect"), kernel, groups=3)
        sigma2_sq = F.conv2d(F.pad((img2 - mu2) ** 2, (pad, pad, pad, pad), mode="reflect"), kernel, groups=3)
        sigma12 = F.conv2d(F.pad((img1 - mu1) * (img2 - mu2), (pad, pad, pad, pad), mode="reflect"), kernel, groups=3)

        ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / (
            (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2)
        )

        return 1 - ssim_map.mean()
