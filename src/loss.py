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

        # Mean
        mu1 = F.conv2d(F.pad(img1, (pad, pad, pad, pad), mode="reflect"), self.kernel)
        mu2 = F.conv2d(F.pad(img2, (pad, pad, pad, pad), mode="reflect"), self.kernel)

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        # Variance
        sigma1_sq = F.conv2d(F.pad(img1 ** 2, (pad, pad, pad, pad), mode="reflect"), self.kernel) - mu1_sq
        sigma2_sq = F.conv2d(F.pad(img2 ** 2, (pad, pad, pad, pad), mode="reflect"), self.kernel) - mu2_sq
        sigma12 = F.conv2d(F.pad(img1 * img2, (pad, pad, pad, pad), mode="reflect"), self.kernel) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
            (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        )

        if self.size_average:
            return 1 - ssim_map.mean()
        return 1 - ssim_map.mean(dim=(1, 2, 3))
