"""
Real-world degradation functions for blind SR training.

Pipeline:
  HR → Strong Blur (optical defocus) → Bicubic down → Noise + JPEG → LR

Strong blur is applied on HR before downsampling to simulate camera defocus.
Noise and JPEG are applied on LR after downsampling to simulate sensor noise
and compression artifacts — these won't be washed out by downscaling.
"""

import random
import io
import numpy as np
from PIL import Image, ImageFilter


def random_gaussian_blur(img, sigma_range=(3.0, 10.0)):
    """Apply strong Gaussian blur (simulates camera defocus)."""
    sigma = random.uniform(*sigma_range)
    return img.filter(ImageFilter.GaussianBlur(radius=sigma))


def random_noise(img, noise_range=(1, 30)):
    """Add Gaussian noise with random sigma."""
    np_img = np.array(img).astype(np.float32)
    sigma = random.uniform(*noise_range)
    noise = np.random.randn(*np_img.shape).astype(np.float32) * sigma
    np_img = np.clip(np_img + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(np_img)


def random_jpeg(img, quality_range=(15, 80)):
    """Apply JPEG compression with random quality."""
    quality = random.randint(*quality_range)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def degrade_hr(hr_img, prob=0.9):
    """Apply strong blur on HR before downsampling (optical defocus)."""
    if random.random() > prob:
        return hr_img
    img = hr_img.copy()
    if random.random() < 0.8:
        img = random_gaussian_blur(img)
    return img


def degrade_lr(lr_img, prob=0.9):
    """Apply noise and JPEG on LR after downsampling (sensor + compression)."""
    if random.random() > prob:
        return lr_img
    img = lr_img.copy()
    if random.random() < 0.7:
        img = random_noise(img)
    if random.random() < 0.5:
        img = random_jpeg(img)
    return img
