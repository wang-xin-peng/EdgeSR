"""
Real-world degradation functions for blind SR training.

Pipeline: HR → Blur → Noise → JPEG compression → Bicubic down → LR
"""

import random
import io
import numpy as np
from PIL import Image, ImageFilter


def random_gaussian_blur(img, kernel_range=(3, 7), sigma_range=(0.5, 3.0)):
    """Apply Gaussian blur with random kernel size and sigma."""
    kernel_size = random.randrange(kernel_range[0], kernel_range[1] + 1, 2)
    sigma = random.uniform(*sigma_range)
    return img.filter(ImageFilter.GaussianBlur(radius=sigma))


def random_noise(img, noise_range=(1, 15)):
    """Add Gaussian noise with random sigma."""
    np_img = np.array(img).astype(np.float32)
    sigma = random.uniform(*noise_range)
    noise = np.random.randn(*np_img.shape).astype(np.float32) * sigma
    np_img = np.clip(np_img + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(np_img)


def random_jpeg(img, quality_range=(30, 95)):
    """Apply JPEG compression with random quality."""
    quality = random.randint(*quality_range)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def apply_degradation(hr_img, prob=0.8, use_blur=True, use_noise=True, use_jpeg=True):
    """
    Apply random degradation to HR image before downsampling.

    Args:
        hr_img: PIL Image (HR patch)
        prob: probability of applying degradation
        use_blur: enable Gaussian blur
        use_noise: enable Gaussian noise
        use_jpeg: enable JPEG compression

    Returns:
        degraded PIL Image
    """
    if random.random() > prob:
        return hr_img

    img = hr_img.copy()

    if use_blur and random.random() < 0.6:
        img = random_gaussian_blur(img)

    if use_noise and random.random() < 0.5:
        img = random_noise(img)

    if use_jpeg and random.random() < 0.4:
        img = random_jpeg(img)

    return img
