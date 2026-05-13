"""
Real-world degradation functions for blind SR training.

Pipeline (Real-ESRGAN style second-order degradation):
  HR → 1st order: blur → resize → noise → JPEG
     → 2nd order: blur → resize → noise → JPEG
     → LR

Each order picks random blur type (Gaussian / average / motion),
random resize mode (bicubic / bilinear / nearest / lanczos), and
random noise/JPEG parameters. Two sequential orders produce vastly
more diverse degradation than a single pass.
"""

import random
import io
import math
import numpy as np
from PIL import Image, ImageFilter
from scipy.ndimage import convolve


# ---------------------------------------------------------------------------
# Blur types
# ---------------------------------------------------------------------------

def gaussian_blur(img):
    """Gaussian blur with random sigma."""
    sigma = random.uniform(0.5, 6.0)
    return img.filter(ImageFilter.GaussianBlur(radius=sigma))


def average_blur(img):
    """Box (average) blur with random radius."""
    radius = random.randint(2, 6)
    return img.filter(ImageFilter.BoxBlur(radius))


def _motion_kernel(ksize, angle):
    """Create a line kernel at the given angle (degrees) for motion blur."""
    kernel = np.zeros((ksize, ksize))
    center = ksize // 2
    rad = math.radians(angle)
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    for i in range(ksize):
        dy = i - center
        for j in range(ksize):
            dx = j - center
            dist = abs(dx * cos_a + dy * sin_a)
            if dist < 0.7:
                kernel[i, j] = 1.0
    total = kernel.sum()
    if total == 0:
        kernel[center, center] = 1.0
        total = 1.0
    return kernel / total


def motion_blur(img):
    """Motion blur with random kernel size and direction."""
    ksize = random.randrange(7, 21, 2)
    angle = random.uniform(0, 180)
    kernel = _motion_kernel(ksize, angle)
    np_img = np.array(img).astype(np.float32)
    channels = [convolve(np_img[..., c], kernel, mode="reflect") for c in range(3)]
    result = np.stack(channels, axis=-1)
    return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))


def random_blur(img):
    """Apply a randomly chosen blur type."""
    blur_type = random.choice(["gaussian", "average", "motion"])
    if blur_type == "gaussian":
        return gaussian_blur(img)
    elif blur_type == "average":
        return average_blur(img)
    else:
        return motion_blur(img)


# ---------------------------------------------------------------------------
# Noise & JPEG
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Resize helpers
# ---------------------------------------------------------------------------

_RESIZE_MODES = [Image.BICUBIC, Image.BILINEAR, Image.NEAREST, Image.LANCZOS]


def random_resize(img, factor):
    """Resize image by `factor` (>=1 = downsampling) using a random interp."""
    if factor <= 1:
        return img
    w, h = img.size
    new_w = max(1, int(round(w / factor)))
    new_h = max(1, int(round(h / factor)))
    mode = random.choice(_RESIZE_MODES)
    return img.resize((new_w, new_h), mode)


# ---------------------------------------------------------------------------
# One stage of degradation
# ---------------------------------------------------------------------------

def degrade_one_stage(img, scale_factor, prob=0.9):
    """One round of: blur → resize → noise → JPEG."""
    if random.random() > prob:
        return random_resize(img, scale_factor) if scale_factor > 1 else img

    out = img.copy()

    # Blur (60% chance)
    if random.random() < 0.6:
        out = random_blur(out)

    # Resize down
    if scale_factor > 1:
        out = random_resize(out, scale_factor)

    # Noise (50% chance)
    if random.random() < 0.5:
        out = random_noise(out)

    # JPEG (40% chance)
    if random.random() < 0.4:
        out = random_jpeg(out)

    return out


# ---------------------------------------------------------------------------
# Second-order degradation (main entry point)
# ---------------------------------------------------------------------------

def apply_degradation(hr_img, total_scale=2, p_stage1=0.9, p_stage2=0.8):
    """Apply Real-ESRGAN-style second-order degradation.

    The total scale factor is randomly split into two parts so that
    each order sees a different downsampling factor, greatly increasing
    the diversity of output degradations.
    """
    # Split total_scale into two factors
    if total_scale <= 1:
        return hr_img

    s1 = random.uniform(max(1.05, total_scale / 1.5), min(total_scale / 1.05, 1.6))
    s2 = total_scale / s1

    # First order
    lr = degrade_one_stage(hr_img, s1, prob=p_stage1)

    # Second order on the already-degraded image
    lr = degrade_one_stage(lr, s2, prob=p_stage2)

    # Ensure exact target size to avoid shape mismatches in training
    hr_w, hr_h = hr_img.size
    target_w = hr_w // total_scale
    target_h = hr_h // total_scale
    if lr.size != (target_w, target_h):
        lr = lr.resize((target_w, target_h), Image.BICUBIC)

    return lr
