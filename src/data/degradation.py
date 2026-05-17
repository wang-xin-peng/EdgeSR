"""
盲 SR 训练的真实退化函数。

流水线（Real-ESRGAN 风格二阶退化）：
  HR → 第一阶：模糊 → 缩放 → 噪声 → JPEG
     → 第二阶：模糊 → 缩放 → 噪声 → JPEG
     → LR

每阶随机选择模糊类型（高斯/平均/运动）、
缩放方式（双三次/双线性/最近邻/Lanczos）和
噪声/JPEG 参数。两阶顺序组合产生比单次更多样的退化。
"""

import random
import io
import math
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
from scipy.ndimage import convolve



# 模糊类型
def gaussian_blur(img):
    """随机 sigma 的高斯模糊。"""
    sigma = random.uniform(0.5, 6.0)
    return img.filter(ImageFilter.GaussianBlur(radius=sigma))


def average_blur(img):
    """随机半径的盒式（平均）模糊。"""
    radius = random.randint(2, 6)
    return img.filter(ImageFilter.BoxBlur(radius))


def _motion_kernel(ksize, angle):
    """在指定角度（度）上创建运动模糊直线核。"""
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
    """随机核大小和方向的运动模糊。"""
    ksize = random.randrange(7, 21, 2)
    angle = random.uniform(0, 180)
    kernel = _motion_kernel(ksize, angle)
    np_img = np.array(img).astype(np.float32)
    channels = [convolve(np_img[..., c], kernel, mode="reflect") for c in range(3)]
    result = np.stack(channels, axis=-1)
    return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))


def random_blur(img):
    """随机选择一种模糊类型并应用。"""
    blur_type = random.choice(["gaussian", "average", "motion"])
    if blur_type == "gaussian":
        return gaussian_blur(img)
    elif blur_type == "average":
        return average_blur(img)
    else:
        return motion_blur(img)



# 噪声与 JPEG
def random_noise(img, noise_range=(1, 30)):
    """随机 sigma 的高斯噪声。"""
    np_img = np.array(img).astype(np.float32)
    sigma = random.uniform(*noise_range)
    noise = np.random.randn(*np_img.shape).astype(np.float32) * sigma
    np_img = np.clip(np_img + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(np_img)


def random_jpeg(img, quality_range=(15, 80)):
    """随机质量的 JPEG 压缩。"""
    quality = random.randint(*quality_range)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")



# 缩放辅助
_RESIZE_MODES = [Image.BICUBIC, Image.BILINEAR, Image.NEAREST, Image.LANCZOS]


def random_resize(img, factor):
    """用随机插值方式将图像缩放 `factor` 倍（>=1 为下采样）。"""
    if factor <= 1:
        return img
    w, h = img.size
    new_w = max(1, int(round(w / factor)))
    new_h = max(1, int(round(h / factor)))
    mode = random.choice(_RESIZE_MODES)
    return img.resize((new_w, new_h), mode)


# 色彩与 ISP 风格退化（超越 Real-ESRGAN）
def color_jitter(img):
    """随机亮度、对比度、饱和度和色相偏移。"""
    if random.random() < 0.5:
        factor = random.uniform(0.6, 1.4)
        img = ImageEnhance.Brightness(img).enhance(factor)
    if random.random() < 0.5:
        factor = random.uniform(0.6, 1.4)
        img = ImageEnhance.Contrast(img).enhance(factor)
    if random.random() < 0.5:
        factor = random.uniform(0.5, 1.5)
        img = ImageEnhance.Color(img).enhance(factor)
    return img


def double_jpeg(img, base_quality=(30, 70), second_quality=(50, 85)):
    """用不同质量两次 JPEG 压缩——模拟二次上传压缩。"""
    q1 = random.randint(*base_quality)
    q2 = random.randint(*second_quality)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=q1)
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=q2)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


# 单阶退化
def degrade_one_stage(img, scale_factor, prob=0.9):
    """一轮退化：模糊 → 缩放 → 噪声 → JPEG。"""
    if random.random() > prob:
        return random_resize(img, scale_factor) if scale_factor > 1 else img

    out = img.copy()

    # 模糊（60% 概率）
    if random.random() < 0.6:
        out = random_blur(out)

    # 色彩 / ISP 抖动（40% 概率——在缩放前，模拟 ISP 流水线）
    if random.random() < 0.4:
        out = color_jitter(out)

    # 下采样
    if scale_factor > 1:
        out = random_resize(out, scale_factor)

    # 噪声（50% 概率）
    if random.random() < 0.5:
        out = random_noise(out)

    # JPEG 压缩（40% 概率单次，20% 概率双重）
    if random.random() < 0.4:
        if random.random() < 0.3:
            out = double_jpeg(out)
        else:
            out = random_jpeg(out)

    return out


# 二阶退化（主入口）
def apply_degradation(hr_img, total_scale=2, p_stage1=0.9, p_stage2=0.8):
    """应用 Real-ESRGAN 风格的二阶退化。

    总缩放因子被随机拆分为两部分，使每阶看到不同的下采样倍数，
    大幅增加输出退化的多样性。
    """
    # 将总缩放因子拆分为两部分
    if total_scale <= 1:
        return hr_img

    s1 = random.uniform(max(1.05, total_scale / 1.5), min(total_scale / 1.05, 1.6))
    s2 = total_scale / s1

    # 第一阶退化
    lr = degrade_one_stage(hr_img, s1, prob=p_stage1)

    # 对已退化的图像做第二阶退化
    lr = degrade_one_stage(lr, s2, prob=p_stage2)

    # 确保目标尺寸精确，避免训练时形状不匹配
    hr_w, hr_h = hr_img.size
    target_w = hr_w // total_scale
    target_h = hr_h // total_scale
    if lr.size != (target_w, target_h):
        lr = lr.resize((target_w, target_h), Image.BICUBIC)

    return lr
