"""
EdgeSR 数据集类。

加载预裁剪的 HR 补丁，通过双三次下采样生成 LR，
返回（LR, HR）训练/评估对。
"""

import os
import random
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np

from .degradation import apply_degradation


class TrainDataset(Dataset):
    """训练数据集，加载 HR 补丁并通过双三次下采样生成 LR。"""

    def __init__(self, patch_dirs, scale=2, patch_size=96, augment=True, degradation=True):
        """
        参数：
            patch_dirs：包含 HR 补丁图片的目录列表
            scale：上采样因子（2 或 4）
            patch_size：LR 补丁大小（HR 为 patch_size * scale）
            augment：是否应用随机翻转/旋转
            degradation：是否应用真实退化（模糊/噪声/JPEG）
        """
        self.scale = scale
        self.hr_size = patch_size * scale
        self.augment = augment
        self.degradation = degradation
        self.patch_paths = []
        for patch_dir in patch_dirs:
            if not os.path.exists(patch_dir):
                continue
            for root, _, files in os.walk(patch_dir):
                for img_name in sorted(files):
                    if img_name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                        self.patch_paths.append(os.path.join(root, img_name))

    def __len__(self):
        return len(self.patch_paths)

    def __getitem__(self, idx):
        hr_img = Image.open(self.patch_paths[idx]).convert("RGB")
        # 如果 HR 补丁过大，随机裁剪
        if hr_img.width > self.hr_size or hr_img.height > self.hr_size:
            w, h = TF.get_image_size(hr_img)
            top = random.randint(0, h - self.hr_size)
            left = random.randint(0, w - self.hr_size)
            hr_img = TF.crop(hr_img, top, left, self.hr_size, self.hr_size)
        else:
            hr_img = TF.resize(hr_img, (self.hr_size, self.hr_size), Image.BICUBIC)
        # 应用二阶退化（包含模糊 + 缩放 + 噪声 + JPEG）
        if self.degradation:
            lr_img = apply_degradation(hr_img, self.scale)
        else:
            lr_size = self.hr_size // self.scale
            lr_img = TF.resize(hr_img, (lr_size, lr_size), Image.BICUBIC)
        # 转为张量 [0, 1]
        hr_tensor = TF.to_tensor(hr_img)  # [C, H, W], [0, 1]
        lr_tensor = TF.to_tensor(lr_img)
        # 数据增强
        if self.augment:
            if random.random() > 0.5:
                hr_tensor = TF.hflip(hr_tensor)
                lr_tensor = TF.hflip(lr_tensor)
            if random.random() > 0.5:
                hr_tensor = TF.vflip(hr_tensor)
                lr_tensor = TF.vflip(lr_tensor)
            k = random.randint(0, 3)
            if k > 0:
                hr_tensor = TF.rotate(hr_tensor, 90 * k, expand=False)
                lr_tensor = TF.rotate(lr_tensor, 90 * k, expand=False)
        return lr_tensor, hr_tensor


class TestDataset(Dataset):
    """测试数据集，加载完整 HR 图像并生成 LR。"""

    def __init__(self, hr_dir, scale=2):
        self.scale = scale
        self.hr_paths = sorted([
            os.path.join(hr_dir, f)
            for f in os.listdir(hr_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
        ])

    def __len__(self):
        return len(self.hr_paths)

    def __getitem__(self, idx):
        hr_img = Image.open(self.hr_paths[idx]).convert("RGB")
        hr_w, hr_h = hr_img.size
        # 确保尺寸是缩放因子的整数倍
        hr_w = hr_w - hr_w % self.scale
        hr_h = hr_h - hr_h % self.scale
        hr_img = hr_img.crop((0, 0, hr_w, hr_h))
        # 生成 LR
        lr_size = (hr_w // self.scale, hr_h // self.scale)
        lr_img = hr_img.resize(lr_size, Image.BICUBIC)
        hr_tensor = TF.to_tensor(hr_img)
        lr_tensor = TF.to_tensor(lr_img)
        return lr_tensor, hr_tensor


class DRealSRDataset(Dataset):
    """DRealSR 数据集——来自不同 DSLR 变焦级别的真实 LR-HR 对。

    LR 和 HR 分辨率不同（LR 按缩放因子缩小）。
    预裁剪训练补丁：LR ~380x380，HR ~760x760（x2）。
    """

    def __init__(self, drealsr_root, scale=2, patch_size=96, augment=True):
        self.scale = scale
        self.lr_size = patch_size
        self.hr_size = patch_size * scale
        self.augment = augment
        self.pairs = []
        scale_str = f"x{scale}"
        train_dir = os.path.join(drealsr_root, scale_str, f"Train_{scale_str}")
        hr_dir = os.path.join(train_dir, "train_HR")
        lr_dir = os.path.join(train_dir, "train_LR")
        if os.path.isdir(hr_dir) and os.path.isdir(lr_dir):
            for f in sorted(os.listdir(hr_dir)):
                if f.endswith(".png"):
                    lr_name = f.replace(f"x{scale}", "x1")
                    lr_path = os.path.join(lr_dir, lr_name)
                    hr_path = os.path.join(hr_dir, f)
                    if os.path.exists(lr_path):
                        self.pairs.append((lr_path, hr_path))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        lr_path, hr_path = self.pairs[idx]
        lr_img = Image.open(lr_path).convert("RGB")
        hr_img = Image.open(hr_path).convert("RGB")
        # 在对应位置随机裁剪（LR 比 HR 小 scale 倍）
        if lr_img.width > self.lr_size:
            left = random.randint(0, lr_img.width - self.lr_size)
            top = random.randint(0, lr_img.height - self.lr_size)
        else:
            left, top = 0, 0
            lr_img = TF.resize(lr_img, (self.lr_size, self.lr_size), Image.BICUBIC)
            hr_img = TF.resize(hr_img, (self.hr_size, self.hr_size), Image.BICUBIC)
        lr_patch = TF.crop(lr_img, top, left, self.lr_size, self.lr_size)
        hr_patch = TF.crop(hr_img, top * self.scale, left * self.scale, self.hr_size, self.hr_size)
        hr_tensor = TF.to_tensor(hr_patch)
        lr_tensor = TF.to_tensor(lr_patch)
        if self.augment:
            if random.random() > 0.5:
                hr_tensor = TF.hflip(hr_tensor)
                lr_tensor = TF.hflip(lr_tensor)
            if random.random() > 0.5:
                hr_tensor = TF.vflip(hr_tensor)
                lr_tensor = TF.vflip(lr_tensor)
            k = random.randint(0, 3)
            if k > 0:
                hr_tensor = TF.rotate(hr_tensor, 90 * k, expand=False)
                lr_tensor = TF.rotate(lr_tensor, 90 * k, expand=False)
        return lr_tensor, hr_tensor


def create_train_dataloader(config):
    """根据配置创建训练数据加载器"""
    use_drealsr = config["data"].get("use_drealsr", False)
    if use_drealsr:
        dataset = DRealSRDataset(
            drealsr_root=config["data"]["drealsr_root"],
            scale=config["data"]["scale"],
            patch_size=config["data"]["patch_size"] // config["data"]["scale"],
            augment=True,
        )
    else:
        div2k_patches = os.path.join(config["data"]["div2k_root"], f"train_patches_{config['data']['patch_size']}")
        flickr_patches = os.path.join(config["data"]["flickr2k_root"], f"patches_{config['data']['patch_size']}")
        dataset = TrainDataset(
            patch_dirs=[div2k_patches, flickr_patches],
            scale=config["data"]["scale"],
            patch_size=config["data"]["patch_size"] // config["data"]["scale"],
            augment=True,
            degradation=config["data"].get("degradation", False),
        )
    return DataLoader(
        dataset,
        batch_size=config["data"]["train_batch_size"],
        shuffle=True,
        num_workers=config["data"]["num_workers"],
        pin_memory=True,
    )


def create_test_dataloader(config, name):
    """为基准集创建测试数据加载器"""
    hr_dir = os.path.join(config["data"]["test_root"], name)
    dataset = TestDataset(hr_dir, scale=config["data"]["scale"])
    return DataLoader(
        dataset,
        batch_size=config["data"]["test_batch_size"],
        shuffle=False,
        num_workers=config["data"]["num_workers"],
    ), dataset.hr_paths
