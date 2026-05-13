"""
Dataset classes for EdgeSR.

Loads pre-cropped HR patches, generates LR via bicubic downsampling,
and returns (LR, HR) pairs for training/evaluation.
"""

import os
import random
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np

from .degradation import apply_degradation


class TrainDataset(Dataset):
    """Training dataset loading HR patches and generating LR via bicubic downsampling."""

    def __init__(self, patch_dirs, scale=2, patch_size=96, augment=True, degradation=True):
        """
        Args:
            patch_dirs: list of directories containing HR patch images
            scale: upsampling factor (2 or 4)
            patch_size: LR patch size (HR will be patch_size * scale)
            augment: whether to apply random flip/rotation
            degradation: whether to apply real-world degradation (blur/noise/JPEG)
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
        # If HR patch is larger than needed, randomly crop
        if hr_img.width > self.hr_size or hr_img.height > self.hr_size:
            w, h = TF.get_image_size(hr_img)
            top = random.randint(0, h - self.hr_size)
            left = random.randint(0, w - self.hr_size)
            hr_img = TF.crop(hr_img, top, left, self.hr_size, self.hr_size)
        else:
            hr_img = TF.resize(hr_img, (self.hr_size, self.hr_size), Image.BICUBIC)
        # Apply second-order degradation (handles blur + resize + noise + JPEG)
        if self.degradation:
            lr_img = apply_degradation(hr_img, self.scale)
        else:
            lr_size = self.hr_size // self.scale
            lr_img = TF.resize(hr_img, (lr_size, lr_size), Image.BICUBIC)
        # Convert to tensor [0, 1]
        hr_tensor = TF.to_tensor(hr_img)  # [C, H, W], [0, 1]
        lr_tensor = TF.to_tensor(lr_img)
        # Data augmentation
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
    """Test dataset loading full HR images and generating LR."""

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
        # Ensure dimensions are multiples of scale
        hr_w = hr_w - hr_w % self.scale
        hr_h = hr_h - hr_h % self.scale
        hr_img = hr_img.crop((0, 0, hr_w, hr_h))
        # Generate LR
        lr_size = (hr_w // self.scale, hr_h // self.scale)
        lr_img = hr_img.resize(lr_size, Image.BICUBIC)
        hr_tensor = TF.to_tensor(hr_img)
        lr_tensor = TF.to_tensor(lr_img)
        return lr_tensor, hr_tensor


class DRealSRDataset(Dataset):
    """DRealSR dataset — real LR-HR pairs from different DSLR zoom levels.

    LR and HR have different resolutions (LR is smaller by scale factor).
    Pre-cropped training patches at LR ~380x380, HR ~760x760 for x2.
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
        # Random crop at corresponding positions (LR smaller by scale factor)
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
    """Create training DataLoader from config."""
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
    """Create test DataLoader for a benchmark set."""
    hr_dir = os.path.join(config["data"]["test_root"], name)
    dataset = TestDataset(hr_dir, scale=config["data"]["scale"])
    return DataLoader(
        dataset,
        batch_size=config["data"]["test_batch_size"],
        shuffle=False,
        num_workers=config["data"]["num_workers"],
    ), dataset.hr_paths
