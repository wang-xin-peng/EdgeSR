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


class TrainDataset(Dataset):
    """Training dataset loading HR patches and generating LR via bicubic downsampling."""

    def __init__(self, patch_dirs, scale=2, patch_size=96, augment=True):
        """
        Args:
            patch_dirs: list of directories containing HR patch images
            scale: upsampling factor (2 or 4)
            patch_size: LR patch size (HR will be patch_size * scale)
            augment: whether to apply random flip/rotation
        """
        self.scale = scale
        self.hr_size = patch_size * scale
        self.augment = augment
        self.patch_paths = []
        for patch_dir in patch_dirs:
            if not os.path.exists(patch_dir):
                continue
            for img_name in sorted(os.listdir(patch_dir)):
                if img_name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                    self.patch_paths.append(os.path.join(patch_dir, img_name))

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
        # Generate LR via bicubic downsampling
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


def create_train_dataloader(config):
    """Create training DataLoader from config."""
    div2k_patches = os.path.join(config["data"]["div2k_root"], f"train_patches_{config['data']['patch_size']}")
    flickr_patches = os.path.join(config["data"]["flickr2k_root"], f"patches_{config['data']['patch_size']}")
    dataset = TrainDataset(
        patch_dirs=[div2k_patches, flickr_patches],
        scale=config["data"]["scale"],
        patch_size=config["data"]["patch_size"] // config["data"]["scale"],
        augment=True,
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
