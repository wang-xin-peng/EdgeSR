"""
Dataset download and preprocessing for EdgeSR.

Downloads DIV2K, Flickr2K (training) and Set5, Set14, BSD100 (testing).
Pre-crops HR images into patches for faster training data loading.

Usage:
    python src/data/download.py --div2k_root ./data/DIV2K --flickr2k_root ./data/Flickr2K
"""

import os
import argparse
import zipfile
import tarfile
import requests
from tqdm import tqdm
from multiprocessing import Pool
import cv2
import numpy as np

DIV2K_URLS = {
    "train_hr": "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip",
    "valid_hr": "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip",
}

FLICKR2K_URL = "https://cvnote.ddns.net/Flickr2K/Flickr2K.tar.gz"

BENCHMARK_URLS = {
    "Set5": "https://cvnote.ddns.net/SR_test_datasets/Set5.zip",
    "Set14": "https://cvnote.ddns.net/SR_test_datasets/Set14.zip",
    "BSD100": "https://cvnote.ddns.net/SR_test_datasets/BSD100.zip",
}


def download_file(url, save_path):
    """Download a file from url to save_path with progress bar."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if os.path.exists(save_path):
        print(f"  Already exists: {save_path}")
        return
    print(f"  Downloading {url}...")
    response = requests.get(url, stream=True, timeout=300)
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))
    with open(save_path, "wb") as f:
        with tqdm(total=total, unit="B", unit_scale=True) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))
    print(f"  Saved to {save_path}")


def extract_zip(zip_path, extract_to):
    """Extract a zip file."""
    if os.path.exists(extract_to) and len(os.listdir(extract_to)) > 0:
        print(f"  Already extracted: {extract_to}")
        return
    print(f"  Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)
    print(f"  Extracted to {extract_to}")


def extract_tar_gz(tar_path, extract_to):
    """Extract a tar.gz file."""
    if os.path.exists(extract_to) and len(os.listdir(extract_to)) > 0:
        print(f"  Already extracted: {extract_to}")
        return
    print(f"  Extracting {tar_path}...")
    with tarfile.open(tar_path, "r:gz") as tf:
        tf.extractall(extract_to)
    print(f"  Extracted to {extract_to}")


def preprocess_hr_patch(args):
    """Pre-crop a single HR image into patches."""
    img_path, save_dir, patch_size = args
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        return
    h, w, _ = img.shape
    # Ensure dimensions are multiples of patch_size
    h = h - h % patch_size
    w = w - w % patch_size
    if h == 0 or w == 0:
        return
    img = img[:h, :w]
    name = os.path.splitext(os.path.basename(img_path))[0]
    patch_dir = os.path.join(save_dir, name)
    os.makedirs(patch_dir, exist_ok=True)
    idx = 0
    for y in range(0, h, patch_size):
        for x in range(0, w, patch_size):
            patch = img[y:y + patch_size, x:x + patch_size]
            out_path = os.path.join(patch_dir, f"{idx:04d}.png")
            cv2.imwrite(out_path, patch)
            idx += 1


def preprocess_dataset(hr_dir, patch_dir, patch_size=192, num_workers=8):
    """Pre-crop all HR images in hr_dir into patches."""
    if os.path.exists(patch_dir) and len(os.listdir(patch_dir)) > 0:
        print(f"  Patches already exist: {patch_dir}")
        return
    print(f"  Pre-cropping {hr_dir} into patches ({patch_size}x{patch_size})...")
    img_paths = [
        os.path.join(hr_dir, f)
        for f in sorted(os.listdir(hr_dir))
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
    ]
    args_list = [(p, patch_dir, patch_size) for p in img_paths]
    with Pool(num_workers) as pool:
        list(tqdm(pool.imap_unordered(preprocess_hr_patch, args_list), total=len(args_list)))
    print(f"  Patches saved to {patch_dir}")


def main():
    parser = argparse.ArgumentParser(description="Download and preprocess SR datasets")
    parser.add_argument("--div2k_root", type=str, default="./data/DIV2K")
    parser.add_argument("--flickr2k_root", type=str, default="./data/Flickr2K")
    parser.add_argument("--benchmark_root", type=str, default="./data/benchmark")
    parser.add_argument("--patch_size", type=int, default=192)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--skip_download", action="store_true", help="Skip download (extract only)")
    args = parser.parse_args()

    # --- Download ---
    if not args.skip_download:
        print("=== Downloading DIV2K ===")
        for key, url in DIV2K_URLS.items():
            save_path = os.path.join(args.div2k_root, f"{key}.zip")
            download_file(url, save_path)

        print("\n=== Downloading Flickr2K ===")
        save_path = os.path.join(args.flickr2k_root, "Flickr2K.tar.gz")
        download_file(FLICKR2K_URL, save_path)

        print("\n=== Downloading Benchmark Sets ===")
        for name, url in BENCHMARK_URLS.items():
            save_path = os.path.join(args.benchmark_root, f"{name}.zip")
            download_file(url, save_path)

    # --- Extract ---
    print("\n=== Extracting DIV2K ===")
    for key in DIV2K_URLS:
        zip_path = os.path.join(args.div2k_root, f"{key}.zip")
        if key == "train_hr":
            extract_to = os.path.join(args.div2k_root, "DIV2K_train_HR")
        else:
            extract_to = os.path.join(args.div2k_root, "DIV2K_valid_HR")
        if os.path.exists(zip_path):
            extract_zip(zip_path, extract_to)

    print("\n=== Extracting Flickr2K ===")
    tar_path = os.path.join(args.flickr2k_root, "Flickr2K.tar.gz")
    if os.path.exists(tar_path):
        extract_tar_gz(tar_path, args.flickr2k_root)

    print("\n=== Extracting Benchmark Sets ===")
    for name in BENCHMARK_URLS:
        zip_path = os.path.join(args.benchmark_root, f"{name}.zip")
        extract_to = os.path.join(args.benchmark_root, name)
        if os.path.exists(zip_path):
            extract_zip(zip_path, extract_to)

    # --- Pre-crop training patches ---
    print("\n=== Pre-cropping DIV2K train HR patches ===")
    train_hr_dir = os.path.join(args.div2k_root, "DIV2K_train_HR")
    train_patch_dir = os.path.join(args.div2k_root, f"train_patches_{args.patch_size}")
    if os.path.exists(train_hr_dir):
        preprocess_dataset(train_hr_dir, train_patch_dir, args.patch_size, args.num_workers)

    print("\n=== Pre-cropping DIV2K valid HR patches ===")
    valid_hr_dir = os.path.join(args.div2k_root, "DIV2K_valid_HR")
    valid_patch_dir = os.path.join(args.div2k_root, f"valid_patches_{args.patch_size}")
    if os.path.exists(valid_hr_dir):
        preprocess_dataset(valid_hr_dir, valid_patch_dir, args.patch_size, args.num_workers)

    print("\n=== Pre-cropping Flickr2K HR patches ===")
    flickr_hr_dir = os.path.join(args.flickr2k_root, "Flickr2K")
    flickr_patch_dir = os.path.join(args.flickr2k_root, f"patches_{args.patch_size}")
    if os.path.exists(flickr_hr_dir):
        preprocess_dataset(flickr_hr_dir, flickr_patch_dir, args.patch_size, args.num_workers)

    print("\nAll done!")


if __name__ == "__main__":
    main()
