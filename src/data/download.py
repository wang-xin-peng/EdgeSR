"""
Dataset download and preprocessing for EdgeSR.

Downloads DIV2K, Flickr2K (training) and Set5, Set14, BSD100 (testing).
Pre-crops HR images into patches for faster training data loading.

Usage:
    python src/data/download.py --div2k_root ./data/DIV2K

Notes:
    - Flickr2K is optional. The model trains well on DIV2K alone (900 images).
    - If downloads fail, place datasets manually in the expected directories:
      ./data/DIV2K/DIV2K_train_HR/  (800 PNGs)
      ./data/DIV2K/DIV2K_valid_HR/  (100 PNGs)
      ./data/benchmark/Set5/        (5 PNGs)
      ./data/benchmark/Set14/       (14 PNGs)
      ./data/benchmark/BSD100/      (100 PNGs)
"""

import os
import argparse
import zipfile
import tarfile
import shutil
import requests
import urllib3
from tqdm import tqdm
from multiprocessing import Pool
from PIL import Image

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

DIV2K_URLS = {
    "train_hr": "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip",
    "valid_hr": "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip",
}

# Flickr2K — 原版 EDSR 论文提供的官方下载地址（SNU 首尔大学服务器）
FLICKR2K_URL = "https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar"

# 测试集 — EDSR 官方 benchmark.tar（Set5 + Set14 + BSD100 + Urban100，~50MB）
# 来自 SNU 首尔大学视觉实验室
BENCHMARK_URL = "https://cv.snu.ac.kr/research/EDSR/benchmark.tar"


def download_file(url, save_path, verify=True):
    """Download a file from url to save_path with progress bar."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
        print(f"  Already exists: {save_path}")
        return
    print(f"  Downloading {url}...")
    try:
        response = requests.get(url, stream=True, timeout=(30, 300), verify=verify)
        response.raise_for_status()
    except (requests.exceptions.SSLError, requests.exceptions.ConnectionError):
        if verify:
            print("  Connection error, retrying without verification...")
            download_file(url, save_path, verify=False)
            return
        raise
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


def extract_tar(tar_path, extract_to):
    """Extract a tar file (auto-detect compression)."""
    if os.path.exists(extract_to) and len(os.listdir(extract_to)) > 0:
        print(f"  Already extracted: {extract_to}")
        return
    print(f"  Extracting {tar_path}...")
    with tarfile.open(tar_path, "r:*") as tf:
        tf.extractall(extract_to)
    print(f"  Extracted to {extract_to}")


def preprocess_hr_patch(args):
    """Pre-crop a single HR image into patches using PIL."""
    img_path, save_dir, patch_size = args
    try:
        img = Image.open(img_path).convert("RGB")
    except Exception:
        print(f"  Warning: could not read {img_path}", flush=True)
        return
    w, h = img.size
    h = h - h % patch_size
    w = w - w % patch_size
    if h == 0 or w == 0:
        return
    img = img.crop((0, 0, w, h))
    name = os.path.splitext(os.path.basename(img_path))[0]
    patch_dir = os.path.join(save_dir, name)
    os.makedirs(patch_dir, exist_ok=True)
    idx = 0
    for y in range(0, h, patch_size):
        for x in range(0, w, patch_size):
            patch = img.crop((x, y, x + patch_size, y + patch_size))
            out_path = os.path.join(patch_dir, f"{idx:04d}.png")
            patch.save(out_path)
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
    parser.add_argument("--skip_flickr2k", action="store_true", help="Skip Flickr2K download")
    parser.add_argument("--skip_benchmark", action="store_true", help="Skip benchmark download")
    args = parser.parse_args()

    # --- Download ---
    if not args.skip_download:
        print("=== Downloading DIV2K ===")
        for key, url in DIV2K_URLS.items():
            save_path = os.path.join(args.div2k_root, f"{key}.zip")
            download_file(url, save_path)

        if not args.skip_flickr2k:
            print("\n=== Downloading Flickr2K ===")
            save_path = os.path.join(args.flickr2k_root, "Flickr2K.tar")
            download_file(FLICKR2K_URL, save_path)

        if not args.skip_benchmark:
            print("\n=== Downloading Benchmark Sets (EDSR benchmark.tar) ===")
            tar_path = os.path.join(args.benchmark_root, "benchmark.tar")
            download_file(BENCHMARK_URL, tar_path)

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
    tar_path = os.path.join(args.flickr2k_root, "Flickr2K.tar")
    if os.path.exists(tar_path):
        extract_tar(tar_path, args.flickr2k_root)

    print("\n=== Extracting Benchmark Sets ===")
    tar_path = os.path.join(args.benchmark_root, "benchmark.tar")
    if os.path.exists(tar_path):
        extract_dir = os.path.join(args.benchmark_root, "_tmp_extract")
        os.makedirs(extract_dir, exist_ok=True)
        print(f"  Extracting {tar_path}...")
        with tarfile.open(tar_path, "r:*") as tf:
            tf.extractall(extract_dir)
        # Move HR files: _tmp_extract/benchmark/{name}/HR/* → benchmark_root/{name}/*
        src_root = os.path.join(extract_dir, "benchmark")
        if os.path.exists(src_root):
            for name in os.listdir(src_root):
                hr_dir = os.path.join(src_root, name, "HR")
                if not os.path.isdir(hr_dir):
                    continue
                target = os.path.join(args.benchmark_root, name)
                os.makedirs(target, exist_ok=True)
                for fname in os.listdir(hr_dir):
                    shutil.move(os.path.join(hr_dir, fname), os.path.join(target, fname))
                print(f"  Moved {name} HR images to {target}")
        shutil.rmtree(extract_dir, ignore_errors=True)
        print(f"  Benchmark sets ready at {args.benchmark_root}")

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
