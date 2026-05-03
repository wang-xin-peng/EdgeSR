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
import requests
from tqdm import tqdm
from multiprocessing import Pool
from PIL import Image

DIV2K_URLS = {
    "train_hr": "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip",
    "valid_hr": "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip",
}

# Multiple mirrors for Flickr2K (all may not be available; script will try each)
FLICKR2K_URLS = [
    "https://cvnote.ddns.net/Flickr2K/Flickr2K.tar.gz",
    "https://huggingface.co/datasets/zh-plus/Flickr2K/resolve/main/Flickr2K.tar.gz",
]

# Multiple mirrors for benchmark datasets
BENCHMARK_URLS = {
    "Set5": [
        "https://cvnote.ddns.net/SR_test_datasets/Set5.zip",
        "https://huggingface.co/datasets/lllych/Set5/resolve/main/Set5.zip",
    ],
    "Set14": [
        "https://cvnote.ddns.net/SR_test_datasets/Set14.zip",
        "https://huggingface.co/datasets/lllych/Set14/resolve/main/Set14.zip",
    ],
    "BSD100": [
        "https://cvnote.ddns.net/SR_test_datasets/BSD100.zip",
        "https://huggingface.co/datasets/lllych/BSD100/resolve/main/BSD100.zip",
    ],
}


def download_file(url, save_path):
    """Download a file from url to save_path with progress bar. Returns True on success."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
        print(f"  Already exists: {save_path}")
        return True
    try:
        print(f"  Downloading {url}...")
        response = requests.get(url, stream=True, timeout=(30, 300))
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with open(save_path, "wb") as f:
            with tqdm(total=total, unit="B", unit_scale=True) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
        print(f"  Saved to {save_path}")
        return True
    except requests.RequestException as e:
        print(f"  Failed: {e}")
        return False


def download_with_fallback(urls, save_path):
    """Try multiple URLs for the same file. Returns True if any succeeds."""
    if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
        print(f"  Already exists: {save_path}")
        return True
    for url in urls:
        if download_file(url, save_path):
            return True
        # Remove partial download on failure
        if os.path.exists(save_path):
            os.remove(save_path)
    return False


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
            save_path = os.path.join(args.flickr2k_root, "Flickr2K.tar.gz")
            if not download_with_fallback(FLICKR2K_URLS, save_path):
                print("  Warning: Flickr2K download failed. Training will use DIV2K only.")

        if not args.skip_benchmark:
            print("\n=== Downloading Benchmark Sets ===")
            for name, urls in BENCHMARK_URLS.items():
                save_path = os.path.join(args.benchmark_root, f"{name}.zip")
                if not download_with_fallback(urls, save_path):
                    print(f"  Warning: {name} download failed. Will skip evaluation on this set.")

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
