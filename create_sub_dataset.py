"""
Docstring for create_sub_dataset
Usage example:
python make_sub_dataset.py --train-total 5000 --val-total 500
"""



#!/usr/bin/env python3
import argparse
import random
import shutil
from pathlib import Path


def collect_images_in_dir(dir_path):
    exts = {".jpg", ".jpeg", ".JPEG", ".JPG"}
    return [p for p in dir_path.iterdir() if p.is_file() and p.suffix in exts]


def make_train_subset(root: Path, train_total: int, seed: int = 42):
    train_dir    = root / "train"
    train_sub_dir = root / "train_{}".format(train_total)

    if not train_dir.is_dir():
        raise SystemExit(f"Train directory not found: {train_dir}")

    print(f"Building train_n subset in: {train_sub_dir}")
    train_sub_dir.mkdir(parents=True, exist_ok=True)

    class_dirs = sorted([d for d in train_dir.iterdir() if d.is_dir()])
    num_classes = len(class_dirs)
    if num_classes == 0:
        raise SystemExit(f"No class folders found in: {train_dir}")

    per_class = train_total // num_classes
    if per_class == 0:
        raise SystemExit(f"train_total={train_total} too small for {num_classes} classes.")

    print(f"Found {num_classes} classes.")
    print(f"Sampling {per_class} images per class to reach ≈{train_total} total.")

    random.seed(seed)
    total_copied = 0

    for class_dir in class_dirs:
        class_name = class_dir.name
        dst_class_dir = train_sub_dir / class_name
        dst_class_dir.mkdir(parents=True, exist_ok=True)

        images = collect_images_in_dir(class_dir)
        if not images:
            print(f"  [SKIP] {class_name}: no images found.")
            continue

        take = min(per_class, len(images))
        selected = random.sample(images, take)

        print(f"  {class_name}: taking {take} / {len(images)}")

        for img in selected:
            shutil.copy2(img, dst_class_dir / img.name)
            total_copied += 1

    print(f"\ntrain_n created with {total_copied} images.")


def make_val_subset(root: Path, val_total: int, seed: int = 42):
    val_dir    = root / "val"
    val_sub_dir = root / "val_{}".format(val_total)

    if not val_dir.is_dir():
        raise SystemExit(f"Val directory not found: {val_dir}")

    print(f"\nBuilding val_n subset in: {val_sub_dir}")
    val_sub_dir.mkdir(parents=True, exist_ok=True)

    images = collect_images_in_dir(val_dir)
    if not images:
        raise SystemExit(f"No validation images found in: {val_dir}")

    random.seed(seed + 1)
    take = min(val_total, len(images))
    selected = random.sample(images, take)

    print(f"Sampling {take} images for val_n...")

    for img in selected:
        shutil.copy2(img, val_sub_dir / img.name)

    print(f"val_n created with {take} images.")


def main():
    parser = argparse.ArgumentParser(
        description="Create small ImageNet subsets (train_n, val_n)."
    )
    parser.add_argument(
        "--root",
        type=str,
        default="./imagenet",
        help="Root directory containing 'train' and 'val' folders. Default: ./imagenet",
    )
    parser.add_argument(
        "--train-total",
        type=int,
        default=5000,
        help="Total images for train_n (balanced). Default: 5000.",
    )
    parser.add_argument(
        "--val-total",
        type=int,
        default=500,
        help="Total images for val_n. Default: 500.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed. Default: 42.",
    )

    args = parser.parse_args()
    root = Path(args.root).resolve()

    print(f"Using ImageNet root: {root}")

    make_train_subset(root, args.train_total, seed=args.seed)
    make_val_subset(root, args.val_total, seed=args.seed)


if __name__ == "__main__":
    main()
