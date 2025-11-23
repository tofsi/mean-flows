import os
import tarfile
import urllib.request
from pathlib import Path
from typing import Tuple, List

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchvision.datasets.folder import default_loader

from scipy.io import loadmat  # pip install scipy


### Uses main imagenet dataset (ILSVRC2012) #################
# ---------------------------------------------------------
# 1. URLs and basic config
# ---------------------------------------------------------


def extract_tar(src: Path, dest_dir: Path):
    print(f"[extract] Extracting {src} → {dest_dir}")
    dest_dir.mkdir(parents=True, exist_ok=True)
    mode = (
        "r:gz" if src.suffixes[-2:] == [".tar", ".gz"] or src.suffix == ".gz" else "r:"
    )
    with tarfile.open(src, mode) as tar:
        tar.extractall(path=dest_dir)
    print(f"[extract] Done: {src}")


# ---------------------------------------------------------
# 2. Preparing train, val, test folders
# ---------------------------------------------------------


def prepare_train(train_tar: Path, root: Path):
    """
    ILSVRC2012_img_train.tar contains 1000 tar files (one per class),
    e.g. n01440764.tar, each with images for that class.
    We unpack them into: root/train/<wnid>/*.JPEG
    """
    train_root = root / "train"
    if train_root.exists():
        print("[train] Train directory already exists, skipping train extraction.")
        return

    tmp_dir = root / "train_archives"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # First extract the big tar to tmp_dir
    extract_tar(train_tar, tmp_dir)

    train_root.mkdir(parents=True, exist_ok=True)

    # Now tmp_dir contains *.tar per class
    for wnid_tar in sorted(tmp_dir.glob("*.tar")):
        wnid = wnid_tar.stem  # e.g. n01440764
        class_dir = train_root / wnid
        class_dir.mkdir(exist_ok=True)
        print(f"[train] Extracting {wnid_tar.name} → {class_dir}")
        with tarfile.open(wnid_tar, "r:") as tar:
            tar.extractall(path=class_dir)

    print("[train] Finished preparing train set.")


def prepare_test(test_tar: Path, root: Path):
    """
    The test tar has images without labels. We just extract them
    into root/test/ and build a custom Dataset around it.
    """
    test_root = root / "test"
    if test_root.exists() and any(test_root.iterdir()):
        print("[test] Test directory already exists, skipping.")
        return

    extract_tar(test_tar, test_root)
    print("[test] Finished preparing test set.")


# ---------------------------------------------------------
# 3. Custom test dataset (no labels)
# ---------------------------------------------------------


class ImageNetTestDataset(Dataset):
    """
    Test dataset for ImageNet without labels.
    Returns (image_tensor, path_string).
    """

    def __init__(self, root: str, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.paths = sorted(
            [
                p
                for p in self.root.glob("**/*")
                if p.suffix.lower() in [".jpeg", ".jpg", ".png"]
            ]
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, str(path)


# ---------------------------------------------------------
# 4. Building DataLoaders
# ---------------------------------------------------------


def build_dataloaders(
    root_dir: str,
    batch_size: int = 256,
    num_workers: int = 8,
) -> Tuple[DataLoader, DataLoader]:
    """
    Assumes data has already been downloaded and organized into:
      root/train/<wnid>/
      root/test/...
    Builds standard ImageNet transforms + DataLoaders.
    """
    root = Path(root_dir)

    # Standard ImageNet transforms
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )

    val_test_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]
    )

    train_dataset = datasets.ImageFolder(root / "train", transform=train_transform)
    test_dataset = ImageNetTestDataset(root / "test", transform=val_test_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    return train_loader, test_loader


# ---------------------------------------------------------
# 5. Main: download, prepare, and build loaders
# ---------------------------------------------------------


def get_dataloaders(batch_size: int):
    root = Path("/home/silpasoninallacheruvu/imagenet")
    downloads = root / "downloads"
    dev_dir = root / "ILSVRC2012_devkit_t12"

    # 1) Download all files
    dev_tar = downloads / "ILSVRC2012_devkit_t12.tar.gz"
    train_tar = downloads / "ILSVRC2012_img_train.tar"
    val_tar = downloads / "ILSVRC2012_img_val.tar"

    # 2) Extract devkit
    if not dev_dir.exists():
        extract_tar(dev_tar, root)

    # 3) Prepare splits
    prepare_train(train_tar, root)
    prepare_test(val_tar, root)

    # 4) Build dataloaders
    train_loader, test_loader = build_dataloaders(
        root_dir=str(root),
        batch_size=batch_size,
        num_workers=8,
    )

    # 5) Quick sanity check: one batch
    print(">>> Checking one batch from each loader...")
    images, labels = next(iter(train_loader))
    print("Train batch:", images.shape, labels.shape)

    images, _ = next(iter(test_loader))
    print("Test batch:", images.shape)

    print("All done. Ready to train!")

    return train_loader, test_loader


############################################################
# Simple dataloaders for already-extracted ImageNet
############################################################


class UnlabeledImageDataset(torch.utils.data.Dataset):
    def __init__(self, directory, transform=None):
        self.paths = sorted(
            [
                str(p)
                for p in Path(directory).glob("*.*")
                if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".JPEG"]
            ]
        )
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = default_loader(self.paths[idx])
        if self.transform:
            img = self.transform(img)
        return img


def get_dataloaders_extracted(
    root_dir: str,
    batch_size: int = 256,
    num_workers: int = 8,
    train_subdir: str = "train",
    val_subdir: str = "val",  # or "test" if you only have test extracted
) -> Tuple[DataLoader, DataLoader]:
    """
    Use this when ImageNet is ALREADY extracted.

    Expects:
        root_dir/train/<class>/*.JPEG
        root_dir/val/<class>/*.JPEG   (or test/<class>/ if using test)

    Does NOT download or extract anything.
    """

    root = Path(root_dir)
    train_dir = root / train_subdir
    val_dir = root / val_subdir

    if not train_dir.exists():
        raise FileNotFoundError(
            f"Training directory not found: {train_dir}\n"
            "Expected structure: root/train/<class>/*.JPEG"
        )

    if not val_dir.exists():
        raise FileNotFoundError(
            f"Validation directory not found: {val_dir}\n"
            "Expected structure: root/val/<class>/*.JPEG"
        )

    # Standard ImageNet normalization
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]
    )

    train_dataset = datasets.ImageFolder(str(train_dir), transform=train_transform)

    try:
        val_dataset = datasets.ImageFolder(str(val_dir), transform=val_transform)
    except FileNotFoundError:
        print("[info] Using unlabeled validation images.")
        val_dataset = UnlabeledImageDataset(str(val_dir), transform=val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        drop_last=False,
    )

    print(
        f"[ImageNet extracted] train={len(train_dataset)} images, val={len(val_dataset)} images"
    )
    return train_loader, val_loader
