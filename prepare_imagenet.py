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

from scipy.io import loadmat  # pip install scipy


### Uses main imagenet dataset (ILSVRC2012) #################
# ---------------------------------------------------------
# 1. URLs and basic config
# ---------------------------------------------------------

IMAGENET_URLS = {
    "devkit": "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz",
    "train": "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar",
    "val":   "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar",
    "test":  "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_test.tar",
}


def download_file(url: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"[download] {dest} already exists, skipping.")
        return
    print(f"[download] Downloading {url} → {dest}")
    urllib.request.urlretrieve(url, dest)
    print(f"[download] Done: {dest}")


def extract_tar(src: Path, dest_dir: Path):
    print(f"[extract] Extracting {src} → {dest_dir}")
    dest_dir.mkdir(parents=True, exist_ok=True)
    mode = "r:gz" if src.suffixes[-2:] == [".tar", ".gz"] or src.suffix == ".gz" else "r:"
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


def prepare_val(val_tar: Path, devkit_dir: Path, root: Path):
    """
    The val tar is a flat directory of 50,000 images.
    We use the devkit (meta.mat + ILSVRC2012_validation_ground_truth.txt)
    to reorganize them into root/val/<wnid>/*.JPEG so that ImageFolder can use them.
    """
    val_root = root / "val"
    if val_root.exists() and any(val_root.iterdir()):
        print("[val] Val directory already organized, skipping.")
        return

    # 1) Extract all val images into tmp dir
    tmp_val = root / "val_tmp"
    if not tmp_val.exists() or not any(tmp_val.iterdir()):
        extract_tar(val_tar, tmp_val)

    # 2) Load ground truth labels (1..1000 per image, in order)
    gt_file = devkit_dir / "data" / "ILSVRC2012_validation_ground_truth.txt"
    class_ids = np.loadtxt(gt_file, dtype=int)  # shape (50000,)

    # 3) Load mapping from class_id -> wnid using meta.mat
    meta_mat = devkit_dir / "data" / "meta.mat"
    meta = loadmat(meta_mat, squeeze_me=True)["synsets"]
    # meta is an array of structs; we build dictionary: ilsvrc2012_id -> wnid
    class_id_to_wnid = {}
    for m in meta:
        ilsvrc_id = int(m["ILSVRC2012_ID"])
        wnid = str(m["WNID"])
        class_id_to_wnid[ilsvrc_id] = wnid

    val_root.mkdir(parents=True, exist_ok=True)

    # 4) The official order: sort validation JPEGs by filename
    val_images: List[Path] = sorted(tmp_val.glob("*.JPEG"))

    assert len(val_images) == len(class_ids), \
        f"Mismatch: {len(val_images)} images vs {len(class_ids)} labels."

    print("[val] Organizing validation images into class folders...")
    for img_path, cls_id in zip(val_images, class_ids):
        wnid = class_id_to_wnid[int(cls_id)]
        cls_dir = val_root / wnid
        cls_dir.mkdir(exist_ok=True)
        target_path = cls_dir / img_path.name
        img_path.rename(target_path)

    print("[val] Finished preparing validation set.")
    # Optionally remove tmp_val to save space
    import shutil; shutil.rmtree(tmp_val)


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
            [p for p in self.root.glob("**/*") if p.suffix.lower() in [".jpeg", ".jpg", ".png"]]
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
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Assumes data has already been downloaded and organized into:
      root/train/<wnid>/
      root/val/<wnid>/
      root/test/...
    Builds standard ImageNet transforms + DataLoaders.
    """
    root = Path(root_dir)

    # Standard ImageNet transforms
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    train_dataset = datasets.ImageFolder(root / "train", transform=train_transform)
    val_dataset = datasets.ImageFolder(root / "val", transform=val_test_transform)
    test_dataset = ImageNetTestDataset(root / "test", transform=val_test_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


# ---------------------------------------------------------
# 5. Main: download, prepare, and build loaders
# ---------------------------------------------------------

def main():
    root = Path("./imagenet")       
    downloads = root / "downloads"
    devkit_dir = root / "ILSVRC2012_devkit_t12"

    # 1) Download all files
    devkit_tar = downloads / "ILSVRC2012_devkit_t12.tar.gz"
    train_tar  = downloads / "ILSVRC2012_img_train.tar"
    val_tar    = downloads / "ILSVRC2012_img_val.tar"
    test_tar   = downloads / "ILSVRC2012_img_test.tar"

    download_file(IMAGENET_URLS["devkit"], devkit_tar)
    download_file(IMAGENET_URLS["train"], train_tar)
    download_file(IMAGENET_URLS["val"],   val_tar)
    download_file(IMAGENET_URLS["test"],  test_tar)

    # 2) Extract devkit
    if not devkit_dir.exists():
        extract_tar(devkit_tar, root)

    # 3) Prepare splits
    prepare_train(train_tar, root)
    prepare_val(val_tar, devkit_dir, root)
    prepare_test(test_tar, root)

    # 4) Build dataloaders
    train_loader, val_loader, test_loader = build_dataloaders(
        root_dir=str(root),
        batch_size=256,
        num_workers=8,
    )

    # 5) Quick sanity check: one batch
    print(">>> Checking one batch from each loader...")
    images, labels = next(iter(train_loader))
    print("Train batch:", images.shape, labels.shape)

    images, labels = next(iter(val_loader))
    print("Val batch:", images.shape, labels.shape)

    images, paths = next(iter(test_loader))
    print("Test batch:", images.shape, len(paths), "paths")

    print("All done. Ready to train!")


if __name__ == "__main__":
    main()