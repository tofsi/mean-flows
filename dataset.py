import os
import csv
from dataclasses import dataclass
from typing import List, Tuple, Dict, Iterator, Optional

import numpy as np
from PIL import Image

import jax
import jax.numpy as jnp



###### Uses kaggle dataset ##############################
##############################################################
#  Helpers to read synset mapping and labels             #####
##############################################################

def load_synset_mapping() -> Tuple[Dict[str, int], List[str]]:
    """
    LOC_synset_mapping.txt lines look like:
        n01440764 tench, Tinca tinca
        n01443537 goldfish, Carassius auratus
    We map synset IDs (e.g. 'n01440764') to integer class indices [0..999].
    Returns:
        syn_to_idx: Dict mapping synset ID to class index
        idx_to_syn: List mapping class index to synset ID
    """
    syn_to_idx: Dict[str, int] = {}
    idx_to_syn: List[str] = []

    mapping_path = "/kaggle/input/imagenet-object-localization-challenge/LOC_synset_mapping.txt"

    with open(mapping_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            synset, *_ = line.split(" ")
            if synset not in syn_to_idx:
                syn_to_idx[synset] = len(idx_to_syn)
                idx_to_syn.append(synset)

    return syn_to_idx, idx_to_syn


def load_val_labels() -> Dict[str, str]:
    """
    LOC_val_solution.csv has two columns:
        ImageId, PredictionString - 
        the id of the val image, for example ILSVRC2012_val_00048981
    PredictionString for detection looks like:
        "n01978287 240 170 260 240"
    For classification we take the FIRST synset in PredictionString.
    We return: { "ILSVRC2012_val_00048981": "n01978287", ... }
    """
    imgid_to_syn: Dict[str, str] = {}
    labels_path = "/kaggle/input/imagenet-object-localization-challenge/LOC_val_solution.csv"
    with open(labels_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_id = row["ImageId"]
            pred_str = row["PredictionString"]
            if not pred_str:
                continue
            first_syn = pred_str.split()[0]
            imgid_to_syn[image_id] = first_syn
    return imgid_to_syn


##############################################################
##                Dataset class                          #####
##############################################################

@dataclass
class ImageNetDataset:
    """
    Simple ImageNet classification dataset over a folder of images.

    Each sample is (image, label_index):
      - image: float32 array [H, W, 3] normalized to [0, 1] (or standardized)
      - label_index: int in [0, num_classes)
    """
    mode: str                 # train or val or test mode

    # Filled in after init:
    paths: List[str] = None
    labels: np.ndarray = None
    syn_to_idx: Dict[str, int] = None
    idx_to_syn: List[str] = None
    root_dir: str = "/kaggle/input/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/"        # path to image folder

    def __post_init__(self):
        # 1) Load mapping synset -> class index
        self.syn_to_idx, self.idx_to_syn = load_synset_mapping()

        if self.mode == "val":
        # 2) Load val labels (image_id -> synset)
            imgid_to_syn = load_val_labels()

        # 3) Build list of (image_path, class_idx)
        paths: List[str] = []
        labels: List[int] = []
        mode_dir = os.path.join(self.root_dir, self.mode)

        if self.mode == "val" or self.mode == "test":
            for fname in sorted(os.listdir(mode_dir)):
                if not fname.lower().endswith((".jpeg", ".jpg", ".png")):
                    continue
                img_id = os.path.splitext(fname)[0]  # "ILSVRC2012_val_00000001"
                if img_id not in imgid_to_syn:
                    # Some files might not be in the CSV; skip them
                    continue
                syn = imgid_to_syn[img_id]
                if syn not in self.syn_to_idx:
                    continue
                cls_idx = self.syn_to_idx[syn]

                paths.append(os.path.join(mode_dir, fname))
                labels.append(cls_idx)

            self.paths = paths
            self.labels = np.array(labels, dtype=np.int32)
            print(f"[ImageNetDataset] Loaded {len(self.paths)} images from {mode_dir}")
        
        elif self.mode == "train":  # train mode
            for syn in sorted(os.listdir(mode_dir)):
                syn_dir = os.path.join(mode_dir, syn)
                if not os.path.isdir(syn_dir):
                    continue
                if syn not in self.syn_to_idx:
                    continue
                cls_idx = self.syn_to_idx[syn]

                for fname in sorted(os.listdir(syn_dir)):
                    if not fname.lower().endswith((".jpeg", ".jpg", ".png")):
                        continue
                    img_path = os.path.join(syn_dir, fname)
                    paths.append(img_path)
                    labels.append(cls_idx)
            self.paths = paths
            self.labels = np.array(labels, dtype=np.int32)
            print(f"[ImageNetDataset] Loaded {len(self.paths)} images from {mode_dir}")

    def __len__(self) -> int:
        return len(self.paths)

    def _load_image(self, path: str) -> np.ndarray:
        """Load image from disk and resize to [image_size, image_size]."""
        with Image.open(path) as img:
            img = img.convert("RGB")
            img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
            arr = np.array(img, dtype=np.float32) / 255.0  # [H, W, 3] in [0, 1]

        # apply ImageNet mean/std normalization
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        arr = (arr - mean) / std

        return arr

    def get_example(self, idx: int) -> Tuple[np.ndarray, int]:
        img_path = self.paths[idx]
        img = self._load_image(img_path)
        label = int(self.labels[idx])
        return img, label


##############################################################
#                    DataLoader                          #####  
##############################################################

def imagenet_data_loader(
    dataset: ImageNetDataset,
    batch_size: int,
    rng: jax.Array,
    shuffle: bool = True,
) -> Iterator[Tuple[jax.Array, jax.Array]]:
    """
    Simple generator that yields JAX arrays:
      images: [B, H, W, 3], float32
      labels: [B], int32
    """
    num_samples = len(dataset)
    indices = jnp.arange(num_samples)

    if shuffle:
        indices = jax.random.permutation(rng, indices)
        indices = np.array(indices)  # back to NumPy for indexing

    else:
        indices = np.arange(num_samples)

    for start in range(0, num_samples, batch_size):
        end = min(start + batch_size, num_samples)
        batch_idx = indices[start:end]

        imgs = []
        labels = []
        for i in batch_idx:
            img, lbl = dataset.get_example(int(i))
            imgs.append(img)
            labels.append(lbl)

        batch_imgs = jnp.asarray(np.stack(imgs, axis=0))   # [B, H, W, 3]
        batch_labels = jnp.asarray(np.array(labels, dtype=np.int32))  # [B]

        yield batch_imgs, batch_labels

##############################################################
#                    Main block                          ##### 
##############################################################


def TrainLoader():
    train_dataset = ImageNetDataset(mode="train")
    rng = jax.random.PRNGKey(42)
    train_loader = imagenet_data_loader(train_dataset, batch_size=64, rng=rng, shuffle=True)
    # test the loader
    for batch_imgs, batch_labels in train_loader:
        print("Batch images:", batch_imgs.shape, batch_imgs.dtype)
        print("Batch labels:", batch_labels.shape, batch_labels.dtype)
        break

    return train_loader


def ValLoader():
    val_dataset = ImageNetDataset(mode="val")
    rng = jax.random.PRNGKey(0)
    val_loader = imagenet_data_loader(val_dataset, batch_size=64, rng=rng, shuffle=False)
    # test the loader
    for batch_imgs, batch_labels in val_loader:
        print("Batch images:", batch_imgs.shape, batch_imgs.dtype)
        print("Batch labels:", batch_labels.shape, batch_labels.dtype)
        break

    return val_loader


def TestLoader():
    test_dataset = ImageNetDataset(mode="test")
    rng = jax.random.PRNGKey(1)
    test_loader = imagenet_data_loader(test_dataset, batch_size=64, rng=rng, shuffle=False)
    # test the loader
    for batch_imgs, batch_labels in test_loader:
        print("Batch images:", batch_imgs.shape, batch_imgs.dtype)
        print("Batch labels:", batch_labels.shape, batch_labels.dtype)
        break

    return test_loader





