"""
Compute and save real-data ImageNet-val InceptionV3 statistics for FID.

Works with a *flat* validation directory extracted from:
    ILSVRC2012_img_val.tar

Expected layout:
    ./imagenet/downloads/imagenet_val_raw/
        ILSVRC2012_val_00000001.JPEG
        ILSVRC2012_val_00000002.JPEG
        ...
Output:
    imagenet_inception_stats.npz
        - mu_w  (2048,)
        - cov_w (2048,2048)
"""

import os, glob
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import tensorflow as tf
import fid  # your corrected fid.py


# ---------------------------
# CONFIG
# ---------------------------
IMAGENET_VAL_DIR = "./imagenet/downloads/imagenet_val_raw"
BATCH_SIZE = 16  # Change relative to available memory.
NUM_WORKERS = 8
OUT_FILE = "imagenet_inception_stats.npz"


# ---------------------------
# 0) TF GPU setup (optional CUDA)
# ---------------------------
gpus = tf.config.list_physical_devices("GPU")
print("TF GPUs:", gpus)
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


# ---------------------------
# 1) Flat-folder dataset
# ---------------------------
class ImageNetValFlat(Dataset):
    def __init__(self, root, transform=None):
        self.files = sorted(
            glob.glob(os.path.join(root, "*.JPEG"))
            + glob.glob(os.path.join(root, "*.jpg"))
            + glob.glob(os.path.join(root, "*.png"))
        )
        if len(self.files) == 0:
            raise RuntimeError(f"No images found in {root}")
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        img = Image.open(self.files[i]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img


transform = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.ToTensor(),  # BCHW float32 [0,1]
    ]
)

dataset = ImageNetValFlat(IMAGENET_VAL_DIR, transform=transform)
loader = DataLoader(
    dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
)

print(f"Found {len(dataset)} validation images.")


# ---------------------------
# 2) Load inception backbone
# ---------------------------
with tf.device("/GPU:0" if gpus else "/CPU:0"):
    inception = fid.make_inception()


# ---------------------------
# 3) Generator yielding numpy BHWC
# ---------------------------
def real_image_batches():
    for imgs in loader:  # imgs: torch BCHW [0,1]
        imgs_np = imgs.detach().cpu().numpy()
        imgs_np = np.transpose(imgs_np, (0, 2, 3, 1)).astype(np.float32)  # BHWC
        yield imgs_np


# ---------------------------
# 4) Extract features
# ---------------------------
print("Extracting Inception features for ImageNet val...")
with tf.device("/GPU:0" if gpus else "/CPU:0"):
    feats = fid.extract_features_in_batches(inception, real_image_batches())

print("Feature shape:", feats.shape)


# ---------------------------
# 5) Compute stats + save
# ---------------------------
print("Computing mean/cov")
mu_w = feats.mean(axis=0)
xc = feats - mu_w
cov_w = (xc.T @ xc) / (feats.shape[0] - 1)

print(f"Saving to {OUT_FILE} ...")
np.savez(OUT_FILE, mu_w=mu_w.astype(np.float32), cov_w=cov_w.astype(np.float32))
print("Done! Real FID stats ready.")
