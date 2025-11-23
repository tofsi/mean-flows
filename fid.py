# fid.py — fixed version
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
import numpy as np
import jax.numpy as jnp


# ------------------------------------------------------------
# 1) Inception-V3 backbone (pool3 / avg-pool features = 2048-D)
# ------------------------------------------------------------
def make_inception():
    model = InceptionV3(include_top=False, weights="imagenet", pooling="avg")
    model.trainable = False
    return model


# ------------------------------------------------------------
# 2) Helper: torch BCHW -> numpy BHWC float32
# ------------------------------------------------------------
def torch_images_to_numpy_npwc(images_torch):
    """
    images_torch: (B,3,H,W) float in [0,1]
    returns: (B,H,W,3) float32
    """
    images_np = images_torch.detach().cpu().numpy()
    # FIX: transpose needs axes tuple
    images_np = np.transpose(images_np, (0, 2, 3, 1))
    return images_np.astype(np.float32)


# ------------------------------------------------------------
# 3) Preprocess for Inception
# ------------------------------------------------------------
def preprocess_for_inception_tf(images_uint8_or_float):
    """
    images_uint8_or_float:
        (B,H,W,3) uint8 in [0,255] or float in [0,1] / [0,255]
    returns:
        (B,299,299,3) float32 preprocessed for InceptionV3
    """
    images = tf.convert_to_tensor(images_uint8_or_float, tf.float32)

    # Basic sanity checks (cheap)
    tf.debugging.assert_rank(images, 4, message="Images must be BHWC")
    tf.debugging.assert_equal(
        tf.shape(images)[-1], 3, message="Images must have 3 channels"
    )

    # If in [0,1], scale to [0,255]
    if tf.reduce_max(images) <= 1.5:
        images = images * 255.0

    images = tf.image.resize(images, (299, 299), method="bilinear")
    images = preprocess_input(images)

    # FIX: return processed images
    return images


def inception_features_tf(model, images):
    """
    model: InceptionV3 from make_inception()
    images: (B,H,W,3) numpy or tf.Tensor
    returns: (B,2048) numpy float32 features
    """
    images_pp = preprocess_for_inception_tf(images)
    feats = model(images_pp, training=False)
    return feats


# ------------------------------------------------------------
# 4) Feature extraction over an iterator
# ------------------------------------------------------------
def extract_features_in_batches(model, images_iter, batch_size=128):
    """
    images_iter yields either:
      - batches (B,H,W,3), OR
      - single images (H,W,3)
    Returns:
      (N,2048) numpy array
    """
    feats = []
    buffer = []

    for item in images_iter:
        arr = np.asarray(item)

        # If single image, buffer until batch_size
        if arr.ndim == 3:
            buffer.append(arr)
            if len(buffer) < batch_size:
                continue
            batch = np.stack(buffer, axis=0)
            buffer = []
        else:
            batch = arr  # already a batch

        f = inception_features_tf(model, batch)
        feats.append(np.asarray(f))

    # flush remainder if any
    if buffer:
        batch = np.stack(buffer, axis=0)
        f = inception_features_tf(model, batch)
        feats.append(np.asarray(f))

    return np.concatenate(feats, axis=0)


# ------------------------------------------------------------
# 5) FID computation in JAX
# ------------------------------------------------------------
def compute_stats(features_np):
    x = jnp.asarray(features_np)
    mu = jnp.mean(x, axis=0)
    xc = x - mu
    cov = (xc.T @ xc) / (x.shape[0] - 1)
    return mu, cov


def sqrtm_psd(A, eps=1e-10):
    A = (A + A.T) / 2.0
    w, V = jnp.linalg.eigh(A)
    w = jnp.clip(w, a_min=0.0)
    return (V * jnp.sqrt(w + eps)) @ V.T


def fid_from_stats(mu, cov, mu_w, cov_w):
    diff = mu - mu_w
    diff_sq = diff @ diff
    cov_sqrt = sqrtm_psd(cov @ cov_w)
    return diff_sq + jnp.trace(cov + cov_w - 2.0 * cov_sqrt)
