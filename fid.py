# We need the inceptionV3 model.
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
import numpy as np
import jax.numpy as jnp
import jax


# IMPORT INCEPTION V3 MODEL
def make_inception():
    # 2048-d avg pooled features
    model = InceptionV3(include_top=False, weights="imagenet", pooling="avg")
    model.trainable = False
    return model


# PREPROCESSING AND FEATURE EXTRACTION
def torch_images_to_numpy_npwc(images_torch):
    """
    images_torch: (B,3,256,256) float in [0,1]
    returns: (B,256,256,3) float32 in [0,255] or [0,1] (either ok)
    """
    images_np = images_torch.detach().cpu().numpy()
    images_np = np.transpose(images_np, 0, 2, 3, 1)
    return images_np.astype(np.float32)


def preprocess_for_inception_tf(images_uint8_or_float):
    # Images: should be (B, H, W, 3) values in [0, 255] uint8 or [0,1]/[0,255] float
    images = tf.convert_to_tensor(images_uint8_or_float, tf.float32)
    if tf.reduce_max(images) <= 1.5:
        images = images * 255.0
    images = tf.image.resize(images, (299, 299), method="bilinear")
    images = preprocess_input(images)


@tf.function
def inception_features_tf(model, images):
    images_pp = preprocess_for_inception_tf(images)
    return model(images_pp, training=False)


def extract_features_in_batches(model, images_iter, batch_size=128):
    feats = []
    for batch in images_iter:
        f = inception_features_tf(model, batch)
        feats.append(f.numpy())
    return np.concatenate(feats, axis=0)


# COMPUTE FID
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
