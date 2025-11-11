# --- imports (same as yours) ---
import os
import json
import math
import numpy as np
from scipy import spatial

import jax
import jax.numpy as jnp
from jax import random
from flax import nnx

T = jnp.float32  # global dtype


def metric(error):
    # Square loss, equation (9)
    return jnp.sum(jnp.square(error))


# ===== Batched helpers =====


def sample_t_r(key, batch_size, dtype=T):
    """Return per-sample r < t, shapes: r (B,), t (B,)."""
    u = random.uniform(key, shape=(batch_size, 2), dtype=dtype, minval=0.0, maxval=1.0)
    u_sorted = jnp.sort(u, axis=1)
    r = u_sorted[:, 0]
    t = u_sorted[:, 1]
    return r, t


def p_0(key, batch_size, dim, dtype=T):
    """Batch N(0, I). Shape: (B, dim)."""
    # Equivalent to multivariate_normal with mean=0, cov=I, but faster for batches
    return random.normal(key, shape=(batch_size, dim), dtype=dtype)


# ---- Algorithm 1 (training) ----
# fn is a Python callable; mark it static so JIT doesn't try to trace it as data.

# ===== Batched Algorithm 1: returns mean loss and per-sample losses =====


def algorithm_1(fn, x, key):
    """
    Batched Algorithm 1 loss.
    Args:
      fn: callable, fn(z, r, t) -> u(z, r, t), accepts batched (B,*) and returns (B, D)
      x : (B, D) batch of data points
      key: PRNGKey
    Returns:
      mean_loss: scalar
    """
    B, D = x.shape
    k_rt, k_e = random.split(key, 2)

    # Sample r,t,e for the entire batch
    r, t = sample_t_r(k_rt, B, dtype=T)  # (B,), (B,)
    e = p_0(k_e, B, D, dtype=T)  # (B, D)

    # Conditional path and direction
    z = (1.0 - t)[:, None] * x + t[:, None] * e  # (B, D)
    v = e - x  # (B, D)

    # JVP wrt t for the whole batch in one go:
    # primals: (z, r, t), tangents: (0, 0, 1)
    zeros_z = jnp.zeros_like(z)
    zeros_r = jnp.zeros_like(r)
    ones_t = jnp.ones_like(t)

    u, dudt = jax.jvp(fn, (z, r, t), (zeros_z, zeros_r, ones_t))  # each (B, D)

    # Target and error
    u_tgt = v - (t - r)[:, None] * dudt  # (B, D)
    error = u - jax.lax.stop_gradient(u_tgt)  # (B, D)

    # Loss per sample then mean
    losses = jnp.sum(error * error, axis=1)  # (B,)
    mean_loss = jnp.mean(losses)  # scalar
    return mean_loss


# JIT with fn static by position 0 (broad compatibility)
algorithm_1_jit = jax.jit(algorithm_1, static_argnums=0)
# ===== Batched Algorithm 2 (sampling) =====


def algorithm_2(fn, dim, key, batch_size):
    """
    One-step sampling for a minibatch.
    Args:
      fn: callable, fn(z, r, t) -> u(z, r, t), batched
      dim: data dimension (e.g., 2)
      key: PRNGKey
      batch_size: number of samples to generate
    Returns:
      x: (B, dim)
    """
    e = p_0(key, batch_size, dim, dtype=T)  # (B, dim)

    # r,t can be scalars or (B,)—your MLP broadcasts fine. Use scalars for simplicity:
    r = jnp.array(0.0, dtype=T)
    t = jnp.array(1.0, dtype=T)

    u = fn(e, r, t)  # (B, dim)  <-- batched call
    x = e - u
    return x


algorithm_2_jit = jax.jit(algorithm_2, static_argnums=(0, 1, 3))
