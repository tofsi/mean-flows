# --- imports (same as yours) ---
import os
import json
import math
import numpy as np
from scipy import spatial
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from jax import random
from flax import nnx

T = jnp.float32  # global dtype


def metric(error):
    # Square loss, equation (9)
    return jnp.sum(jnp.square(error))


# ===== Batched helpers =====


def sample_t_r(
    key,
    batch_size,
    ratio_of_sampling,
    distribution: str,
    sampler_args: Optional[Tuple[float, float]],
    dtype=T,
):
    """Return per-sample r < t, shapes: r (B,), t (B,)."""
    t, r = dtype(0.0), dtype(0.0)
    u = None
    key_sampler, key_ratio_of_sampling = random.split(key)
    if distribution == "uniform":
        # Uniform sampler
        u = random.uniform(
            key_sampler, shape=(batch_size, 2), dtype=dtype, minval=0.0, maxval=1.0
        )
    elif distribution == "lognorm":
        # Logit normal sampler
        # First samples from N(a, b) and then applies sigmoid to get between 0 and 1
        u = jax.nn.sigmoid(
            sampler_args[0]
            + sampler_args[1]
            * random.normal(key_sampler, shape=(batch_size, 2), dtype=dtype)
        )
    else:
        raise TypeError("Unsupported sampler argument type")
    # Swap order if necessary to enforce r < t
    u_sorted = jnp.sort(u, axis=1)
    # Enforce ratio of sampling constraint
    t_not_equal_r_mask = random.bernoulli(
        key_ratio_of_sampling, ratio_of_sampling, (batch_size)
    )
    t = u_sorted[:, 1]
    r = t_not_equal_r_mask * u_sorted[:, 0] + (~t_not_equal_r_mask) * t
    return t, r


sample_t_r_jit = jax.jit(
    sample_t_r,
    static_argnums=(1, 2, 3, 4, 5),  # batch_size, sampler_kind, dtype
    # sampler_params can be dynamic; or add 4 here if you want them static too
)


def p_0(key, batch_size, dim, dtype=T):
    """Batch N(0, I). Shape: (B, dim)."""
    # Equivalent to multivariate_normal with mean=0, cov=I, but faster for batches
    return random.normal(key, shape=(batch_size, dim), dtype=dtype)


# ---- Algorithm 1 (training) ----
# fn is a Python callable; mark it static so JIT doesn't try to trace it as data.

# ===== Batched Algorithm 1: returns mean loss and per-sample losses =====


def algorithm_1(
    fn,
    params,
    x,
    key,
    ratio_of_sampling,
    distribution: str,
    sampler_args: Optional[Tuple[float, float]],
):
    """
    Batched Algorithm 1 loss.
    Args:
      fn: callable, fn(params, z, r, t) -> u(params, z, r, t), accepts batched (B,*) and returns (B, D)
      params: Parameters of the parametrized function fn for the field
      x : (B, D) batch of data points
      key: PRNGKey
      ratio_of_sampling: float between 0 and 1, proportion of samples where r!=t
      distribution: str, either "uniform" or "lognorm". For the t, r sampling scheme.
      sampler_args: If None, uniform(0, 1) sampling. Elseif (a, b), logit-normal(a, b)
    Returns:
      mean_loss: scalar
    """
    B, D = x.shape
    k_rt, k_e = random.split(key, 2)

    # Sample r,t,e for the entire batch
    t, r = sample_t_r_jit(
        k_rt, B, ratio_of_sampling, distribution, sampler_args, dtype=T
    )  # (B,), (B,)
    e = p_0(k_e, B, D, dtype=T)  # (B, D)

    # Conditional path and direction
    z = (1.0 - t)[:, None] * x + t[:, None] * e  # (B, D)
    v = e - x  # (B, D)

    # JVP wrt t for the whole batch in one go:
    # primals: (z, r, t), tangents: (0, 0, 1)
    zeros_z = jnp.zeros_like(z)
    zeros_r = jnp.zeros_like(r)
    ones_t = jnp.ones_like(t)

    def fn_static_params(z_, r_, t_):
        return fn({"params": params}, z_, r_, t_)

    u, dudt = jax.jvp(
        fn_static_params, (z, r, t), (zeros_z, zeros_r, ones_t)
    )  # each (B, D)

    # Target and error
    u_tgt = v - (t - r)[:, None] * dudt  # (B, D)
    error = u - jax.lax.stop_gradient(u_tgt)  # (B, D)

    # Loss per sample then mean
    losses = jnp.sum(error * error, axis=1)  # (B,)
    mean_loss = jnp.mean(losses)  # scalar
    return mean_loss


# JIT with fn static by position 0 (broad compatibility)
algorithm_1_jit = jax.jit(algorithm_1, static_argnums=(0, 4, 5, 6))
# ===== Batched Algorithm 2 (sampling) =====


def algorithm_2(fn, dim, key, batch_size, n_steps=1):
    """
    Multi-step sampling (Euler integrator) for the mean flow model.

    Args:
        fn: callable,    fn(z, r, t) -> u(z, r, t), batched
        dim: int         data dimension
        key: PRNGKey
        batch_size: int
        n_steps: int     number of Euler steps (1 = your original algorithm_2)

    Returns:
        x: (B, dim) final generated samples
    """
    # initial sample from base distribution p_0
    x = p_0(key, batch_size, dim, dtype=T)   # (B, dim)

    # time grid: t_0 = 1, t_n = 0
    # shape: (n_steps+1,)
    t_grid = jnp.linspace(1.0, 0.0, n_steps + 1, dtype=T)

    # step function for lax.scan
    def step(x, k):
        t_prev = t_grid[k]      # t_k
        t_next = t_grid[k+1]    # t_{k+1}
        r = t_next              # r = next time
        t = t_prev              # current time

        dt = t - r              # positive step length

        u = fn(x, r, t)         # (B, dim) vector field
        x_new = x - dt * u      # Euler step

        return x_new, None

    # indices 0 .. n_steps-1
    x_final, _ = jax.lax.scan(step, x, jnp.arange(n_steps))

    return x_final


algorithm_2_jit = jax.jit(algorithm_2, static_argnums=(0, 1, 3, 4))
