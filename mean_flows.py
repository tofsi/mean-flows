# --- imports (same as yours) ---
import os
import json
import math
import numpy as np
from scipy import spatial
from typing import Optional, Tuple
from functools import partial

import jax
import jax.numpy as jnp
from jax import random
from flax import nnx

T = jnp.float32  # global dtype
_c = T(1e-6)  # Used for avoiding division by zero in loss computation.
_drop_probability = T(0.1)


@partial(jax.jit, static_argnums=(1,))
def metric(error, p):
    # Weighted square loss, see appendix B.2 Loss Metrics
    B = error.shape[0]
    error_flat = error.reshape(B, -1) 
    losses = jnp.mean(jnp.square(error_flat), axis=1)
    weights = jax.lax.stop_gradient(jnp.pow(losses + _c, -p))
    # Normalize weights to preserve scale
    weights = weights / jnp.mean(weights)
    return jnp.mean(weights * losses)


# ===== Batched helpers =====


@partial(jax.jit, static_argnums=(1, 2, 3, 4, 5))
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
    key_sampler, key_ratio_of_sampling = random.split(key, 2)

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


@partial(jax.jit, static_argnums=(1, 2, 3))
def p_0(key, batch_size, dim, dtype=T):
    """Batch N(0, I). Shape: (B, dim)."""
    # Equivalent to multivariate_normal with mean=0, cov=I, but faster for batches
    return random.normal(key, shape=(batch_size, dim), dtype=dtype)


# ---- Algorithm 1 (training) ----
# fn is a Python callable; mark it static so JIT doesn't try to trace it as data.

# ===== Batched Algorithm 1: returns mean loss and per-sample losses =====


@partial(jax.jit, static_argnums=(0, 5, 6, 7, 8, 9, 10, 11))
def algorithm_1(
    fn,
    params,
    x,
    c,
    key,
    ratio_of_sampling: float,
    distribution: str,
    sampler_args: Optional[Tuple[float, float]],
    p,
    omega: Optional[float],
    embed_t_r=lambda t, r: (t, r),
    jvp_computation_option=(False, True),
):
    """
    Batched Algorithm 1 loss.
    Args:
      fn: callable, fn(params, z, r, t) -> u(params, z, r, t), accepts batched (B,*) and returns (B, D)
      params: Parameters of the parametrized function fn for the field
      x : (B, D) batch of data points
      c : (B, K) batch of (I assume one-hot encoded, can change this) classes for class-conditioning, can set to None if classes are not used
      omega : Coefficient for classifier-free guidance. Set to None if not used
      key: PRNGKey
      ratio_of_sampling: float between 0 and 1, proportion of samples where r!=t
      distribution: str, either "uniform" or "lognorm". For the t, r sampling scheme.
      sampler_args: If None, uniform(0, 1) sampling. Elseif (a, b), logit-normal(a, b)
      p: Loss weighting coefficient (see appendix B.2 in the paper for details)
      positional_embedding : function, Choice of positional embedding of t and r
      omega : float between 0 and 1, degree of CFG. If none, classifier-free guidance is not applied (see section 4.2, Mean Flows with Guidance)
    Returns:
      mean_loss: scalar
      embed_t_r : function from t, r to some tuple of functions of t, r, specifying the positional embedding. See table 1c for examples.
      jvp_computation : (bool, bool) specifying arguments for jax.jvp. The default argument corresponds to the true value. See table 1b for other (incorrect) examples.
    """
    B = x.shape[0]
    x_flat = x.reshape(B, -1)  # (B, D)
    D = x_flat.shape[1]
    k_rt, k_e, k_cfg = random.split(key, 3)

    # Sample r,t,e for the entire batch
    t, r = sample_t_r(
        k_rt, B, ratio_of_sampling, distribution, sampler_args, dtype=T
    )  # (B,), (B,)
    e = p_0(k_e, B, D, dtype=T)  # (B, D)

    # Conditional path and direction
    z = (1.0 - t)[:, None] * x_flat + t[:, None] * e  # (B, D)
    v = e - x_flat  # (B, D)

    # JVP wrt t for the whole batch in one go:
    # Here begins a slightly annoying part.
    # We need to specify whether the network is class conditional or not
    # This is done by checking omega == None (by convention)
    # If class conditional networks are used, we must add c to our arguments.
    # There are probably better ways of doing this idk
    class_conditional = omega != None
    if not class_conditional:

        def fn_z_r_t(z_, r_, t_):
            return fn(
                {"params": params}, z_.reshape(x.shape), embed_t_r(t_, r_)
            ).reshape(B, -1)

        # Target and error
        # Compute tangents with possibly incorrect jvp computation
        u, dudt = jax.jvp(
            fn_z_r_t,
            (z, r, t),
            (
                v,
                jnp.ones_like(r) if jvp_computation_option[0] else jnp.zeros_like(r), #jvp_computation_option[0] + jnp.zeros_like(r),
                jnp.ones_like(t) if jvp_computation_option[1] else jnp.zeros_like(t), #jvp_computation_option[1] + jnp.zeros_like(t),
            ),
        )
        u_tgt = v - (t - r)[:, None] * dudt  # (B, D)
        error = u - jax.lax.stop_gradient(u_tgt)  # (B, D)
        return metric(error, p)
    else:
        drop_mask = random.bernoulli(k_cfg, p=_drop_probability, shape=(t.shape[0],))
        c_some_unconditional = jnp.where(
            drop_mask, 0, c
        )  # We use the convention c == 0 means no class.

        def fn_z_r_t(z_, r_, t_):
            return fn(
                {"params": params},
                z_.reshape(x.shape),
                embed_t_r(t_, r_),
                c_some_unconditional,
            ).reshape(B, -1)

        if omega == 1.0:
            v_tilde = v
        else:
            u_diag = fn_z_r_t(z, r, t)
            v_tilde = omega * v + (1.0 - omega) * u_diag
        u, dudt = jax.jvp(
            fn_z_r_t,
            (z, t, t),
            (
                v_tilde,
                jnp.ones_like(r) if jvp_computation_option[0] else jnp.zeros_like(r), #jvp_computation_option[0] + jnp.zeros_like(r),
                jnp.ones_like(t) if jvp_computation_option[1] else jnp.zeros_like(t), #jvp_computation_option[1] + jnp.zeros_like(t),
            ),
        )

        # Target and error
        u_tgt = v_tilde - (t - r)[:, None] * dudt  # (B, D)
        error = u - jax.lax.stop_gradient(u_tgt)  # (B, D)

        # Loss per sample then mean
        return metric(error, p)


# JIT with fn static by position 0 (broad compatibility)
# ===== Batched Algorithm 2 (sampling) =====


@partial(jax.jit, static_argnums=(0, 1, 3, 4, 5))
def algorithm_2(
    fn, dim, key, batch_size, embed_t_r=lambda t, r: (t, r), n_steps=1, c=None
):
    """
    Multi-step sampling (Euler integrator) for the mean flow model.

    Args:
        fn: callable,    fn(z, r, t) -> u(z, r, t), batched
        dim: int         data dimension
        key: PRNGKey
        batch_size: int
        n_steps: int     number of Euler steps (1 = your original algorithm_2)
        c : optional class. Note, for a class conditional model, the correct option for marginal sampling is *not* c = None but rather c = jnp.zeros(n_classes)

    Returns:
        x: (B, dim) final generated samples
    """
    # initial sample from base distribution p_0
    x = p_0(key, batch_size, dim, dtype=T)  # (B, dim)

    # time grid: t_0 = 1, t_n = 0
    # shape: (n_steps+1,)
    t_grid = jnp.linspace(1.0, 0.0, n_steps + 1, dtype=T)

    # step function for lax.scan
    def step(x, k):
        t_prev = t_grid[k]  # t_k
        t_next = t_grid[k + 1]  # t_{k+1}
        r = jnp.full((batch_size,), t_next)  # r = next time
        t = jnp.full((batch_size,), t_prev)  # current time

        dt = t - r  # positive step length
        u = (
            fn(x, embed_t_r(t, r), c) if c is not None else fn(x, embed_t_r(t, r))
        )  # (B, dim) vector field
        x_new = x - dt[:, None] * u  # Euler step

        return x_new, None

    # indices 0 .. n_steps-1
    x_final, _ = jax.lax.scan(step, x, jnp.arange(n_steps))

    return x_final
