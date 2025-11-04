## Standard libraries
import os
import json
import math
import numpy as np
from scipy import spatial

import jax
import jax.numpy as jnp
import flax
from flax import nnx
from jax import random


T = jnp.float64  # Maybe?

# TODO: Implement mean flow method in a way that can modularly take a network which
# parametrizes the mean flow field and provide the necessary resources for training.


# TODO: Implement algorithm 1
def sample_t_r(key, T):
    # Returns two times from U[0, 1] in ascending order
    return tuple(random.uniform(key, 2, T).sort(axis=0))


def p_0(key, dim, T):
    # Simple prior from which we sample in algorithm 1
    # The following is an example, IDK what they used I'll have to check
    return random.multivariate_normal(key, 0, jnp.identity(dim), dtype=T)


def metric(error):
    # Square loss, equation (9) in the paper
    return jnp.sum(jnp.square(error))


def algorithm_1(fn, x, key):
    """
    Implementation of algorithm 1 from the paper
    Used for training a MeanFlow model

    ## Arguments:
    fn : function
        Parametrized average velocity field u.
        The signature is fn(z, r, t) where
        z = position at t on conditional path
        r = earlier time
        t = later time
        Needs to be autodiffable with jax.jvp!
    x : vector with only one axis!
        Sampled example from our training data
        NOTE: 1d vector only for now, idk final shape or type of x
        TODO: Figure out how to represent x
    key : PRNG? key used for sampling using jax.random

    ## Returns:
    tuple
        error : some kind of vector
            for weight updates of parametrized field fn
            TODO: I think? the error gives the weight update for the square loss (9) in the paper. Is this true?
        loss : float
    """

    r, t = sample_t_r(
        key, T
    )  # I think always r < t so we are parametrizing mean flow from r to t
    e = p_0(key, x.shape[0], T)  # here we assume that x is a vector!
    z = (1 - t) * x + t * e  # Point on conditional path at time t
    v = e - x  # difference vector from starting point e to endpoint x of the path
    u, dudt = jax.jvp(fn, (z, r, t), (v, 0, 1))
    u_tgt = v - (t - r) * dudt
    error = u - jax.lax.stop_gradient(u_tgt)
    loss = metric(error)

    return (error, loss)


# TODO: Implement algorithm 2


def algorithm_2(fn, dim, key):
    """
    Implementation of algorithm 2 from the paper
    1-step sampling of MeanFlow model

    ## Arguments:
    fn : function
        Parametrized average velocity field u.
        The signature is fn(z, r, t) where
        z = position at t on conditional path
        r = earlier time
        t = later time
        Needs to be autodiffable with jax.jvp!
    dim : int
        dimension of generated example
        #TODO: Figure out the representation of this, may not always be vector shape
    key : PRNG? key used for sampling using jax.random

    ## Returns:
    x : Generated sample
    """
    e = p_0(key, dim)
    x = e - fn(e, 0, 1)


# TODO: Test these functions on a simple model fn and a simple data set.
