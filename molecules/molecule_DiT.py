import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
import numpy as np

import os, sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(THIS_DIR)

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from DiT_model import get_1d_sincos_pos_embed, AdaLNZero, modulate, DiTBlock, TimeEmbed

########################################################
#         Molecule DiT (label-free, 56-d latent)    ####
########################################################


class PositionalEmbed1D(nn.Module):
    """
    Fixed 1D sinusoidal positional embeddings for a short token sequence.
    """

    num_tokens: int
    hidden_dim: int

    @nn.compact
    def __call__(self, x):
        """
        x: (B, L, D) where L = num_tokens, D = hidden_dim
        """
        assert x.shape[1] == self.num_tokens, "Token length mismatch"
        pos = np.arange(self.num_tokens, dtype=np.float32)
        pos_embed = get_1d_sincos_pos_embed(self.hidden_dim, pos)  # (L, D)
        pos_embed = jnp.array(pos_embed, dtype=x.dtype)[None, :, :]  # (1, L, D)
        return x + pos_embed


class MoleculeEmbed(nn.Module):
    """
    Embed a JTNNVAE latent vector into a short sequence of tokens.

    We assume: latent_dim = n_tokens * token_latent_dim.
    For JTNNVAE: latent_dim=56, n_tokens=2 -> token_latent_dim=28.
    """

    latent_dim: int  # e.g. 56
    hidden_dim: int
    n_tokens: int = 2  # tree + graph

    @nn.compact
    def __call__(self, z):
        """
        z: (B, latent_dim)
        returns: (B, n_tokens, hidden_dim)
        """
        B, D = z.shape
        assert D == self.latent_dim, f"Expected latent_dim={self.latent_dim}, got {D}"

        token_dim = self.latent_dim // self.n_tokens
        assert (
            token_dim * self.n_tokens == self.latent_dim
        ), "latent_dim must be divisible by n_tokens"

        # Split into tokens: [tree | graph]
        z_tokens = z.reshape(B, self.n_tokens, token_dim)  # (B, n_tokens, token_dim)

        proj = nn.Dense(
            features=self.hidden_dim,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.zeros,
            name="proj",
        )
        x = proj(z_tokens)  # (B, n_tokens, hidden_dim)
        return x


class FinalVectorLayer(nn.Module):
    """
    Final layer to map token hidden states back to per-token latent vectors.
    """

    hidden_dim: int
    token_latent_dim: int  # per-token output dim (e.g. 28 if latent_dim=56, n_tokens=2)

    @nn.compact
    def __call__(self, x, cond):
        """
        x: (B, L, hidden_dim)
        cond: (B, hidden_dim)
        returns: (B, L, token_latent_dim)
        """
        adaLN_final = AdaLNZero(self.hidden_dim, 2)(cond)
        shift, scale = jnp.split(adaLN_final, 2, axis=-1)
        shift = shift[:, None, :]  # (B,1,D)
        scale = scale[:, None, :]  # (B,1,D)

        norm_final = nn.LayerNorm(self.hidden_dim)(x)
        x_mod = modulate(norm_final, shift, scale)

        x_out = nn.Dense(
            features=self.token_latent_dim,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
            name="proj_out",
        )(
            x_mod
        )  # (B, L, token_latent_dim)

        return x_out


class MoleculeDiT(nn.Module):
    """
    DiT-style transformer operating directly in JTNNVAE latent space.

    Label-free: conditioning is only on time; `y` is accepted for API
    compatibility but is completely ignored.

    Input: z of shape (B, latent_dim=56).
    Output: tensor of same shape, used by flow-matching.
    """

    depth: int
    hidden_dim: int
    num_heads: int
    latent_dim: int  # total JTNNVAE latent dimension, here 56
    n_tokens: int = 2  # tree + graph
    num_classes: int = 1  # unused, kept for compatibility with algorithm_1

    @nn.compact
    def __call__(self, z, tr, y, train: bool = True):
        """
        z:  (B, latent_dim)   JTNNVAE latent vector
        tr: tuple of time-like scalars (e.g. (t, r)), each shape (B,)
        y:  (B,) dummy tensor; IGNORED
        """
        del y  # explicitly ignore labels

        B, D = z.shape
        assert D == self.latent_dim, f"Expected latent_dim={self.latent_dim}, got {D}"

        # 1. Embed latent into 2 tokens: tree and graph
        x_tok = MoleculeEmbed(
            latent_dim=self.latent_dim,
            hidden_dim=self.hidden_dim,
            n_tokens=self.n_tokens,
            name="mol_embed",
        )(
            z
        )  # (B, L, hidden_dim)

        # 2. Add 1D positional embeddings
        x_tok = PositionalEmbed1D(
            num_tokens=self.n_tokens,
            hidden_dim=self.hidden_dim,
            name="pos_embed_1d",
        )(
            x_tok
        )  # (B, L, hidden_dim)

        # 3. Time embeddings: same pattern as image DiT
        time_cond = 0.0
        for i, ti in enumerate(tr):
            time_cond = time_cond + TimeEmbed(self.hidden_dim, name=f"time_embed_{i}")(
                ti
            )  # (B, hidden_dim)

        cond = time_cond  # label-free conditioning

        # 4. Transformer blocks
        x = x_tok
        for i in range(self.depth):
            x = DiTBlock(self.hidden_dim, self.num_heads, name=f"block_{i}")(x, cond)

        # 5. Map tokens back to latent space
        token_latent_dim = self.latent_dim // self.n_tokens  # 28
        x_out_tok = FinalVectorLayer(
            hidden_dim=self.hidden_dim,
            token_latent_dim=token_latent_dim,
            name="final_vector_layer",
        )(
            x, cond
        )  # (B, L, token_latent_dim)

        # 6. Flatten tokens back to (B, latent_dim) = (B, 56)
        z_out = x_out_tok.reshape(B, self.latent_dim)
        return z_out

    # keep same API as image DiT
    def forward(self, x, tr, y, train: bool = True):
        return self.__call__(x, tr, y, train=train)


def MoleculeDiT_B():
    """
    "Base" size Molecule DiT for 56-dim JTNNVAE latents.
    """
    return MoleculeDiT(
        depth=8,
        hidden_dim=512,
        num_heads=8,
        latent_dim=56,
        n_tokens=2,
        num_classes=1,  # unused
    )
