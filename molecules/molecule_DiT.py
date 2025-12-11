import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
import numpy as npimport
import numpy as np
import sys

sys.path.append('..') 

from DiT_model import get_1d_sincos_pos_embed, AdaLNZero, TimeEmbed, LabelEmbed, modulate, DiTBlock

#  ---- 1D positional embedding for tokens ----

class PositionalEmbed1D(nn.Module):
    """
    Fixed 1D sinusoidal positional embeddings for a sequence of length `num_tokens`.
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


# ---- embedding of JT-VAE latent to transformer tokens ----

class MoleculeEmbed(nn.Module):
    """
    Embed a JT-VAE latent vector into a short sequence of tokens.

    We assume: latent_dim = n_tokens * token_latent_dim.
    For JT-VAE: n_tokens=2 (tree + graph), token_latent_dim=latent_dim//2.
    Each token is linearly projected to hidden_dim.
    """

    latent_dim: int       # e.g. 256
    hidden_dim: int
    n_tokens: int = 2     # (# of tokens: tree and graph)

    @nn.compact
    def __call__(self, z):
        """
        z: (B, latent_dim)
        returns: (B, n_tokens, hidden_dim)
        """
        B, D = z.shape
        assert D == self.latent_dim, "Unexpected latent_dim"

        token_dim = self.latent_dim // self.n_tokens
        assert token_dim * self.n_tokens == self.latent_dim, "latent_dim must be divisible by n_tokens"

        # Split into tokens along the feature dimension
        z_tokens = z.reshape(B, self.n_tokens, token_dim)  # (B, n_tokens, token_dim)

        # Linear projection for each token -> hidden_dim
        proj = nn.Dense(
            features=self.hidden_dim,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.zeros,
            name="proj",
        )
        x = proj(z_tokens)  # (B, n_tokens, hidden_dim)

        return x


# ---- NEW: final layer for vector output ----

class FinalVectorLayer(nn.Module):
    """
    Final layer to map token hidden states back to per-token latent vectors.
    """

    hidden_dim: int
    token_latent_dim: int  # per-token output dim (e.g., 128 if latent_dim=256, n_tokens=2)

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
        )(x_mod)  # (B, L, token_latent_dim)

        return x_out


# ---- NEW: DiT variant for molecule latents ----

class MoleculeDiT(nn.Module):
    """
    DiT-style transformer that operates directly in JT-VAE latent space.

    - Input: JT-VAE mean latent z_mean of shape (B, latent_dim).
    - Output: tensor of same shape, used in flow-matching to define velocity / score.
    """

    depth: int
    hidden_dim: int
    num_heads: int
    latent_dim: int           # total JT-VAE latent dimension, e.g. 256
    n_tokens: int = 2         # tree + graph
    num_classes: int = 1      # can keep 1 and feed y=0 if unconditional

    @nn.compact
    def __call__(self, z, tr, y, train: bool = True):
        """
        z:  (B, latent_dim)   JT-VAE latent vector
        tr: tuple of time embeddings, e.g. (t, r), each (B,)
        y:  (B,) integer labels or dummy class (e.g., all zeros)
        """
        B, D = z.shape
        assert D == self.latent_dim, "Unexpected latent_dim"

        # 1. Embed latent into tokens
        x_tok = MoleculeEmbed(
            latent_dim=self.latent_dim,
            hidden_dim=self.hidden_dim,
            n_tokens=self.n_tokens,
            name="mol_embed",
        )(z)  # (B, L, hidden_dim), L=n_tokens

        # 2. 1D positional embedding
        x_tok = PositionalEmbed1D(
            num_tokens=self.n_tokens,
            hidden_dim=self.hidden_dim,
            name="pos_embed_1d",
        )(x_tok)  # (B, L, hidden_dim)

        # 3. Time embeddings (variable-length as in your DiT)
        time_cond = 0.0
        for i, ti in enumerate(tr):
            time_cond = time_cond + TimeEmbed(self.hidden_dim, name=f"time_embed_{i}")(
                ti
            )  # (B, hidden_dim)

        # 4. Label embedding (CFG-ready, but you can keep drop_prob=0.0)
        y_emb = LabelEmbed(
            self.hidden_dim,
            self.num_classes,
            drop_prob=0.0,
            name="label_embed",
        )(y, train=train)  # (B, hidden_dim)

        cond = time_cond + y_emb  # (B, hidden_dim)

        # 5. Transformer blocks
        x = x_tok
        for i in range(self.depth):
            x = DiTBlock(self.hidden_dim, self.num_heads, name=f"block_{i}")(x, cond)

        # 6. Final mapping back to latent space
        token_latent_dim = self.latent_dim // self.n_tokens
        x_out_tok = FinalVectorLayer(
            hidden_dim=self.hidden_dim,
            token_latent_dim=token_latent_dim,
            name="final_vector_layer",
        )(x, cond)  # (B, L, token_latent_dim)

        # 7. Flatten tokens back to (B, latent_dim) to match JT-VAE latent shape
        z_out = x_out_tok.reshape(B, self.latent_dim)
        return z_out
