import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
import numpy as np


#######################################################
#               Input Processing                  #####
#######################################################


class PatchEmbed(nn.Module):
    """
    Patch embedding layer
    """

    patch_size: int
    in_channels: int
    hidden_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(
            features=self.hidden_dim,
            kernel_size=(self.patch_size, self.patch_size),
            strides=(self.patch_size, self.patch_size),
            padding="VALID",
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.zeros,
            name="proj",
        )(x)
        B, H, W, C = x.shape
        x = x.reshape(B, H * W, C)  # (B, N, hidden_dim) where N = (H/P)*(W/P)
        return x


class LabelEmbed(nn.Module):
    """ "
    Embeds class labels into vectors of size hidden_dim
    and handles dropouts for classifier-free guidance.
    """

    hidden_dim: int
    num_classes: int
    drop_prob: float = 0.1  # probability of dropping labels

    @nn.compact
    def __call__(self, labels, train: bool = True):
        # Embedding layer for class labels
        embed = nn.Embed(
            num_embeddings=self.num_classes + 1,
            features=self.hidden_dim,
            embedding_init=nn.initializers.normal(stddev=0.02),
        )
        if train and self.drop_prob > 0.0:
            # Randomly drop labels for classifier-free guidance
            rng = self.make_rng("dropout")
            mask = jax.random.bernoulli(rng, p=self.drop_prob, shape=labels.shape)
        else:
            mask = jnp.zeros_like(labels, dtype=bool)
        labels = jnp.where(mask, self.num_classes, labels)
        return embed(labels)


def get_1d_sincos_pos_embed(embed_dim, pos):
    """
    Generate 1D sinusoidal positional embeddings
    :param embed_dim: dimension of the embedding (must be even)
    :param pos: positions to embed (M,)
    return: (M, D) positional embeddings
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / (10000**omega)  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)

    emb_sin = np.sin(out)
    emb_cos = np.cos(out)

    return np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)


def get_2d_sincos_pos_embed(embed_dim, grid_size):
    """
    Generate 2D sinusoidal positional embeddings
    :param embed_dim: dimension of the embedding (must be even)
    :param grid_size: size of the grid (assumed square)
    return: (grid_size*grid_size, D) positional embeddings
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)

    # meshgrid: W first for consistency with MAE
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0).reshape(2, grid_size * grid_size)

    # split half the dimensions to encode h, half for w
    emb_h = get_1d_sincos_pos_embed(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed(embed_dim // 2, grid[1])

    return np.concatenate([emb_h, emb_w], axis=1)  # (HW, D)


class PositionalEmbed(nn.Module):
    """
    Fixed 2D sinusoidal positional embeddings
    """

    num_patches: int
    hidden_dim: int

    @nn.compact
    def __call__(self, x):
        pos_embed = get_2d_sincos_pos_embed(
            self.hidden_dim, int(self.num_patches**0.5)
        )  # (N, D)
        pos_embed = jnp.array(pos_embed, dtype=x.dtype)[None, :, :]  # (1, N, D)
        return x + pos_embed


########################################################
#               Core Transformer Block             #####
########################################################


class DiTBlock(nn.Module):
    """
    DiT Block with adaLN conditioning
    """

    hidden_dim: int
    num_heads: int
    mlp_ratio: float = 4.0

    @nn.compact
    def __call__(self, x, cond):
        # adaptive Layer Normalization (adaLN - Zero) for time and context conditioning
        adaLN = AdaLNZero(self.hidden_dim, 6)(cond)
        # get modulation parameters from adaLN
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = jnp.split(
            adaLN, 6, axis=-1
        )
        # broadcast conditioning parameters to match x shape
        shift_msa = shift_msa[:, None, :]
        scale_msa = scale_msa[:, None, :]
        gate_msa = gate_msa[:, None, :]
        shift_mlp = shift_mlp[:, None, :]
        scale_mlp = scale_mlp[:, None, :]
        gate_mlp = gate_mlp[:, None, :]
        # apply attention and MLP with modulation and gating
        norm1 = nn.LayerNorm(self.hidden_dim)(x)
        attn_in = modulate(norm1, shift_msa, scale_msa)
        attn_out = nn.SelfAttention(
            num_heads=self.num_heads,
            qkv_features=self.hidden_dim,
            out_features=self.hidden_dim,
            use_bias=True,
        )(attn_in)
        x = x + gate_msa * attn_out
        norm2 = nn.LayerNorm(self.hidden_dim)(x)
        mlp_in = modulate(norm2, shift_mlp, scale_mlp)
        mlp_out = MLP_with_GELU(self.hidden_dim, self.mlp_ratio)(mlp_in)
        x = x + gate_mlp * mlp_out
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT
    """

    hidden_dim: int
    patch_size: int
    out_channels: int

    @nn.compact
    def __call__(self, x, cond):
        adaLN_final = AdaLNZero(self.hidden_dim, 2)(cond)
        shift, scale = jnp.split(adaLN_final, 2, axis=-1)
        shift = shift[:, None, :]
        scale = scale[:, None, :]
        norm_final = nn.LayerNorm(self.hidden_dim)(x)
        x = modulate(norm_final, shift, scale)
        x = nn.Dense(
            features=self.patch_size * self.patch_size * self.out_channels,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
        )(x)
        return x


class DiT(nn.Module):
    """
    DiT Model with transformer blocks
    """

    depth: int
    hidden_dim: int
    num_heads: int
    patch_size: int
    in_channels: int = 4
    out_channels: int = 4
    num_classes: int = 1000  # TODO: check this value

    # Weights are initialized via kernel_init / bias_init in submodules of PatchEmbed,
    # LabelEmbed, DiT blocks, adaLN-Zero and FinalLayer

    def unpatchify(self, x):
        """
        Unpatchify the output to get the image
        x: (B, N, patch_size*patch_size*C)
        return: (B, H, W, C)
        """
        c = self.out_channels
        p = self.patch_size
        h = w = int((x.shape[1]) ** 0.5)
        assert h * w == x.shape[1], "Number of patches does not match"
        x = x.reshape(x.shape[0], h, w, p, p, c)
        x = jnp.einsum("bhwpqc->bhpwqc", x)
        imgs = x.reshape(x.shape[0], h * p, w * p, c)
        return imgs

    def forward(self, x, t, r, y, train=True):
        """
        Forward pass through the model
        x: input image (B, H, W, C)
        t: diffusion/ flow time (B,)
        r: reference time (B,)
        y: class labels (B,)
        """
        # Patch embedding
        x_emb = PatchEmbed(self.patch_size, self.in_channels, self.hidden_dim)(
            x
        )  # (B, N, hidden_dim)
        # Positional embedding
        x_emb = PositionalEmbed(x_emb.shape[1], self.hidden_dim)(
            x_emb
        )  # (B, N, hidden_dim)
        # Time embeddings
        t_emb = TimeEmbed(self.hidden_dim)(t)  # (B, hidden_dim)
        r_emb = TimeEmbed(self.hidden_dim)(r)  # (B, hidden_dim)
        # Label embeddings
        y_emb = LabelEmbed(self.hidden_dim, self.num_classes, drop_prob=0.1)(
            y, train=train
        )  # (B, hidden_dim)
        # Combine into one conditioning vector per sample
        # TODO check : Mean Flows logic: combine both times + label
        cond = r_emb + t_emb + y_emb  # (B, D)
        # Transformer blocks with adaLN-Zero conditioning
        for _ in range(self.depth):
            block = DiTBlock(self.hidden_dim, self.num_heads)
            x = block(x_emb, cond)
        # Final layer
        x = FinalLayer(self.hidden_dim, self.patch_size, self.out_channels)(
            x, cond
        )  # (B, N, patch_size*patch_size*C)
        x = self.unpatchify(x)  # (B, H, W, C)
        return x

    def forward_with_cfg(self, x, cfg):
        """
        Forward pass through the model with cfg
        x: input image (B, H, W, C)
        cfg: configuration dictionary with keys:
            't': diffusion/ flow time (B,)
            'r': reference time (B,)
            'y': class labels (B,)
            'scale': cfg scale for classifier-free guidance
        """

        # TODO: check why the batch was halfed and discarded
        t = cfg["t"]
        r = cfg["r"]
        y = cfg["y"]
        # conditional pass: real labels
        cond_pass = self.forward(x, t, r, y, train=False)
        # unconditional pass: null labels
        y_uncond = jnp.full_like(y, fill_value=self.num_classes)  # null label index
        uncond_pass = self.forward(x, t, r, y_uncond, train=False)
        # no random dropout here
        # apply classifier-free guidance on the full output (all channels)
        cond_eps, uncond_eps = cond_pass, uncond_pass
        guided_eps = uncond_eps + cfg["scale"] * (cond_eps - uncond_eps)
        return guided_eps


########################################################
#              Time and Context embedding          #####
########################################################


class TimeEmbed(nn.Module):
    """
    Time embedding using sinosoidal embeddings
    """

    hidden_dim: int
    freq_dim: int = 256

    @nn.compact
    def __call__(self, t):
        # standard transformer sinusoidal embedding
        half_dim = self.freq_dim // 2
        freqs = jnp.exp(-jnp.arange(half_dim) * (jnp.log(10000.0) / half_dim))
        args = t[:, None] * freqs[None, :]
        emb = jnp.concatenate([jnp.sin(args), jnp.cos(args)], axis=-1)
        if self.freq_dim % 2 == 1:
            emb = jnp.concatenate([emb, jnp.zeros((emb.shape[0], 1))], axis=-1)
        t = MLP_with_SiLU(self.hidden_dim)(emb)
        return t


########################################################
#          Helper Functions and Classes            #####
########################################################


def modulate(x, shift, scale):
    """
    Modulate the input x with shift and scale
    """
    return x * (1 + scale) + shift


class MLP_with_GELU(nn.Module):
    """
    2-layer MLP with GELU activation
    """

    hidden_dim: int
    mlp_ratio: float = 4.0

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(
            features=int(self.hidden_dim * self.mlp_ratio),
            kernel_init=nn.initializers.normal(stddev=0.02),
            bias_init=nn.initializers.zeros,
        )(x)
        # no dropout as mentioned in the paper
        x = nn.gelu(x)
        x = nn.Dense(
            features=self.hidden_dim,
            kernel_init=nn.initializers.normal(stddev=0.02),
            bias_init=nn.initializers.zeros,
        )(x)
        return x


class MLP_with_SiLU(nn.Module):
    """
    2-layer MLP with SiLU activation
    """

    hidden_dim: int
    mlp_ratio: float = 4.0

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(
            features=int(self.hidden_dim * self.mlp_ratio),
            kernel_init=nn.initializers.normal(stddev=0.02),
            bias_init=nn.initializers.zeros,
        )(x)
        # no dropout as mentioned in the paper
        x = nn.silu(x)
        x = nn.Dense(
            features=self.hidden_dim,
            kernel_init=nn.initializers.normal(stddev=0.02),
            bias_init=nn.initializers.zeros,
        )(x)
        return x


class AdaLNZero(nn.Module):
    """
    adaptive Layer Normalization (adaLN - Zero) for conditioning
    """

    hidden_dim: int
    parts: int

    @nn.compact
    def __call__(self, h):
        h = nn.silu(h)
        # last linear has zero init for kernel and bias
        h = nn.Dense(
            self.hidden_dim * self.parts,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
        )(h)
        return h


########################################################
#               DiT Config Classes                 #####
########################################################


# check for params=131M, flops=5.6G
def DiT_B_4(**kwargs):
    return DiT(depth=12, hidden_dim=768, num_heads=12, patch_size=4, **kwargs)


# check for params=131M, flops=23.1G
def DiT_B_2(**kwargs):
    return DiT(depth=12, hidden_dim=768, num_heads=12, patch_size=2, **kwargs)


# check for params=497.8M, flops=54G
def DiT_M_2(**kwargs):
    return DiT(depth=16, hidden_dim=1024, num_heads=16, patch_size=2, **kwargs)


# check for params=459M, flops=119G
def DiT_L_2(**kwargs):
    return DiT(depth=24, hidden_dim=1024, num_heads=16, patch_size=2, **kwargs)


# check for params=676M, flops=119G
def DiT_XL_2(**kwargs):
    return DiT(depth=28, hidden_dim=1152, num_heads=16, patch_size=2, **kwargs)
