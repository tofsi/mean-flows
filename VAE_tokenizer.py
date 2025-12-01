# from diffusers.models import AutoencoderKL
# from diffusers import StableDiffusionPipeline

# model = "CompVis/stable-diffusion-v1-4"
# vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
# pipe = StableDiffusionPipeline.from_pretrained(model, vae=vae)


import torch

# import jax
import jax.numpy as jnp

# from jax import random
#import numpy as np
from diffusers.models import AutoencoderKL

#import torch.utils.dlpack
#import jax.dlpack


# from torchvision import transforms, datasets
# from torch.utils.data import DataLoader

""" 
# ========================================
# 1. SETUP VAE (PyTorch)
# ========================================
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
vae = vae.to("cuda" if torch.cuda.is_available() else "cpu")
vae.eval()
print(f"VAE scaling factor: {vae.config.scaling_factor}")  # Should be ~0.18215


def encode_images_to_latents(images_batch):
    # #
    # Encode images to latents for DiT.
# 
    # Args:
    #     images_batch: torch.Tensor (B, 3, 256, 256) in range [0, 1]
# 
    # Returns:
    #     latents: numpy array (B, 32, 32, 4) - JAX format (channels last)
    #
    device = next(vae.parameters()).device
    images_batch = images_batch.to(device)

    # Normalize to [-1, 1] for VAE
    images_batch = images_batch * 2 - 1

    with torch.no_grad():
        latent_dist = vae.encode(images_batch).latent_dist
        latents = latent_dist.sample()
        latents = latents * vae.config.scaling_factor  # Scale

    # Convert from PyTorch (B, C, H, W) to JAX (B, H, W, C)
    latents_np = latents.cpu().numpy()  # (B, 4, 32, 32)
    latents_np = np.transpose(latents_np, (0, 2, 3, 1))  # (B, 32, 32, 4)

    return latents_np


def decode_latents_to_images(latents_jax):
    
    #Decode latents back to images.
#
    #Args:
    #    latents_jax: JAX array (B, 32, 32, 4) or numpy array
#
    #Returns:
    #    images: torch.Tensor (B, 3, 256, 256) in range [0, 1]
    
    device = next(vae.parameters()).device

    # Convert from JAX to numpy if needed
    if isinstance(latents_jax, jnp.ndarray):
        latents_np = np.array(latents_jax)
    else:
        latents_np = latents_jax

    # Convert from JAX format (B, H, W, C) to PyTorch (B, C, H, W)
    latents_np = np.transpose(latents_np, (0, 3, 1, 2))  # (B, 4, 32, 32)

    latents_torch = torch.from_numpy(latents_np).to(device)
    latents_torch = latents_torch / vae.config.scaling_factor  # Unscale

    with torch.no_grad():
        images = vae.decode(latents_torch).sample

    return images
 """


class VAETokenizer:
    """
    Wrapper around Stable Diffusion VAE to interface nicely with JAX:

    - encode_images_to_latents:
        images: torch.Tensor (B, 3, 256, 256) in [0, 1]
        -> latents: np.ndarray (B, 32, 32, 4) in BHWC (JAX-friendly)

    - decode_latents_to_images:
        latents: jnp.ndarray or np.ndarray (B, 32, 32, 4) BHWC
        -> images: torch.Tensor (B, 3, 256, 256) in [0, 1]
    """

    def __init__(self):
        """
        device: "cpu" or "cuda". For your 6GB GPU setup, "cpu" is safer to avoid
        VRAM issues; change to "cuda" if you explicitly want VAE on GPU.
        """
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            print("[info (VAETokenizer)] CUDA unavailable, falling back to CPU")
            self.device = torch.device("cpu")
        self.vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
        self.vae = self.vae.to(self.device)
        self.vae.eval()

        self.scaling_factor = float(self.vae.config.scaling_factor)
        print(f"VAE scaling factor: {self.scaling_factor}")  # ~0.18215

    # -----------------------
    # Encoding: images -> latents
    # -----------------------
    def encode_images_to_latents(self, images_batch: torch.Tensor) -> jnp.ndarray:
        """
        Args:
            images_batch: torch.Tensor (B, 3, 256, 256) in [0, 1]

        Returns:
            latents: numpy array (B, 32, 32, 4) in BHWC (for JAX)
        """
        images_batch = images_batch.to(self.device)

        with torch.no_grad():
            latent_dist = self.vae.encode(images_batch).latent_dist
            latents = latent_dist.sample()
            latents = latents * self.scaling_factor  # Scale

        latents_jnp_old  = jnp.asarray(latents)
        # Convert from PyTorch (B, C, H, W) to JAX (B, H, W, C) as output is described in the paper 
        latents_jnp = jnp.transpose(latents_jnp_old, (0, 2, 3, 1))  # (B, 32, 32, 4)
        
        return latents_jnp

    # -----------------------
    # Decoding: latents -> images
    # -----------------------
    def decode_latents_to_images(self, latents_jax) -> torch.Tensor:
        """
        Args:
            latents_jax: jnp.ndarray or np.ndarray (B, 32, 32, 4) in BHWC

        Returns:
            images: torch.Tensor (B, 3, 256, 256) in [0, 1]
        """

        # BHWC -> BCHW 
        latents_jnp = jnp.transpose(latents_jax, (0, 3, 1, 2))  # (B, 4, 32, 32)
    
        latents_torch = torch.utils.dlpack.from_dlpack(latents_jnp)
        latents_torch = latents_torch / vae.config.scaling_factor  # Unscale

        with torch.no_grad():
            images = self.vae.decode(latents_torch).sample  
            
        return images
