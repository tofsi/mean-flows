#from diffusers.models import AutoencoderKL
#from diffusers import StableDiffusionPipeline

#model = "CompVis/stable-diffusion-v1-4"
#vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
#pipe = StableDiffusionPipeline.from_pretrained(model, vae=vae)


import torch
import jax
import jax.numpy as jnp
from jax import random
import numpy as np
from diffusers.models import AutoencoderKL
from torchvision import transforms, datasets
from torch.utils.data import DataLoader


# ========================================
# 1. SETUP VAE (PyTorch)
# ========================================
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
vae = vae.to("cuda" if torch.cuda.is_available() else "cpu")
vae.eval()

print(f"VAE scaling factor: {vae.config.scaling_factor}")  # Should be ~0.18215

def encode_images_to_latents(images_batch):
    """
    Encode images to latents for DiT.
    
    Args:
        images_batch: torch.Tensor (B, 3, 256, 256) in range [0, 1]
    
    Returns:
        latents: numpy array (B, 32, 32, 4) - JAX format (channels last)
    """
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
    """
    Decode latents back to images.
    
    Args:
        latents_jax: JAX array (B, 32, 32, 4) or numpy array
    
    Returns:
        images: torch.Tensor (B, 3, 256, 256) in range [0, 1]
    """
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
    
    # Convert from [-1, 1] to [0, 1]
    images = (images + 1) / 2
    images = torch.clamp(images, 0, 1)
    
    return images


# # ========================================
# #  TRAINING SETUP
# # ========================================
# def create_train_step(model, optimizer):
#     """Create a JIT-compiled training step."""
    
#     @jax.jit
#     def train_step(state, x, y, key):
#         """
#         Single training step.
        
#         Args:
#             state: optimizer state
#             x: (B, 32, 32, 4) latent batch
#             y: (B,) labels
#             key: PRNGKey
        
#         Returns:
#             new_state, loss
#         """
#         def loss_fn(params):
#             def model_forward(x_in, t_in, r_in, y_in, train):
#                 return model.apply(
#                     {'params': params},
#                     x_in, t_in, r_in, y_in, train,
#                     rngs={'dropout': key} if train else None
#                 )
#             return algorithm_1_dit(model_forward, x, y, key)
        
#         loss, grads = jax.value_and_grad(loss_fn)(state.params)
#         new_state = optimizer.update(grads, state)
        
#         return new_state, loss
    
#     return train_step

# # ========================================
# # 4. MAIN TRAINING LOOP
# # ========================================
# def train_meanflow_dit():
#     # Initialize model
#     key = random.PRNGKey(0)
#     key, init_key = random.split(key)
    
#     # Create DiT model
#     model = DiT_XL_2(
#         in_channels=4,
#         out_channels=4,
#         num_classes=1000
#     )
    
#     # Initialize with dummy input
#     dummy_x = jnp.zeros((1, 32, 32, 4))
#     dummy_t = jnp.zeros((1,))
#     dummy_r = jnp.zeros((1,))
#     dummy_y = jnp.zeros((1,), dtype=jnp.int32)
    
#     variables = model.init(
#         init_key,
#         dummy_x, dummy_t, dummy_r, dummy_y,
#         train=True
#     )
#     params = variables['params']
    
#     # Setup optimizer (use your preferred optimizer)
#     import optax
#     learning_rate = 1e-4
#     optimizer = optax.adam(learning_rate)
#     opt_state = optimizer.init(params)
    
#     # Create training step function
#     train_step = create_train_step(model, optimizer)
    
#     # ImageNet dataset
#     transform = transforms.Compose([
#         transforms.Resize(256),
#         transforms.CenterCrop(256),
#         transforms.ToTensor(),
#     ])
    
#     train_dataset = datasets.ImageNet(
#         root='/path/to/imagenet',
#         split='train',
#         transform=transform
#     )
    
#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=32,
#         shuffle=True,
#         num_workers=4,
#         pin_memory=True
#     )
    
#     # Training loop
#     num_epochs = 10
#     for epoch in range(num_epochs):
#         for batch_idx, (images, labels) in enumerate(train_loader):
#             # Encode to latents
#             latents_np = encode_images_to_latents(images)  # (B, 32, 32, 4)
#             x = jnp.array(latents_np, dtype=T)
            
#             # Convert labels to JAX
#             y = jnp.array(labels.numpy(), dtype=jnp.int32)
            
#             # Training step
#             key, step_key = random.split(key)
#             opt_state, loss = train_step(opt_state, x, y, step_key)
            
#             if batch_idx % 100 == 0:
#                 print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss:.4f}")
        
#         print(f"Epoch {epoch} completed")

# # ========================================
# # 5. GENERATION
# # ========================================
# def generate_images_with_dit(model, params, num_images=50, batch_size=10):
#     """
#     Generate images using trained DiT + VAE.
    
#     Returns:
#         images: torch.Tensor (num_images, 3, 256, 256)
#     """
#     key = random.PRNGKey(42)
#     all_images = []
    
#     num_batches = (num_images + batch_size - 1) // batch_size
    
#     for i in range(num_batches):
#         current_batch_size = min(batch_size, num_images - i * batch_size)
        
#         # Sample random labels
#         key, label_key = random.split(key)
#         y = random.randint(label_key, (current_batch_size,), 0, 1000)
        
#         # Generate latents using MeanFlow Algorithm 2
#         key, gen_key = random.split(key)
        
#         def model_forward(x_in, t_in, r_in, y_in, train):
#             return model.apply(
#                 {'params': params},
#                 x_in, t_in, r_in, y_in, train
#             )
        
#         latents_jax = algorithm_2_dit(
#             model_forward,
#             y,
#             gen_key,
#             current_batch_size
#         )  # (B, 32, 32, 4)
        
#         # Decode to images
#         images = decode_latents_to_images(latents_jax)  # (B, 3, 256, 256)
#         all_images.append(images)
    
#     return torch.cat(all_images, dim=0)[:num_images]

# # ========================================
# # 6. TEST EVERYTHING
# # ========================================
# if __name__ == "__main__":
#     print("Testing VAE + DiT integration...")
    
#     # Test 1: VAE encoding/decoding
#     from PIL import Image
#     test_img = Image.open("test.jpg")
#     test_transform = transforms.Compose([
#         transforms.Resize(256),
#         transforms.CenterCrop(256),
#         transforms.ToTensor(),
#     ])
#     test_tensor = test_transform(test_img).unsqueeze(0)
    
#     latent = encode_images_to_latents(test_tensor)
#     print(f"✓ Latent shape: {latent.shape}")  # (1, 32, 32, 4)
#     assert latent.shape == (1, 32, 32, 4), "Wrong latent shape!"
    
#     reconstructed = decode_latents_to_images(latent)
#     print(f"✓ Reconstructed shape: {reconstructed.shape}")  # (1, 3, 256, 256)
    
#     # Test 2: DiT forward pass with latents
#     key = random.PRNGKey(0)
#     model = DiT_XL_2(in_channels=4, out_channels=4, num_classes=1000)
    
#     # Initialize
#     dummy_x = jnp.array(latent)  # (1, 32, 32, 4)
#     dummy_t = jnp.array([0.5])
#     dummy_r = jnp.array([0.0])
#     dummy_y = jnp.array([1], dtype=jnp.int32)
    
#     variables = model.init(key, dummy_x, dummy_t, dummy_r, dummy_y, train=False)
#     output = model.apply(variables, dummy_x, dummy_t, dummy_r, dummy_y, train=False)
#     print(f"✓ DiT output shape: {output.shape}")  # (1, 32, 32, 4)
#     assert output.shape == (1, 32, 32, 4), "Wrong DiT output shape!"
    
#     print("\n✅ All tests passed! Ready to train.")
