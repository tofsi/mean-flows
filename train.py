########################################################
#               Training DiT Model                 #####
########################################################

import jax
import numpy as np
import jax.numpy as jnp
import flax
import flax.linen as nn
import optax
from dataclasses import dataclass
from typing import Tuple

from prepare_imagenet import build_dataloaders
from VAE_tokenizer import encode_images_to_latents, decode_latents_to_images
from DiT_model import DiT_B_4, DiT_B_2, DiT_M_2, DiT_L_2, DiT_XL_2
from mean_flows import algorithm_1, algorithm_2
import fid

LATENT_SHAPE = (32, 32, 4)  # To match paper at page 14
LATENT_DIM = np.prod(LATENT_SHAPE)


@dataclass
class TrainingParams:
    """
    Data class to hold training parameters.
    """

    architecture: str
    epochs: int
    lr: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.95
    ema_decay: float = 0.9999
    p: float  # loss metric
    omega: float  # weight for classification loss
    embed_t_r: str  # method to embed t and r
    jvp_computation: Tuple[bool, bool] = (False, True)  # JVP computation options


class Trainer(nn.Module):
    TrainingParams: TrainingParams

    def adam_optimizer():
        """Create Adam optimizer with given learning rate and betas."""
        return optax.adam(
            learning_rate=trainingParams.lr,
            b1=trainingParams.beta1,
            b2=trainingParams.beta2,
        )

    def model():
        """Select DiT model based on architecture parameter."""
        if trainingParams.architecture == "DiT-B-4":
            return DiT_B_4()
        elif trainingParams.architecture == "DiT-B-2":
            return DiT_B_2()
        elif trainingParams.architecture == "DiT-M-2":
            return DiT_M_2()
        elif trainingParams.architecture == "DiT-L-2":
            return DiT_L_2()
        elif trainingParams.architecture == "DiT-XL-2":
            return DiT_XL_2()
        else:
            raise ValueError("Unsupported architecture")

    def train(self):
        """Main training loop."""
        # 1. Build dataloaders
        train_loader, val_loader, test_loader = build_dataloaders(
            root_dir="./imagenet",
            batch_size=32,
            num_workers=4,
        )

        # 2. Initialize model and optimizer
        model = Trainer.model()
        key = jax.random.PRNGKey(42)
        optimizer = self.adam_optimizer()
        # TODO: what goes here?
        params = None
        opt_state = optimizer.init(params)

        # 3. Training loop
        for epoch in range(trainingParams.epochs):
            for batch_idx, (images, labels) in enumerate(train_loader):
                # Encode to latents
                latents_np = encode_images_to_latents(images)
                x = jnp.array(latents_np)  # Convert to JAX array
                y = jnp.array(labels, dtype=jnp.int32)
                key, subkey = jax.random.split(key)
                loss, grads = jax.value_and_grad(algorithm_1)(
                    model.forward,
                    params,
                    x,
                    y,
                    subkey,
                    0.5,
                    "uniform",
                    None,
                    trainingParams.p,
                    trainingParams.omega,
                    trainingParams.embed_t_r,
                    trainingParams.jvp_computation,
                )
                updates, opt_state = optimizer.update(grads, opt_state)
                params = optax.apply_updates(params, updates)
                if batch_idx % 100 == 0:
                    print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss}")
            print(f"Epoch {epoch} completed")
        return params

    def generate_samples(self, params, num_samples=50_000, batch_size=128):
        """
        Generates decoded RGB images using:
            - trained DiT params
            - algorithm_2 sampler
            - VAE decode_latents_to_images()
        Returns a Python generator that yields (B, H, W, 3) numpy batches.
        """

        # Build DiT model once
        model = Trainer.model()

        # Wrapper around model.forward that algorithm_2 expects
        def fn_wrapped(x_flat, t, r, y=None):
            """
            x_flat: (B, LATENT_DIM)
            t, r: scalar floats or arrays (B,)
            y: class labels (B,) or None
            """
            B = x_flat.shape[0]
            x = x_flat.reshape(B, *LATENT_SHAPE)

            # If unconditional: use null class = num_classes
            if y is None:
                y = jnp.full((B,), fill_value=model.num_classes)

            # DiT forward: model.forward(x, t, r, y)
            u = model.forward(params, x, t, r, y, train=False)
            return u.reshape(B, LATENT_DIM)

        # Batch generator for FID
        key = jax.random.PRNGKey(0)
        n_batches = num_samples // batch_size

        for _ in range(n_batches):
            key, sub = jax.random.split(key)

            # mean-flow sample in latent space
            latents_flat = algorithm_2(
                fn_wrapped,
                LATENT_DIM,
                sub,
                batch_size,
                self.TrainingParams.embed_t_r,  # (t, r) embedding fn
                n_steps=1,  # MeanFlow 1-NFE
                c=None,
            )

            latents = latents_flat.reshape(batch_size, *LATENT_SHAPE)

            # decode to RGB torch tensor (B,3,256,256)
            imgs_torch = decode_latents_to_images(latents)

            # convert to numpy BHWC for Inception
            imgs_np = np.transpose(imgs_torch.cpu().numpy(), (0, 2, 3, 1)).astype(
                "float32"
            )

            yield imgs_np

    def evaluate_model(self, params, num_samples=50_000):
        inception = fid.make_inception()

        # Load precomputed real stats
        stats = np.load("imagenet_inception_stats.npz")
        # TODO: We need to generate the above file
        mu_w = jnp.asarray(stats["mu_w"])
        cov_w = jnp.asarray(stats["cov_w"])

        # Use the generator we just wrote
        image_iter = self.generate_samples(params, num_samples)

        # Extract features
        gen_feats = fid.extract_features_in_batches(inception, image_iter)

        # Compute stats + FID
        mu, cov = fid.compute_stats(gen_feats)
        fid_value = fid.fid_from_stats(mu, cov, mu_w, cov_w)

        print("FID-50K:", float(fid_value))
        return float(fid_value)


if __name__ == "__main__":
    trainingParams = TrainingParams(
        architecture="DiT-B-4",
        epochs=10,
        p=0.0,
        omega=1.0,
        embed_t_r=lambda t, r: (t, r),
    )
    trainer = Trainer(trainingParams)
    trained_params = trainer.train()
