########################################################
#               Training DiT Model                 #####
########################################################
import os, json, time
from flax.training import checkpoints
import jax
import numpy as np
import jax.numpy as jnp
import flax
import flax.linen as nn
import optax
from dataclasses import dataclass, field
from typing import Tuple, Callable, Optional, Any, Dict

from prepare_imagenet import get_dataloaders, get_dataloaders_extracted
from VAE_tokenizer import (
    VAETokenizer,
)  # encode_images_to_latents, decode_latents_to_images
from DiT_model import DiT_B_4, DiT_B_2, DiT_M_2, DiT_L_2, DiT_XL_2
from mean_flows import algorithm_1, algorithm_2

# import fid
import fid
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
IMAGENET_ROOT = PROJECT_DIR / "imagenet"

LATENT_SHAPE = (32, 32, 4)  # To match paper at page 14
LATENT_DIM = np.prod(LATENT_SHAPE)
# Cheap proxy FID settings
FID_K = 2  # FID-1K proxy
FID_BATCH_SIZE = 2


@dataclass
class TrainingParams:
    """
    Data class to hold training parameters.
    """

    architecture: str
    epochs: int

    lr: float  # = 1e-4
    beta1: float  # = 0.9
    beta2: float  # = 0.95
    ema_decay: float  # = 0.9999
    # Required hyperparameters matching algorithm_1
    p: float
    omega: float
    ratio_r_not_eq_t: float  # r-sampling ratio

    # JVP computation option (from Table 1b)
    jvp_computation: Tuple[bool, bool]

    # Embedding choice
    embed_t_r_name: str  # string label
    embed_t_r: Callable[[Any, Any], tuple]  # actual lambda injected by launcher
    time_embed_dim: int
    # Time sampler: None means uniform, otherwise lognorm(a, b)
    time_sampler_params: Optional[Tuple[float, float]]

    @property
    def time_sampler_name(self):
        return "uniform" if self.time_sampler_params is None else "lognorm"


class Trainer:
    trainingParams: TrainingParams

    def __init__(
        self,
        trainingParams,
        validation_fid_stats_path=IMAGENET_ROOT / "imagenet_val_stats.npz",
    ):
        self.trainingParams = trainingParams

        arch = trainingParams.architecture
        if arch == "DiT-B-4":
            self.model = DiT_B_4()
        elif arch == "DiT-B-2":
            self.model = DiT_B_2()
        elif arch == "DiT-M-2":
            self.model = DiT_M_2()
        elif arch == "DiT-L-2":
            self.model = DiT_L_2()
        elif arch == "DiT-XL-2":
            self.model = DiT_XL_2()
        else:
            raise ValueError("Unsupported architecture")

        # define apply-shaped fn ONCE
        def fn_apply(vars, x_bhwc, tr, y, rng):
            return self.model.apply(
                vars,
                x_bhwc,
                tr,
                y,
                train=True,
                method=self.model.forward,
                rngs={"dropout": rng},
            )

        self.fn_apply = fn_apply

        # Setup FID Calculations
        self.fid_extract = fid.make_fid_feature_extractor()
        # Statistics of validation set, to be compared with values from generated images.
        stats = np.load(validation_fid_stats_path)
        self.mu_w = jnp.asarray(stats["mu"])
        self.cov_w = jnp.asarray(stats["cov"])

        # Tokenizer to extract latent representations
        self.vae_tokenizer = VAETokenizer()  # TODO: change

    def adam_optimizer(self):
        """Create Adam optimizer with given learning rate and betas."""
        return optax.adam(
            learning_rate=self.trainingParams.lr,
            b1=self.trainingParams.beta1,
            b2=self.trainingParams.beta2,
        )

    def train(self):
        """Main training loop."""
        # 1. Build dataloaders
        # If you have not already extracted imagenet, uncomment the below version.
        # train_loader, test_loader = get_dataloaders(batch_size=32)
        train_loader, val_loader = get_dataloaders_extracted(
            root_dir=str(IMAGENET_ROOT),  # your extracted folder
            batch_size=2,  # NOTE: Increase batch size relative to GPU memory.
            num_workers=2,
            max_train_samples=4,
            max_val_samples=4,
        )

        # 2. Initialize model and optimizer
        key = jax.random.PRNGKey(42)
        key, key_params = jax.random.split(key)
        # Latents are BHWC = (B, 32, 32, 4) from your VAE
        dummy_x = jnp.zeros((1, 32, 32, 4), dtype=jnp.float32)

        # dummy time embedding
        dummy_t = jnp.ones((1,), dtype=jnp.float32)
        dummy_r = jnp.ones((1,), dtype=jnp.float32)

        # build whatever embed_t_r wants (length varies by ablation)
        dummy_tr = self.trainingParams.embed_t_r(dummy_t, dummy_r)

        # y are class labels: shape (B,)
        dummy_y = jnp.zeros((1,), dtype=jnp.int32)

        # IMPORTANT: DiT has no __call__, so init via method=model.forward.
        # Also set train=False so LabelEmbed won't try to use dropout RNG at init.
        variables = self.model.init(
            key_params,
            dummy_x,
            dummy_tr,
            dummy_y,
            train=False,
            method=self.model.forward,
        )
        params = variables["params"]
        optimizer = self.adam_optimizer()
        opt_state = optimizer.init(params)

        # 3. Training loop
        global_step = 0
        for epoch in range(self.trainingParams.epochs):
            t0 = time.time()
            epoch_losses = []
            for batch_idx, (images, labels) in enumerate(train_loader):
                # Encode to latents
                latents_np = self.vae_tokenizer.encode_images_to_latents(images)
                x = jnp.array(latents_np)  # Convert to JAX array
                y = jnp.array(labels, dtype=jnp.int32)
                key, subkey, dropout_key = jax.random.split(key, 3)

                loss, grads = jax.value_and_grad(algorithm_1, argnums=1)(
                    lambda vars, x, tr, y: self.fn_apply(vars, x, tr, y, dropout_key),
                    params,
                    x,
                    y,
                    subkey,
                    self.trainingParams.ratio_r_not_eq_t,
                    self.trainingParams.time_sampler_name,
                    self.trainingParams.time_sampler_params,
                    self.trainingParams.p,
                    self.trainingParams.omega,
                    self.trainingParams.embed_t_r,
                    self.trainingParams.jvp_computation,
                )
                updates, opt_state = optimizer.update(grads, opt_state)
                params = optax.apply_updates(params, updates)
                epoch_losses.append(float(loss))
                global_step += 1
                if batch_idx % 100 == 0:
                    print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss}")

            print(f"Epoch {epoch} completed")
            # ---- end of epoch: compute mean loss ----
            mean_loss = float(np.mean(epoch_losses))
            # Do FID-k once per epoch
            fid_proxy = self.eval_fid(params, FID_K, FID_BATCH_SIZE)
            epoch_time = time.time() - t0
            metrics = {
                "epoch": epoch,
                "global_step": global_step,
                "mean_loss": mean_loss,
                f"fid_{FID_K}": fid_proxy,
                "epoch_time_sec": epoch_time,
            }

            print(
                f"[epoch {epoch}] mean_loss={mean_loss:.4f}  FID-{FID_K}={fid_proxy:.3f}  time={epoch_time/60:.1f}m"
            )
            # ---- save params + metrics (overwrites params, appends metrics) ----
            self.save_checkpoint_and_metrics(params, epoch, metrics)
        return params

    def generate_samples(self, params, num_samples=50_000, batch_size=128):
        """
        Generates decoded RGB images using:
            - trained DiT params
            - algorithm_2 sampler
            - VAE decode_latents_to_images()
        Returns a Python generator that yields (B, H, W, 3) numpy batches.
        """

        # Wrapper around model.forward that algorithm_2 expects
        def fn_wrapped(x_flat, tr, y=None):
            """
            x_flat: (B, LATENT_DIM)
            t, r: scalar floats or arrays (B,)
            y: class labels (B,) or None
            """
            B = x_flat.shape[0]
            x = x_flat.reshape(B, *LATENT_SHAPE)

            # If unconditional: use null class = num_classes
            # TODO: Consistent unconditional convention. I think 0 should be used?
            if y is None:
                y = jnp.full((B,), fill_value=self.model.num_classes)

            # DiT forward pass.
            u = self.model.apply(
                {"params": params},
                x,
                tr,
                y,
                train=False,
                method=self.model.forward,  # or method=model.__call__ if you use __call__
            )  # (B, H, W, C) in latent space shape
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
                self.trainingParams.embed_t_r,  # (t, r) embedding fn
                n_steps=1,  # MeanFlow 1-NFE
                c=None,
            )

            latents = latents_flat.reshape(batch_size, *LATENT_SHAPE)

            # decode to RGB torch tensor (B,3,256,256)
            imgs_torch = self.vae_tokenizer.decode_latents_to_images(latents)

            # convert to numpy BHWC for Inception
            imgs_np = np.transpose(imgs_torch.cpu().numpy(), (0, 2, 3, 1)).astype(
                "float32"
            )

            yield imgs_np

    def eval_fid(self, trained_params, num_samples=5000, batch_size=64):
        batch_size = min(batch_size, num_samples)

        feats_fake = []

        # 2. Loop over generated batches
        for imgs_np in self.generate_samples(trained_params, num_samples, batch_size):
            # imgs_np: (B, 256, 256, 3) in [0,1], float32
            imgs_jax = jnp.asarray(imgs_np)  # (B,H,W,3)

            # 3. Extract Inception features for this batch (B,2048)
            f = self.fid_extract(imgs_jax)
            feats_fake.append(np.array(f))  # move to host

        # 4. Concatenate all features
        feats_fake = np.concatenate(feats_fake, axis=0)  # (N, 2048)

        # 5. Compute fake stats in JAX
        x = jnp.asarray(feats_fake)
        mu = jnp.mean(x, axis=0)
        xc = x - mu
        cov = (xc.T @ xc) / (x.shape[0] - 1)

        # 6. FID
        fid_value = fid.fid_from_stats(
            np.array(mu),
            np.array(cov),
            np.array(self.mu_w),
            np.array(self.cov_w),
        )
        return float(fid_value)

    def save_checkpoint_and_metrics(
        self, params, epoch, metrics, ckpt_dir="checkpoints"
    ):
        ckpt_dir = os.path.abspath(ckpt_dir)
        os.makedirs(ckpt_dir, exist_ok=True)

        # Overwrite/latest checkpoint is fine:
        checkpoints.save_checkpoint(
            ckpt_dir=ckpt_dir,
            target=params,
            step=epoch,
            overwrite=True,
            keep=1,  # keep only latest
        )

        # Append metrics to a JSONL file
        metrics_path = os.path.join(ckpt_dir, "metrics.jsonl")
        with open(metrics_path, "a") as f:
            f.write(json.dumps(metrics) + "\n")


if __name__ == "__main__":
    trainingParams = TrainingParams(
        architecture="DiT-B-4",
        epochs=10,
        p=0.0,
        omega=1.0,
        ratio_r_not_eq_t=0.25,
        time_sampler_params=None,
        embed_t_r_name="tr",
        embed_t_r=lambda t, r: (t, t - r),
    )
    trainer = Trainer(trainingParams)
    trained_params = trainer.train()

    fid_final = trainer.eval_fid(trained_params, num_samples=50_000)
