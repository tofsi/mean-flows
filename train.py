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
from VAE_tokenizer import encode_images_to_latents, decode_latents_to_images
from DiT_model import DiT_B_4, DiT_B_2, DiT_M_2, DiT_L_2, DiT_XL_2
from mean_flows import algorithm_1, algorithm_2
import fid
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
IMAGENET_ROOT = PROJECT_DIR / "imagenet"

LATENT_SHAPE = (32, 32, 4)  # To match paper at page 14
LATENT_DIM = np.prod(LATENT_SHAPE)


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

    # Time sampler: None means uniform, otherwise lognorm(a, b)
    time_sampler_params: Optional[Tuple[float, float]]

    @property
    def time_sampler_name(self):
        return "uniform" if self.time_sampler_params is None else "lognorm"


class Trainer:
    trainingParams: TrainingParams

    def __init__(self, trainingParams):
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
            batch_size=32,
            num_workers=4,
        )

        # 2. Initialize model and optimizer
        key = jax.random.PRNGKey(42)
        key, key_params = jax.random.split(key)
        # Latents are BHWC = (B, 32, 32, 4) from your VAE
        dummy_x = jnp.zeros((1, 32, 32, 4), dtype=jnp.float32)

        # t and r are per-sample times: shape (B,)
        dummy_t = jnp.ones((1,), dtype=jnp.float32)
        dummy_r = jnp.ones((1,), dtype=jnp.float32)

        # y are class labels: shape (B,)
        dummy_y = jnp.zeros((1,), dtype=jnp.int32)

        # IMPORTANT: DiT has no __call__, so init via method=model.forward.
        # Also set train=False so LabelEmbed won't try to use dropout RNG at init.
        variables = self.model.init(
            key_params,
            dummy_x,
            dummy_t,
            dummy_r,
            dummy_y,
            train=False,
            method=self.model.forward,
        )
        params = variables["params"]
        optimizer = self.adam_optimizer()
        opt_state = optimizer.init(params)

        # ---- FID setup (done ONCE) ----
        inception = fid.make_inception()
        stats = np.load("imagenet_inception_stats.npz")
        mu_w = jnp.asarray(stats["mu_w"])
        cov_w = jnp.asarray(stats["cov_w"])

        # Cheap proxy FID settings
        fid_k = 1000  # FID-1K proxy

        # 3. Training loop
        global_step = 0
        for epoch in range(self.trainingParams.epochs):
            t0 = time.time()
            epoch_losses = []
            for batch_idx, (images, labels) in enumerate(train_loader):
                # Encode to latents
                latents_np = encode_images_to_latents(images)
                x = jnp.array(latents_np)  # Convert to JAX array
                y = jnp.array(labels, dtype=jnp.int32)
                key, subkey = jax.random.split(key)
                loss, grads = jax.value_and_grad(algorithm_1)(
                    self.model.forward,
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
            fid_proxy = self.evaluate_model(
                params, num_samples=fid_k, inception=inception, mu_w=mu_w, cov_w=cov_w
            )
            epoch_time = time.time() - t0
            metrics = {
                "epoch": epoch,
                "global_step": global_step,
                "mean_loss": mean_loss,
                f"fid_{fid_k}": fid_proxy,
                "epoch_time_sec": epoch_time,
            }

            print(
                f"[epoch {epoch}] mean_loss={mean_loss:.4f}  FID-{fid_k}={fid_proxy:.3f}  time={epoch_time/60:.1f}m"
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
                y = jnp.full((B,), fill_value=self.model.num_classes)

            # DiT forward: model.forward(x, t, r, y)
            u = self.model.forward(params, x, t, r, y, train=False)
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

    def evaluate_model(
        self, params, num_samples=50_000, inception=None, mu_w=None, cov_w=None
    ):
        # Allow cacheing to avoid reloading during training:
        if inception is None:
            inception = fid.make_inception()

        # Load precomputed real stats
        if mu_w is None or cov_w is None:
            stats = np.load("imagenet_inception_stats.npz")
            # TODO: We need to generate the above file
            mu_w = jnp.asarray(stats["mu_w"])
            cov_w = jnp.asarray(stats["cov_w"])

        # Gemerated image iterator
        image_iter = self.generate_samples(params, num_samples)

        # Extract features
        gen_feats = fid.extract_features_in_batches(inception, image_iter)

        # Compute stats + FID
        mu, cov = fid.compute_stats(gen_feats)
        fid_value = fid.fid_from_stats(mu, cov, mu_w, cov_w)

        print("FID_{}: {}".format(num_samples, float(float(fid_value))))
        return float(fid_value)

    def save_checkpoint_and_metrics(
        self, params, epoch, metrics, ckpt_dir="checkpoints"
    ):
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
    fid_final = trainer.evaluate_model(trained_params, num_samples=50_000)
