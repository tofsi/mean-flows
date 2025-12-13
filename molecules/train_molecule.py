import numpy as np
import sys

sys.path.append("../..")
from molecule_DiT import MoleculeDiT_B
from train import Trainer, TrainingParams
import os

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
import pickle
from mean_flows import algorithm_1, algorithm_2

class MoleculeLatentDataset:
    """
    Simple iterable that yields (latent_batch, dummy_label_batch) for molecule training.
    Labels are dummy zeros and are ignored by MoleculeDiT.
    """

    def __init__(self, latents: np.ndarray, batch_size: int):
        assert latents.ndim == 2, "Expected (N, latent_dim) array of latents"
        self.latents = latents.astype(np.float32)
        self.batch_size = batch_size

    def __iter__(self):
        N = self.latents.shape[0]
        indices = np.random.permutation(N)
        for start in range(0, N, self.batch_size):
            idx = indices[start : start + self.batch_size]
            x = self.latents[idx]  # (B, 56)
            y = np.zeros(x.shape[0], dtype=np.int32)  # dummy labels, unused
            yield x, y


class MoleculeTrainer(Trainer):
    trainingParams: TrainingParams
    _batch_size = 128

    def __init__(
        self,
        trainingParams: TrainingParams,
        latent_path: str,
        checkpoint_dir: str = "checkpoints_mol",
        resume_from_checkpoints: bool = True,
    ):
        self.trainingParams = trainingParams
        self.latent_path = latent_path
        self.resume_from_checkpoints = resume_from_checkpoints
        self.checkpoint_dir = os.path.abspath(checkpoint_dir)
        arch = trainingParams.architecture
        if arch == "Mol-DiT-B":
            self.model = MoleculeDiT_B()
        else:
            raise ValueError(f"Unsupported molecule architecture: {arch}")

        def fn_apply(vars, x, tr, y, rng):
            return self.model.apply(
                vars,
                x,
                tr,
                y,
                train=True,
                method=self.model.forward,
                rngs={"dropout": rng},
            )

        self.fn_apply = fn_apply

        latent_npz = np.load(self.latent_path, allow_pickle=True)
        self.molecule_latents = latent_npz["latent_vectors"]

    def train(self):
        """Main training loop for molecule latents."""
        # 1. Build training loop for molecule latents.

        train_loader = MoleculeLatentDataset(self.molecule_latents, self._batch_size)

        # 2. Init or restore params
        key = jax.random.PRNGKey(42)
        optimizer = self.adam_optimizer()
        start_epoch = 0
        global_step = 0
        start_from_scratch = not self.resume_from_checkpoints
        if self.resume_from_checkpoints:
            restored = self.load_checkpoint()
            if restored is not None:
                print(f"[MoleculeTrainer] Resumed full state from {self.checkpoint_dir}")
                params = restored.get("params")
                opt_state = restored.get("opt_state")
                global_step = int(restored.get("global_step", 0))
                # resume from next epoch
                start_epoch = int(restored.get("epoch", 0)) + 1
                key = restored.get("key", key)
            else:
                start_from_scratch = True
                print(f"[MoleculeTrainer][Warning] Failed to restore state from {self.checkpoint_dir}")
        if start_from_scratch:
            key, key_params = jax.random.split(key)
            dummy_x = jnp.zeros((1, 56), dtype=jnp.float32)
            dummy_t = jnp.ones((1,), dtype=jnp.float32)
            dummy_r = jnp.ones((1,), dtype=jnp.float32)
            dummy_tr = self.trainingParams.embed_t_r(dummy_t, dummy_r)
            dummy_y = jnp.zeros((1,), dtype=jnp.int32)  # ignored

            variables = self.model.init(
                key_params,
                dummy_x,
                dummy_tr,
                dummy_y,
                train=False,
                method=self.model.forward,
            )
            params = variables["params"]
            opt_state = optimizer.init(params)

        # 3. Training loop
        for epoch in range(start_epoch, self.trainingParams.epochs):
            t0 = time.time()
            epoch_losses = []

            for batch_idx, (latents_np, labels_np) in enumerate(train_loader):
                # latents_np: (B, 56), labels_np are dummy zeros
                x = jnp.array(latents_np, dtype=jnp.float32)
                y = jnp.array(labels_np, dtype=jnp.int32)  # ignored by MoleculeDiT
                key, subkey = jax.random.split(key, 2)

                @jax.jit
                def compute_loss_and_grads(params, x, y, key):
                    subkey, dropout_key = jax.random.split(key, 2)

                    def fn_for_algo1(vars_dict, x, tr):
                        return self.fn_apply(vars_dict, x, tr, y, dropout_key)

                    loss, grads = jax.value_and_grad(algorithm_1, argnums=1)(
                        fn_for_algo1,
                        params,
                        x,
                        y,
                        subkey,
                        self.trainingParams.ratio_r_not_eq_t,
                        self.trainingParams.time_sampler_name,
                        self.trainingParams.time_sampler_params,
                        self.trainingParams.p,
                        self.trainingParams.omega,
                        self.model.num_classes,
                        self.trainingParams.embed_t_r,
                        self.trainingParams.jvp_computation,
                    )
                    return loss, grads
                

                loss, grads = compute_loss_and_grads(params, x, y, subkey)
                updates, opt_state = optimizer.update(grads, opt_state)
                params = optax.apply_updates(params, updates)
                epoch_losses.append(float(loss))
                global_step += 1
                if batch_idx % 16 == 0:
                    grad_norm = jnp.sqrt(
                        sum(
                            jnp.sum(jnp.square(g))
                            for g in jax.tree_util.tree_leaves(grads)
                        )
                    )
                    print(
                        f"[MoleculeTrainer] Epoch {epoch}, Batch {batch_idx}, "
                        f"Loss: {loss}, Grad norm: {grad_norm:.6f}"
                    )

            mean_loss = float(np.mean(epoch_losses))
            epoch_time = time.time() - t0
            print(
                f"[MoleculeTrainer] epoch {epoch}  mean_loss={mean_loss:.4f}  time={epoch_time/60:.1f}m"
            )
            state = {
                "params": params,
                "opt_state": opt_state,
                "epoch": epoch,
                "global_step": global_step,
                "key": key,
            }
            metrics = {
                "epoch": epoch,
                "global_step": global_step,
                "mean_loss": mean_loss,
                "epoch_time_sec": epoch_time,
            }
            self.save_checkpoint_and_metrics(
                state,
                epoch,
                metrics,
            )
        return params


if __name__ == "__main__":
    trainingParams = TrainingParams(
        architecture="Mol-DiT-B",
        epochs=20,
        lr=1e-4,
        beta1=0.9,
        beta2=0.95,
        ema_decay=0.9999,
        p=0.0,
        omega=None,  # 1.0,
        ratio_r_not_eq_t=0.25,
        jvp_computation=(False, False),
        embed_t_r_name="tr",
        embed_t_r=lambda t, r: (t, t - r),
        time_embed_dim=256,
        time_sampler_params=None,
    )

    latent_path = "vae_qm9/qvae/latent_vectors_sample.npz"
    trainer = MoleculeTrainer(trainingParams, latent_path)
    trained_params = trainer.train()
