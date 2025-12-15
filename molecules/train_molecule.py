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
from vae_qm9.jtnn import Vocab, JTNNVAE
import torch


LATENT_DIM = 56

def to_torch(x, device="cuda", dtype=torch.float32):
    # Accepts JAX array / NumPy array / Python list / Torch tensor
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    # JAX arrays implement __array__ -> convert via NumPy
    x_np = np.asarray(x)
    return torch.from_numpy(x_np).to(device=device, dtype=dtype)

def decode_latents(latents, vocab_path, model_path):
    """
    decodes molecule latents using JTNNVAE decode()
    """
    vocab = Vocab([x.strip() for x in open(vocab_path)])
    model = JTNNVAE(vocab, 450, LATENT_DIM, 3, stereo=True)
    model.load_state_dict(torch.load(model_path))
    model = model.cuda()
    model.eval()
    results = []
    for i in range(latents.shape[0]):
        latent = latents[i]
        tree_vec = latent[:, :28]
        mol_vec = latent[:, 28:]
        tree_vec = to_torch(tree_vec, device="cuda")
        mol_vec  = to_torch(mol_vec,  device="cuda")
        result = model.decode(tree_vec, mol_vec, prob_decode=False)
        print(f"result : {result}")
        results.append(result)
    return results

class MoleculeLatentDataset:
    """
    Iterable that yields (latent_batch, dummy_label_batch).
    Supports indexing into a subset via `indices`.
    """

    def __init__(self, latents: np.ndarray, indices: np.ndarray, batch_size: int, shuffle: bool = True):
        assert latents.ndim == 2, "Expected (N, latent_dim) array of latents"
        self.latents = latents                # keep original (can be memmap)
        self.indices = np.array(indices)      # subset indices
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        idxs = self.indices.copy()
        if self.shuffle:
            np.random.shuffle(idxs)

        for start in range(0, len(idxs), self.batch_size):
            batch_idx = idxs[start:start + self.batch_size]
            x = self.latents[batch_idx]  # (B, 56)
            y = np.zeros(x.shape[0], dtype=np.int32)  # dummy labels (ignored)
            yield x.astype(np.float32, copy=False), y


class MoleculeTrainer(Trainer):
    trainingParams: TrainingParams
    _batch_size = 2**13

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

        self.fn_apply_train = fn_apply
        def fn_apply_eval(vars, x, tr, y):
            return self.model.apply(
                vars, x, tr, y,
                train=False,
                method=self.model.forward,
            )
        self.fn_apply_eval = fn_apply_eval
        latent_npz = np.load(self.latent_path, allow_pickle=True)
        self.molecule_latents = latent_npz["latent_vectors"]

        N = self.molecule_latents.shape[0]
        rng = np.random.default_rng(42)
        perm = rng.permutation(N)

        n_train = int(0.9 * N)
        self.train_idx = perm[:n_train]
        self.val_idx = perm[n_train:]

    def train(self):
        """Main training loop for molecule latents."""
        # 1. Build training loop for molecule latents.

        train_loader = MoleculeLatentDataset(self.molecule_latents, self.train_idx, self._batch_size, shuffle=True)
        val_loader   = MoleculeLatentDataset(self.molecule_latents, self.val_idx,   self._batch_size, shuffle=False)

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
            dummy_x = jnp.zeros((1, LATENT_DIM), dtype=jnp.float32)
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
                # latents_np: (B, LATENT_DIM), labels_np are dummy zeros
                x = jnp.array(latents_np, dtype=jnp.float32)
                y = jnp.array(labels_np, dtype=jnp.int32)  # ignored by MoleculeDiT
                key, subkey = jax.random.split(key, 2)

                @jax.jit
                def compute_loss_and_grads(params, x, y, key):
                    subkey, dropout_key = jax.random.split(key, 2)

                    def fn_for_algo1(vars_dict, x, tr):
                        return self.fn_apply_train(vars_dict, x, tr, y, dropout_key)

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
            # Ugly fct for validation loss.
            @jax.jit
            def compute_loss_only(params, x, y, key):
                subkey, _dropout_key = jax.random.split(key, 2)

                def fn_for_algo1(vars_dict, x, tr):
                    # eval: train=False, no dropout rng
                    return self.fn_apply_eval(vars_dict, x, tr, y)

                loss = algorithm_1(
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
                return loss
            # ---- validation loss ----
            val_losses = []
            for val_latents_np, val_labels_np in val_loader:
                x_val = jnp.array(val_latents_np, dtype=jnp.float32)
                y_val = jnp.array(val_labels_np, dtype=jnp.int32)

                key, vkey = jax.random.split(key, 2)
                vloss = compute_loss_only(params, x_val, y_val, vkey)
                val_losses.append(float(vloss))

            mean_val_loss = float(np.mean(val_losses)) if len(val_losses) > 0 else float("nan")
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
                "mean_val_loss" : mean_val_loss,
                "epoch_time_sec": epoch_time,
            }
            self.save_checkpoint_and_metrics(
                state,
                epoch,
                metrics,
            )
        return params

    def generate_samples(self, params, num_samples=1000, batch_size=128, seed=0, vocab_path="vae_qm9/qvae/data/merged/vocab.txt", model_path="vae_qm9/qvae/var_model/model.epoch-15"):
        """
        Generates decoded molecule images using:
            - trained DiT params
            - algorithm_2 sampler
            - decode_latents 
        Returns a Python generator that yields (B, LATENT_DIM) numpy batches.
        """
        # Wrapper around model.forward that algorithm_2 expects
        def fn_wrapped(x_flat, tr, y=None):
            """
            x_flat: (B, LATENT_DIM)
            t, r: scalar floats or arrays (B,)
            y: class labels (B,) or None
            """
            B = x_flat.shape[0]
            LATENT_DIM = x_flat.shape[1]
            x = x_flat
            #x = x_flat.reshape(B, *LATENT_SHAPE)

            if y is None:
                y = jnp.full((B,), fill_value=self.model.num_classes)

            # DiT forward pass.
            u = self.model.apply(
                {"params": params},
                x,
                tr,
                y,
                train=False,
                method=self.model.forward, 
            )  # (B, LATENT_DIM) 
            return u

        # Batch generator
        key = jax.random.PRNGKey(seed)
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
            ) # (batch_size, LATENT_DIM)
            
            # TODO: check if this is correct
            latents = latents_flat.reshape(batch_size, 1, LATENT_DIM)

            # decode to molecule (batch_size, 1, LATENT_DIM)
            mol_torch = decode_latents(latents, vocab_path, model_path)

            yield mol_torch


    
        
    


if __name__ == "__main__":
    trainingParams = TrainingParams(
        architecture="Mol-DiT-B",
        epochs=150,
        lr=1e-4,
        beta1=0.9,
        beta2=0.95,
        ema_decay=0.9999,
        p=0.0,
        omega=None,  # 1.0,
        ratio_r_not_eq_t=0.25,
        jvp_computation=(False, True),
        embed_t_r_name="tr",
        embed_t_r=lambda t, r: (t, t - r),
        time_embed_dim=256,
        time_sampler_params=None,
    )

    latent_path = "vae_qm9/qvae/all_latent_vectors.npz"
    trainer = MoleculeTrainer(trainingParams, latent_path)
    trained_params = trainer.train()
