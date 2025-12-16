from train_molecule import MoleculeTrainer, decode_latents, LATENT_DIM
import numpy as np
import matplotlib.pyplot as plt
from flax.training import checkpoints
import os
import sys

sys.path.append("..")
from train import TrainingParams

from rdkit import RDLogger
RDLogger.logger().setLevel(RDLogger.CRITICAL)
from rdkit import Chem
from rdkit.Chem import Draw


def _as_smiles(x):
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="ignore")
    return str(x)


def plot_molecule_comparison_grid(
    dataset_smiles,
    prior_smiles,
    model_smiles,
    out_path="results/prior_vs_model_vs_data.png",
    titles=("from data set", "from prior", "from trained model"),
):
    """3x3 grid: 3 molecules per column (dataset / prior / model)."""
    cols = [dataset_smiles, prior_smiles, model_smiles]
    n_rows = 3
    n_cols = 3

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(9, 9))

    for c in range(n_cols):
        axes[0, c].set_title(titles[c])

    for r in range(n_rows):
        for c in range(n_cols):
            ax = axes[r, c]
            ax.axis("off")

            smiles_list = cols[c]
            if r >= len(smiles_list):
                continue

            s = _as_smiles(smiles_list[r])
            if (not s) or ("ERROR" in s):
                ax.text(0.5, 0.5, "ERROR", ha="center", va="center")
                continue

            mol = Chem.MolFromSmiles(s)
            if mol is None:
                ax.text(0.5, 0.5, "invalid", ha="center", va="center")
                continue

            img = Draw.MolToImage(mol, size=(300, 300))
            ax.imshow(img)
            ax.text(0.5, -0.05, s[:30], ha="center", va="top", transform=ax.transAxes, fontsize=8)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.show()
    plt.close(fig)
    print(f"Saved comparison grid to: {out_path}")


def plotting(num_per_method=3, train=False, seed=42,
             vocab_path="vae_qm9/qvae/data/merged/vocab.txt",
             model_path="vae_qm9/qvae/var_model/model.epoch-15"):
    """Generates a 3x3 comparison: data vs prior vs trained model."""

    trainingParams = TrainingParams(
        architecture="Mol-DiT-B",
        epochs=150,
        lr=1e-4,
        beta1=0.9,
        beta2=0.95,
        ema_decay=0.9999,
        p=0.0,
        omega=None,
        ratio_r_not_eq_t=0.5,
        jvp_computation=(False, True),
        embed_t_r_name="tr",
        embed_t_r=lambda t, r: (t, t - r),
        time_embed_dim=2,
        time_sampler_params=None,
    )

    ckpt_dir = "checkpoints_mol"

    trainer = MoleculeTrainer(
        trainingParams,
        checkpoint_dir=ckpt_dir,
        resume_from_checkpoints=True,
        latent_path="vae_qm9/qvae/all_latent_vectors.npz",
    )

    if train:
        trained_params = trainer.train()
    else:
        restored = trainer.load_checkpoint()
        trained_params = restored.get("params")

    # --- (A) molecules from the data set (decode existing latents) ---
    # Use a stable slice from the validation indices so it doesn't depend on loader shuffling.
    idxs = np.array(trainer.val_idx[:num_per_method], dtype=int)
    latents_flat = trainer.molecule_latents[idxs].astype(np.float32)
    latents = latents_flat.reshape(num_per_method, 1, LATENT_DIM)
    dataset_smiles = decode_latents(latents, vocab_path, model_path)

    # --- (B) molecules from prior z ~ N(0, I) ---
    prior_smiles = None
    for batch in trainer.generate_samples_from_prior(
        num_samples=num_per_method,
        batch_size=num_per_method,
        seed=seed + 1,
        vocab_path=vocab_path,
        model_path=model_path,
    ):
        prior_smiles = batch
        break

# --- (C) molecules from the trained model sampler ---
    model_smiles = None
    for batch in trainer.generate_samples(
        trained_params,
        num_samples=num_per_method,
        batch_size=num_per_method,
        seed=seed+4,
        vocab_path=vocab_path,
        model_path=model_path,
    ):
        model_smiles = batch
        break

    plot_molecule_comparison_grid(
        dataset_smiles=dataset_smiles,
        prior_smiles=prior_smiles,
        model_smiles=model_smiles,
        out_path="results/data_vs_prior_vs_model.png",
        titles=("From Data Set", "From Prior", "From Trained Model"),
    )


if __name__ == "__main__":
    plotting(num_per_method=3)
