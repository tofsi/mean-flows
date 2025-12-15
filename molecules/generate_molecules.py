from train_molecule import MoleculeTrainer
import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
import argparse, copy, json, os
from flax.training import checkpoints
import sys
sys.path.append("..")
from train import TrainingParams

from rdkit import RDLogger
RDLogger.logger().setLevel(RDLogger.CRITICAL)

from rdkit import Chem
from rdkit.Chem import Draw


def reconstruct_molecule(molecule_list, n_samples, output_path="results/generated_molecules.png"):
    print("Reconstructing molecules")
    
    for i in range(min(n_samples, len(molecule_list))):
        plt.subplot(4, 4, i + 1)
        decoded = molecule_list[i]
        if "ERROR" not in decoded:
            mol_dec = Chem.MolFromSmiles(decoded)
            if mol_dec:
                img_dec = Draw.MolToImage(mol_dec, size=(300, 300))
                plt.imshow(img_dec)
        plt.suptitle(f"Decoded: {decoded[:30]}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.show()
    plt.close()
    print(f"Saved to : {output_path}")
    
    
    

def plotting(num_samples=8, train = False):
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
    ckpt_dir = "checkpoints_mol"
    latest = checkpoints.latest_checkpoint(ckpt_dir)
    resume = True

    if latest is not None:
        print(f"[run_ablation] Found existing checkpoint: {latest}")
        resume = True

    
    trainer = MoleculeTrainer(
        trainingParams,
        checkpoint_dir=ckpt_dir,
        resume_from_checkpoints=True,
        latent_path="vae_qm9/qvae/all_latent_vectors.npz"
    )
    
    trained_params = None
    if train == True:
        trained_params = trainer.train()
    else:
        restored = trainer.load_checkpoint()
        trained_params = restored.get("params")
    for mole_np in trainer.generate_samples(trained_params, num_samples=num_samples, batch_size=num_samples, seed=42):
        print(f"generated molecule : {mole_np}") # (batch_size, B, LATENT_DIM)
        reconstruct_molecule(mole_np,num_samples)

if __name__=='__main__':
    plotting()