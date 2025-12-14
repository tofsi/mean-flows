from train_molecule import MoleculeTrainer
import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
import argparse, copy, json, os
from flax.training import checkpoints

from rdkit import RDLogger
RDLogger.logger().setLevel(RDLogger.CRITICAL)

from rdkit import Chem
from rdkit.Chem import Draw


def reconstruct_molecule(molecule_np, num_samples, output_path="results/generated_molecules.png"):
    print("Reconstructing molecules")
    
    for i in range(min(n_samples, molecule_np.shape[0])):
        plt.subplot(4, 4, i + 1)
        decoded = molecule_np[i]
        if "ERROR" not in decoded:
            mol_dec = Chem.MolFromSmiles(decoded)
            if mol_dec:
                img_dec = Draw.MolToImage(mol_dec, size=(300, 300))
                plt.imshow(img_dec)
        plt.set_title(f"Decoded: {decoded[:30]}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.show()
    plt.close()
    print(f"Saved to : {output_path}")
    
    
    

def plotting(num_samples=8):
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
    ckpt_dir = "/checkpoints_mol"
    latest = checkpoints.latest_checkpoint(ckpt_dir)
    if latest is not None:
        print(f"[run_ablation] Found existing checkpoint: {latest}")
        resume = True
    
    trainer = MoleculeTrainer(
        tp,
        checkpoint_dir=ckpt_dir,
        resume_from_checkpoints=resume,
    )
    trained_params = trainer.train()
    for mole_np in trainer.generate_samples(trained_params, num_samples=num_samples, batch_size=num_samples, seed=42):
        print(f"generated molecule shape : {mole_np.shape}") # (batch_size, B, LATENT_DIM)
        reconstruct_molecule(mole_np,num_samples)

