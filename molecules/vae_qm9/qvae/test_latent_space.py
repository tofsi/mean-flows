import torch
import pickle
import numpy as np
import sys
sys.path.append('.')
from jtnn import Vocab, JTNNVAE

vocab = Vocab([x.strip() for x in open('qvae/data/merged/vocab.txt')])

model = JTNNVAE(vocab, 450, 56, 3, stereo=True)
model.load_state_dict(torch.load('qvae/var_model/model.epoch-5'))
model = model.cuda()
model.eval()

with open('qvae/data/merged/train_shuffled/chunk_0.pkl', 'rb') as f:
    data = pickle.load(f)

# Get two molecules
mol1, mol2 = data[0], data[10]

# Encode both
mol_batch = [mol1]
_, tree_vec1, mol_vec1 = model.encode(mol_batch)
latent1 = torch.cat([model.T_mean(tree_vec1), model.G_mean(mol_vec1)], dim=1)

mol_batch = [mol2]
_, tree_vec2, mol_vec2 = model.encode(mol_batch)
latent2 = torch.cat([model.T_mean(tree_vec2), model.G_mean(mol_vec2)], dim=1)

print(f"Molecule 1: {mol1.smiles}")
print(f"Molecule 2: {mol2.smiles}")
print(f"\nInterpolating in latent space:\n")

# Interpolate between them
for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
    latent = (1 - alpha) * latent1 + alpha * latent2
    tree_vec = latent[:, :28]
    mol_vec = latent[:, 28:]
    
    try:
        result = model.decode(tree_vec, mol_vec, prob_decode=False)
        print(f"alpha={alpha}: {result}")
    except Exception as e:
        print(f"alpha={alpha}: Error - {e}")
