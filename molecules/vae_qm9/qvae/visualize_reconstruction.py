import torch
import pickle
import sys
sys.path.append('.')

from rdkit import RDLogger
RDLogger.logger().setLevel(RDLogger.CRITICAL)

from rdkit import Chem
from rdkit.Chem import Draw
import matplotlib.pyplot as plt

from jtnn import Vocab, JTNNVAE

# Load model
vocab = Vocab([x.strip() for x in open('qvae/data/merged/vocab.txt')])
model = JTNNVAE(vocab, 450, 56, 3, stereo=True)
model.load_state_dict(torch.load('qvae/var_model/model.epoch-15'))
model = model.cuda()
model.eval()

# Load some molecules
with open('qvae/data/merged/train_shuffled/chunk_0.pkl', 'rb') as f:
    data = pickle.load(f)

# Test 10 samples
n_samples = 10
results = []

print("=" * 60)
print("ENCODER INPUT → DECODER OUTPUT")
print("=" * 60)

for i in range(n_samples):
    mol_tree = data[i]
    original = mol_tree.smiles
    
    try:
        # Encode
        mol_batch = [mol_tree]
        tree_mess, tree_vec, mol_vec = model.encode(mol_batch)
        tree_mean = model.T_mean(tree_vec)
        mol_mean = model.G_mean(mol_vec)
        
        # Get latent vector
        latent = torch.cat([tree_mean, mol_mean], dim=1)
        latent_str = f"shape={latent.shape}, mean={latent.mean().item():.3f}, std={latent.std().item():.3f}"
        
        # Decode
        decoded = model.decode(tree_mean, mol_mean, prob_decode=False)
        
        match = "✓ MATCH" if decoded == original else "✗ DIFFERENT"
        
    except Exception as e:
        decoded = f"ERROR: {e}"
        latent_str = "N/A"
        match = "✗ ERROR"
    
    results.append({
        'original': original,
        'decoded': decoded,
        'match': match,
        'latent': latent_str
    })
    
    print(f"\nSample {i+1}:")
    print(f"  Input (original):  {original}")
    print(f"  Latent vector:     {latent_str}")
    print(f"  Output (decoded):  {decoded}")
    print(f"  Result:            {match}")

print("\n" + "=" * 60)

# Count matches
matches = sum(1 for r in results if "MATCH" in r['match'])
print(f"Accuracy: {matches}/{n_samples} = {matches/n_samples*100:.0f}%")

# Save results to file
with open('qvae/reconstruction_examples.txt', 'w') as f:
    f.write("ENCODER INPUT → DECODER OUTPUT\n")
    f.write("=" * 60 + "\n\n")
    
    for i, r in enumerate(results):
        f.write(f"Sample {i+1}:\n")
        f.write(f"  Input (original):  {r['original']}\n")
        f.write(f"  Latent vector:     {r['latent']}\n")
        f.write(f"  Output (decoded):  {r['decoded']}\n")
        f.write(f"  Result:            {r['match']}\n\n")
    
    f.write(f"Accuracy: {matches}/{n_samples} = {matches/n_samples*100:.0f}%\n")

print("\nSaved to qvae/reconstruction_examples.txt")

# Create molecule visualization
print("\nCreating molecule images...")

fig, axes = plt.subplots(n_samples, 2, figsize=(10, 3*n_samples))
fig.suptitle('Original (left) vs Reconstructed (right)', fontsize=14)

for i, r in enumerate(results):
    # Original
    mol_orig = Chem.MolFromSmiles(r['original'])
    if mol_orig:
        img_orig = Draw.MolToImage(mol_orig, size=(300, 300))
        axes[i, 0].imshow(img_orig)
    axes[i, 0].set_title(f"Original: {r['original'][:30]}...")
    axes[i, 0].axis('off')
    
    # Decoded
    if "ERROR" not in r['decoded']:
        mol_dec = Chem.MolFromSmiles(r['decoded'])
        if mol_dec:
            img_dec = Draw.MolToImage(mol_dec, size=(300, 300))
            axes[i, 1].imshow(img_dec)
    axes[i, 1].set_title(f"Decoded: {r['decoded'][:30]}... {r['match']}")
    axes[i, 1].axis('off')

plt.tight_layout()
plt.savefig('qvae/reconstruction_examples.png', dpi=150)
print("Saved to qvae/reconstruction_examples.png")
