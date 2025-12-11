import torch
import pickle
from tqdm import tqdm
import sys
sys.path.append('.')
from jtnn import Vocab, JTNNVAE

vocab = Vocab([x.strip() for x in open('qvae/data/merged/vocab.txt')])

# Load test data (use first chunk, first 500 molecules)
with open('qvae/data/merged/train_shuffled/chunk_0.pkl', 'rb') as f:
    test_data = pickle.load(f)[:500]

print(f"Testing on {len(test_data)} molecules\n")

models_to_test = [
    'qvae/var_model/model.epoch-2',
    'qvae/var_model/model.epoch-3',
    'qvae/var_model/model.epoch-4',
    'qvae/var_model/model.epoch-5',
]

results = {}

for model_path in models_to_test:
    print(f"\n=== Testing {model_path} ===")
    
    model = JTNNVAE(vocab, 450, 56, 3, stereo=True)
    try:
        model.load_state_dict(torch.load(model_path))
    except:
        print(f"Could not load {model_path}, skipping")
        continue
    model = model.cuda()
    model.eval()
    
    correct = 0
    total = 0
    
    for mol_tree in tqdm(test_data, desc="Reconstructing"):
        try:
            mol_batch = [mol_tree]
            tree_mess, tree_vec, mol_vec = model.encode(mol_batch)
            tree_mean = model.T_mean(tree_vec)
            mol_mean = model.G_mean(mol_vec)
            
            result = model.decode(tree_mean, mol_mean, prob_decode=False)
            
            if result == mol_tree.smiles:
                correct += 1
            total += 1
        except:
            continue
    
    accuracy = correct / total * 100 if total > 0 else 0
    results[model_path] = accuracy
    print(f"Accuracy: {correct}/{total} = {accuracy:.1f}%")

print("\n=== SUMMARY ===")
for model_path, acc in sorted(results.items(), key=lambda x: -x[1]):
    print(f"{model_path}: {acc:.1f}%")
