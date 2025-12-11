import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pickle
import glob
from tqdm import tqdm
import os
import gc
import sys
sys.path.append('.')
from jtnn import *

class ChunkDataset(Dataset):
    def __init__(self, chunk_file):
        with open(chunk_file, 'rb') as f:
            self.data = pickle.load(f)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

# Config
vocab = Vocab([x.strip() for x in open('qvae/data/merged/vocab.txt')])
hidden_size, latent_size, depth = 450, 56, 3
batch_size = 64
beta = 0.005

model = JTNNVAE(vocab, hidden_size, latent_size, depth, stereo=True)
print(f"Model #Params: {sum(p.numel() for p in model.parameters())//1000}K")

# Load from best model so far
model.load_state_dict(torch.load('qvae/var_model/model.epoch-5'))
print("Loaded model.epoch-5")

model = model.cuda()

optimizer = optim.Adam(model.parameters(), lr=0.0003)  # Lower LR for fine-tuning
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.95)  # Slower decay

chunk_files = sorted(glob.glob("qvae/data/merged/train_shuffled/chunk_*.pkl"))
print(f"Found {len(chunk_files)} chunks")

os.makedirs("qvae/var_model", exist_ok=True)

start_epoch = 6
total_epochs = 26  # Train epochs 6-25 (20 more epochs)

for epoch in range(start_epoch, total_epochs):
    print(f"\n=== Epoch {epoch} ===")
    skipped = 0
    epoch_loss = 0
    epoch_batches = 0
    
    for chunk_idx, chunk_file in enumerate(chunk_files):
        gc.collect()
        torch.cuda.empty_cache()
        
        dataset = ChunkDataset(chunk_file)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                                num_workers=0, collate_fn=lambda x: x, drop_last=True)
        
        pbar = tqdm(dataloader, desc=f"E{epoch}C{chunk_idx}")
        for batch in pbar:
            try:
                loss, kl, wacc, tacc, sacc, _ = model(batch, beta)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                epoch_batches += 1
                pbar.set_postfix({'loss': f'{loss.item():.2f}', 'kl': f'{kl:.1f}', 'skip': skipped})
            except Exception as e:
                skipped += 1
                continue
        
        del dataset, dataloader
        gc.collect()
        torch.cuda.empty_cache()
        
        # Overwrite checkpoint every 5 chunks
        if (chunk_idx + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'chunk': chunk_idx,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, f"qvae/var_model/checkpoint_latest.pt")
    
    scheduler.step()
    avg_loss = epoch_loss / max(epoch_batches, 1)
    print(f"Epoch {epoch} done: loss={avg_loss:.3f}, skipped={skipped}, lr={scheduler.get_last_lr()[0]:.6f}")
    
    # Save model every 5 epochs
    if epoch % 5 == 0 or epoch == total_epochs - 1:
        torch.save(model.state_dict(), f"qvae/var_model/model.epoch-{epoch}")
        print(f"*** Saved model.epoch-{epoch} ***")

print("\nTraining complete!")
print("Saved models: model.epoch-10, model.epoch-15, model.epoch-20, model.epoch-25")
