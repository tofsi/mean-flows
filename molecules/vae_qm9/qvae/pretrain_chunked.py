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
epochs = 3

model = JTNNVAE(vocab, hidden_size, latent_size, depth, stereo=True)
print(f"Model #Params: {sum(p.numel() for p in model.parameters())//1000}K")
model = model.cuda()

optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.9)

chunk_files = sorted(glob.glob("qvae/data/merged/train_shuffled/chunk_*.pkl"))
print(f"Found {len(chunk_files)} chunks")

os.makedirs("qvae/enc_model", exist_ok=True)

for epoch in range(epochs):
    print(f"\n=== Epoch {epoch} ===")
    skipped = 0
    
    for chunk_idx, chunk_file in enumerate(chunk_files):
        # Force cleanup before loading new chunk
        gc.collect()
        torch.cuda.empty_cache()
        
        dataset = ChunkDataset(chunk_file)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                                num_workers=0, collate_fn=lambda x: x, drop_last=True)
        
        chunk_loss = 0
        chunk_batches = 0
        pbar = tqdm(dataloader, desc=f"E{epoch}C{chunk_idx}")
        for batch in pbar:
            try:
                loss, kl, wacc, tacc, sacc, _ = model(batch, 0)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                chunk_loss += loss.item()
                chunk_batches += 1
                pbar.set_postfix({'loss': f'{loss.item():.2f}', 'skip': skipped})
            except Exception as e:
                skipped += 1
                continue
        
        # Cleanup after chunk
        del dataset, dataloader
        gc.collect()
        torch.cuda.empty_cache()
        
        avg_loss = chunk_loss / max(chunk_batches, 1)
        print(f"Chunk {chunk_idx}: loss={avg_loss:.3f}")
        
        # Checkpoint every 5 chunks
        if (chunk_idx + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'chunk': chunk_idx,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, f"qvae/enc_model/checkpoint_e{epoch}_c{chunk_idx}.pt")
            print(f"Checkpoint saved")
    
    scheduler.step()
    print(f"Epoch {epoch} done, lr={scheduler.get_last_lr()[0]:.6f}")
    torch.save(model.state_dict(), f"qvae/enc_model/model.epoch-{epoch}")

print("Done!")
