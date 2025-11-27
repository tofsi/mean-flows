import numpy as np
import jax.numpy as jnp
from tqdm import tqdm

import fid  # your JAX FID file
from prepare_imagenet import get_dataloaders_extracted  # or your loader


def main():
    # 1. Build dataloaders (we only need val)
    train_loader, val_loader = get_dataloaders_extracted(
        root_dir="./imagenet",
        batch_size=64,
        num_workers=0,  # avoid fork/JAX issues
    )

    # 2. JAX FID feature extractor
    fid_extract = fid.make_fid_feature_extractor()

    feats_real = []

    # 3. Iterate over val, handling both (images, labels) and images-only
    for batch in tqdm(val_loader, desc="Extracting Inception features (real)"):
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            images, _ = batch
        else:
            images = batch

        # images: torch tensor (B,3,H,W), typically in [0,1]
        images_np = images.permute(0, 2, 3, 1).numpy().astype("float32")  # (B,H,W,3)

        # JAX: (B,H,W,3) -> (B,2048)
        feats = fid_extract(jnp.asarray(images_np))
        feats_real.append(np.array(feats))

    feats_real = np.concatenate(feats_real, axis=0)  # (N,2048)

    # 4. Compute stats in JAX
    x = jnp.asarray(feats_real)
    mu = jnp.mean(x, axis=0)
    xc = x - mu
    cov = (xc.T @ xc) / (x.shape[0] - 1)

    # 5. Save NPZ
    np.savez("imagenet_val_fid_stats_jax.npz", mu=np.array(mu), cov=np.array(cov))

    print("Saved FID stats to imagenet_val_fid_stats_jax.npz")
    print("mu shape:", mu.shape, "cov shape:", cov.shape)


if __name__ == "__main__":
    main()
