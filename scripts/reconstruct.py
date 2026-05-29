# -*- coding: utf-8 -*-
"""
MindBridge Stage 1 — Image Reconstruction Demo
Runs on held-out val images and saves comparison grids.

Modal usage:
    modal run modal_app.py --mode reconstruct --subj subj01   # single subject
    modal run modal_app.py --mode reconstruct-all             # all 8 subjects, 2 images each

Local usage:
    python reconstruct.py --subj subj01 --root /mnt/mindbridge --n 2

Outputs (saved to root/reconstructions/subj/):
    vae_grid_subj01.png  — ground truth vs VAE-decoded predictions for that subject
    vd_grid_subj01.png   — Versatile Diffusion reconstructions (--vd flag)
    gt_00.png ... gt_01.png, vae_00.png ... vae_01.png  — individual images for slides
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import h5py

# =============================================================================
# MODEL DEFINITION (must match train_mindbridge.py exactly)
# =============================================================================
class BrainMLP(nn.Module):
    def __init__(self, n_voxels, d_model=512, M=16, dropout=0.5):
        super().__init__()
        self.M, self.d_model = M, d_model
        self.net = nn.Sequential(
            nn.Linear(n_voxels, 4096), nn.LayerNorm(4096), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(4096, 4096),     nn.LayerNorm(4096), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(4096, M * d_model),
        )
    def forward(self, x):
        return self.net(x).view(x.size(0), self.M, self.d_model)


class LatentQueryDecoder(nn.Module):
    def __init__(self, d_model=512, M=16, n_heads=8, K=2, dropout=0.1):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(1, M, d_model) * 0.02)
        self.layers  = nn.ModuleList([
            nn.ModuleDict({
                "self_attn":  nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True),
                "cross_attn": nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True),
                "ffn":  nn.Sequential(nn.Linear(d_model, d_model*4), nn.GELU(), nn.Linear(d_model*4, d_model)),
                "norm1": nn.LayerNorm(d_model), "norm2": nn.LayerNorm(d_model), "norm3": nn.LayerNorm(d_model),
            }) for _ in range(K)
        ])
    def forward(self, brain_tokens):
        B = brain_tokens.size(0)
        q = self.queries.expand(B, -1, -1)
        for layer in self.layers:
            q2, _ = layer["self_attn"](q, q, q);               q = layer["norm1"](q + q2)
            q2, _ = layer["cross_attn"](q, brain_tokens, brain_tokens); q = layer["norm2"](q + q2)
            q     = layer["norm3"](q + layer["ffn"](q))
        return q


class MindBridgeMLP(nn.Module):
    def __init__(self, n_voxels, d_model=512, M=16, d_clip=768, d_dino=1024,
                 vae_shape=(4, 64, 64), dropout=0.5):
        super().__init__()
        self.encoder   = BrainMLP(n_voxels, d_model, M, dropout)
        self.decoder   = LatentQueryDecoder(d_model, M, n_heads=8, K=2)
        self.clip_head = nn.Linear(d_model, d_clip)
        self.dino_head = nn.Linear(d_model, d_dino)
        self.vae_head  = nn.Sequential(nn.Linear(d_model*M, vae_shape[0]*vae_shape[1]*vae_shape[2]))
        self.vae_shape = vae_shape
        self.M         = M
    def forward(self, x):
        brain_tokens  = self.encoder(x)
        visual_tokens = self.decoder(brain_tokens)
        pooled        = visual_tokens.mean(dim=1)
        pred_clip     = self.clip_head(pooled)
        pred_dino     = self.dino_head(pooled)
        flat          = visual_tokens.reshape(visual_tokens.size(0), -1)
        pred_vae      = self.vae_head(flat).view(-1, *self.vae_shape)
        return pred_clip, pred_dino, pred_vae


# =============================================================================
# HELPERS
# =============================================================================
def tensor_to_pil(t):
    """Convert a (C, H, W) tensor in [-1, 1] to a PIL Image."""
    t = t.clamp(-1, 1)
    t = (t + 1) / 2
    t = (t * 255).byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(t)


def save_comparison_grid(gt_imgs, pred_imgs, labels, save_path, title):
    """Save a 2-row grid: top = ground truth, bottom = reconstructed."""
    n = len(gt_imgs)
    fig, axes = plt.subplots(2, n, figsize=(n * 2.5, 6))
    fig.suptitle(title, fontsize=13, y=1.01)
    for i in range(n):
        axes[0, i].imshow(gt_imgs[i])
        axes[0, i].set_title(f"trial {labels[i]}", fontsize=8)
        axes[0, i].axis("off")
        axes[1, i].imshow(pred_imgs[i])
        axes[1, i].axis("off")
    axes[0, 0].set_ylabel("Ground truth",  fontsize=9, rotation=90, labelpad=8)
    axes[1, 0].set_ylabel("Reconstructed", fontsize=9, rotation=90, labelpad=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


# =============================================================================
# CORE RECONSTRUCTION LOGIC (callable from modal_app.py or __main__)
# =============================================================================
def run_reconstruction(subj: str = "subj01",
                       root: Path = Path("/mnt/mindbridge"),
                       n: int = 12,
                       run_vd: bool = False,
                       seed: int = 42):
    """
    Reconstruct n held-out val images for one subject.
    Val split matches train_mindbridge.py exactly: first 27000 = train, last 3000 = val.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Subject:  {subj}")
    print(f"Device:   {device}")
    print(f"Images:   {n}")

    # Point HF / torch cache at the volume so weights are only downloaded once
    os.environ["TRANSFORMERS_CACHE"] = str(root / "hf_cache")
    os.environ["HF_HOME"]            = str(root / "hf_cache")
    os.environ["TORCH_HOME"]         = str(root / "torch_cache")

    out_dir = root / "reconstructions"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load checkpoint ───────────────────────────────────────────────────────
    ckpt_path = root / "checkpoints" / subj / "best_stage1.pt"
    assert ckpt_path.exists(), f"Checkpoint not found: {ckpt_path}"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    n_voxels   = ckpt["n_voxels"]
    voxel_mean = ckpt["voxel_mean"]   # (1, n_voxels) numpy array saved during training
    voxel_std  = ckpt["voxel_std"]
    print(f"Loaded checkpoint: epoch {ckpt['epoch']}, n_voxels={n_voxels}")

    model = MindBridgeMLP(n_voxels=n_voxels, d_model=256, M=16).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # ── Load betas and apply the exact same val split as training ─────────────
    # train_mindbridge.py: N_TRAIN = 27000, sequential split (no shuffle)
    # DO NOT use np.random.permutation here — that was the original bug.
    betas_file = root / "nsd_meta" / f"betas_flat_{subj}.npy"
    assert betas_file.exists(), f"Betas not found: {betas_file}"
    betas_flat = np.load(betas_file)

    N_TRAIN  = 27000
    val_betas_full = betas_flat[N_TRAIN:]          # shape (3000, n_voxels)
    val_global_idx = np.arange(N_TRAIN, len(betas_flat))  # indices into betas_flat

    assert len(val_betas_full) >= n, (
        f"Requested {n} val images but val set only has {len(val_betas_full)} trials."
    )

    # Pick n random samples from val (reproducible)
    rng    = np.random.default_rng(seed)
    chosen = np.sort(rng.choice(len(val_betas_full), size=n, replace=False))  # indices into val set

    sample_val_betas = val_betas_full[chosen]                    # (n, n_voxels)
    sample_global_idx = val_global_idx[chosen]                   # indices into betas_flat (for stim_ids)

    # Normalize using training stats from checkpoint
    sample_val_betas = (sample_val_betas - voxel_mean) / voxel_std
    val_tensor = torch.tensor(sample_val_betas, dtype=torch.float32).to(device)

    # ── Run model inference ───────────────────────────────────────────────────
    print("Running model inference...")
    with torch.no_grad():
        pred_clip, pred_dino, pred_vae = model(val_tensor)
    print(f"  pred_clip: {pred_clip.shape}")
    print(f"  pred_vae:  {pred_vae.shape}")

    # ── Map trial indices → stimulus IDs ─────────────────────────────────────
    # masterordering is 1-indexed; subtract 1 for 0-indexed HDF5 lookup
    from scipy.io import loadmat
    expdesign       = loadmat(str(root / "nsd_meta" / "nsd_expdesign.mat"))
    masterordering  = expdesign["masterordering"].squeeze() - 1  # (73000,), 0-indexed
    # Each NSD trial maps to a stimulus; we only used the first len(betas_flat) trials
    stim_ids        = masterordering[:len(betas_flat)]           # (30000,)
    sample_stim_ids = stim_ids[sample_global_idx]                # (n,) — correct indexing

    # ── Load ground truth images ──────────────────────────────────────────────
    stim_path = root / "nsd_meta" / "nsd_stimuli.hdf5"
    assert stim_path.exists(), f"Stimuli HDF5 not found: {stim_path}"
    print("Loading ground truth images...")
    gt_imgs = []
    with h5py.File(stim_path, "r") as f:
        for sid in sample_stim_ids:
            img = Image.fromarray(f["imgBrick"][int(sid)])
            img = img.resize((256, 256), Image.LANCZOS)
            gt_imgs.append(img)

    trial_labels = [f"#{int(i)}" for i in sample_global_idx]

    # ── Method 1: Decode predicted VAE latents ────────────────────────────────
    print("\n[Method 1] Decoding predicted VAE latents...")
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")
    vae = vae.to(device).eval()

    with torch.no_grad():
        # SD VAE expects latents divided by the scaling factor 0.18215
        scaled  = pred_vae / 0.18215
        decoded = vae.decode(scaled).sample   # (n, 3, 512, 512)

    vae_imgs = [tensor_to_pil(decoded[i]).resize((256, 256), Image.LANCZOS)
                for i in range(n)]

    del vae
    torch.cuda.empty_cache()

    save_comparison_grid(
        gt_imgs, vae_imgs, trial_labels,
        out_dir / f"vae_grid_{subj}.png",
        f"MindBridge VAE decode — {subj} (n={n})"
    )

    # ── Method 2: Versatile Diffusion (optional) ──────────────────────────────
    if run_vd:
        print("\n[Method 2] Running Versatile Diffusion reconstruction...")
        from diffusers import VersatileDiffusionImageVariationPipeline

        pipe = VersatileDiffusionImageVariationPipeline.from_pretrained(
            "shi-labs/versatile-diffusion",
            torch_dtype=torch.float16,
        ).to(device)
        pipe.set_progress_bar_config(disable=True)

        vd_imgs   = []
        clip_norm = F.normalize(pred_clip, dim=-1).half()
        for i in range(n):
            emb = clip_norm[i].unsqueeze(0).unsqueeze(0)   # (1, 1, 768)
            with torch.no_grad():
                out = pipe(
                    image_embeddings=emb,
                    num_inference_steps=20,
                    guidance_scale=7.5,
                    generator=torch.Generator(device=device).manual_seed(seed),
                ).images[0]
            vd_imgs.append(out.resize((256, 256), Image.LANCZOS))
            print(f"  VD {i+1}/{n} done")

        del pipe
        torch.cuda.empty_cache()

        save_comparison_grid(
            gt_imgs, vd_imgs, trial_labels,
            out_dir / f"vd_grid_{subj}.png",
            f"MindBridge Versatile Diffusion — {subj} (n={n})"
        )

    # ── Save individual images for slides (first 4) ───────────────────────────
    print("\nSaving individual images...")
    for i in range(min(n, 4)):
        gt_imgs[i].save(out_dir / f"gt_{subj}_{i:02d}.png")
        vae_imgs[i].save(out_dir / f"vae_{subj}_{i:02d}.png")

    print(f"\nAll outputs saved to: {out_dir}")
    print("Done.")


# =============================================================================
# LOCAL ENTRYPOINT
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subj", default=os.environ.get("NSD_SUBJ", "subj01"))
    parser.add_argument("--root", default="/mnt/mindbridge")
    parser.add_argument("--n",    type=int, default=12)
    parser.add_argument("--vd",   action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_reconstruction(
        subj=args.subj,
        root=Path(args.root),
        n=args.n,
        run_vd=args.vd,
        seed=args.seed,
    )
