# -*- coding: utf-8 -*-
"""
MindBridge Stage 1 — Image Reconstruction Demo
Runs on 10-12 held-out val images and saves comparison grids.

Usage:
    python reconstruct.py --subj subj01 --root /scratch/users/ampinto/mindbridge --n 12

Outputs (saved to root/reconstructions/subj/):
    vae_grid.png        — ground truth vs VAE-decoded predictions
    vd_grid.png         — ground truth vs Versatile Diffusion reconstructions (if --vd flag)
"""

import sys
sys.path = [p for p in sys.path if "share/software" not in p and "jupyterlab" not in p]

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

from paths import ensure_runtime_dirs, get_root, resolve_checkpoint
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import h5py

# =============================================================================
# MODEL DEFINITION (must match train_mindbridge_loss.py exactly)
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

    axes[0, 0].set_ylabel("Ground truth", fontsize=9, rotation=90, labelpad=8)
    axes[1, 0].set_ylabel("Reconstructed", fontsize=9, rotation=90, labelpad=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subj",  default=os.environ.get("NSD_SUBJ", "subj01"))
    parser.add_argument(
        "--root",
        default=os.environ.get("MINDBRIDGE_ROOT", "/scratch/users/ampinto/mindbridge"),
    )
    parser.add_argument("--n",     type=int, default=12, help="Number of images to reconstruct")
    parser.add_argument("--vd",    action="store_true", help="Also run Versatile Diffusion reconstruction")
    parser.add_argument("--seed",  type=int, default=42)
    args = parser.parse_args()

    root    = get_root(args.root)
    subj    = args.subj
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n       = args.n

    print(f"Subject:  {subj}")
    print(f"Device:   {device}")
    print(f"Images:   {n}")
    print(f"Root:     {root}")

    ensure_runtime_dirs(root, subj)

    os.environ["TRANSFORMERS_CACHE"] = str(root / "hf_cache")
    os.environ["HF_HOME"]            = str(root / "hf_cache")
    os.environ["TORCH_HOME"]         = str(root / "torch_cache")

    out_dir = root / "reconstructions" / subj
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load checkpoint ───────────────────────────────────────────────────────
    ckpt_path = resolve_checkpoint(root, subj)
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    n_voxels   = ckpt["n_voxels"]
    voxel_mean = ckpt["voxel_mean"]
    voxel_std  = ckpt["voxel_std"]
    print(f"Loaded checkpoint from epoch {ckpt['epoch']}, n_voxels={n_voxels}")

    model = MindBridgeMLP(n_voxels=n_voxels, d_model=256, M=16).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # ── Load val betas ────────────────────────────────────────────────────────
    betas_file = root / "nsd_meta" / f"betas_flat_{subj}.npy"
    assert betas_file.exists(), f"Betas not found: {betas_file}"
    betas_flat = np.load(betas_file)

    # Recreate the same shuffled val split as training
    np.random.seed(42)
    indices   = np.random.permutation(len(betas_flat))
    N_TRAIN   = int(0.9 * len(betas_flat))
    val_idx   = indices[N_TRAIN:]

    # Pick n evenly-spaced samples from val set
    rng       = np.random.default_rng(args.seed)
    chosen    = rng.choice(len(val_idx), size=n, replace=False)
    chosen    = np.sort(chosen)
    sample_idx = val_idx[chosen]

    val_betas  = betas_flat[sample_idx]
    val_betas  = (val_betas - voxel_mean) / voxel_std
    val_tensor = torch.tensor(val_betas, dtype=torch.float32).to(device)

    # ── Run model inference ───────────────────────────────────────────────────
    print("Running model inference...")
    with torch.no_grad():
        pred_clip, pred_dino, pred_vae = model(val_tensor)

    print(f"  pred_clip: {pred_clip.shape}")
    print(f"  pred_vae:  {pred_vae.shape}")

    # ── Load ground truth stimulus IDs ────────────────────────────────────────
    from scipy.io import loadmat
    expdesign      = loadmat(str(root / "nsd_meta" / "nsd_expdesign.mat"))
    masterordering = expdesign["masterordering"].squeeze() - 1
    stim_ids       = masterordering[:len(betas_flat)]
    sample_stim_ids = stim_ids[sample_idx]

    # ── Load ground truth images from HDF5 ───────────────────────────────────
    stim_path = root / "nsd_meta" / "nsd_stimuli.hdf5"
    assert stim_path.exists(), f"Stimuli HDF5 not found: {stim_path}"
    print("Loading ground truth images...")
    gt_imgs = []
    with h5py.File(stim_path, "r") as f:
        for sid in sample_stim_ids:
            img = Image.fromarray(f["imgBrick"][int(sid)])
            img = img.resize((256, 256), Image.LANCZOS)
            gt_imgs.append(img)

    # ── Method 1: Decode predicted VAE latents ────────────────────────────────
    print("\n[Method 1] Decoding predicted VAE latents...")
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")
    vae = vae.to(device).eval()

    with torch.no_grad():
        # SD VAE expects latents scaled by 1/0.18215
        scaled = pred_vae / 0.18215
        decoded = vae.decode(scaled).sample   # (N, 3, 512, 512)

    vae_imgs = [tensor_to_pil(decoded[i]).resize((256, 256), Image.LANCZOS)
                for i in range(n)]

    del vae
    torch.cuda.empty_cache()

    trial_labels = [f"#{i}" for i in range(n)]
    save_comparison_grid(
        gt_imgs, vae_imgs, trial_labels,
        out_dir / "vae_grid.png",
        f"MindBridge VAE decode — {subj} (n={n})"
    )

    # ── Method 2: Versatile Diffusion (optional, --vd flag) ───────────────────
    if args.vd:
        print("\n[Method 2] Running Versatile Diffusion reconstruction...")
        from diffusers import VersatileDiffusionImageVariationPipeline
        import torch

        pipe = VersatileDiffusionImageVariationPipeline.from_pretrained(
            "shi-labs/versatile-diffusion",
            torch_dtype=torch.float16,
        ).to(device)
        pipe.set_progress_bar_config(disable=True)

        # VD expects normalized CLIP embeddings, shape (1, 1, 768)
        vd_imgs = []
        clip_norm = F.normalize(pred_clip, dim=-1).half()

        for i in range(n):
            emb = clip_norm[i].unsqueeze(0).unsqueeze(0)  # (1, 1, 768)
            with torch.no_grad():
                out = pipe(
                    image_embeddings=emb,
                    num_inference_steps=20,
                    guidance_scale=7.5,
                    generator=torch.Generator(device=device).manual_seed(args.seed),
                ).images[0]
            vd_imgs.append(out.resize((256, 256), Image.LANCZOS))
            print(f"  VD {i+1}/{n} done")

        del pipe
        torch.cuda.empty_cache()

        save_comparison_grid(
            gt_imgs, vd_imgs, trial_labels,
            out_dir / "vd_grid.png",
            f"MindBridge Versatile Diffusion — {subj} (n={n})"
        )

    # ── Save individual images for slides ─────────────────────────────────────
    print("\nSaving individual images...")
    for i in range(min(n, 4)):   # save first 4 as individual files for slides
        gt_imgs[i].save(out_dir / f"gt_{i:02d}.png")
        vae_imgs[i].save(out_dir / f"vae_{i:02d}.png")

    print(f"\nAll outputs saved to: {out_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
