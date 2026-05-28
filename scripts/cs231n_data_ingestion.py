# -*- coding: utf-8 -*-
"""CS231n Data Ingestion — Modal-compatible"""

import os
import gc
import h5py
import torch
import numpy as np
import nibabel as nib
import torchvision.transforms as T
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from scipy.io import loadmat

import open_clip
from diffusers import AutoencoderKL

# Dimensions (module-level constants are fine — no side effects)
D_CLIP = 768
D_DINO = 1024
D_VAE  = (4, 64, 64)


def run_ingestion(subj: str = "subj01", root: Path = Path("/mnt/mindbridge")):
    """
    Full ingestion pipeline for one subject.
    Reads raw NSD data from `root`, writes cached outputs back to `root`.
    Resume-safe: every expensive step is checkpointed individually so a
    disconnect never causes work to be repeated.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Processing subject: {subj}")

    # ── Paths ─────────────────────────────────────────────────────────────────
    # Volume layout (flat — no per-subject subdirectories):
    #   betas/betas_session01.hdf5 ... betas_session40.hdf5
    #   rois/nsdgeneral.nii.gz
    #   nsd_meta/nsd_expdesign.mat
    #   nsd_meta/nsd_stimuli.hdf5
    meta_dir = root / "nsd_meta"
    meta_dir.mkdir(exist_ok=True, parents=True)

    mask_path    = root / "rois" / "nsdgeneral.nii.gz"
    expdesign    = meta_dir / "nsd_expdesign.mat"
    stim_path    = meta_dir / "nsd_stimuli.hdf5"
    betas_file   = meta_dir / f"betas_flat_{subj}.npy"
    targets_file = meta_dir / f"targets_{subj}.npz"
    full_targets = meta_dir / f"targets_full_{subj}.npz"

    # Per-session beta cache dir (resume point for mid-loop crashes)
    sess_cache_dir = meta_dir / f"sess_betas_{subj}"
    sess_cache_dir.mkdir(exist_ok=True, parents=True)

    # Per-model target cache files (resume point if one model pass crashes)
    clip_cache = meta_dir / f"clip_targets_{subj}.npy"
    dino_cache = meta_dir / f"dino_targets_{subj}.npy"
    vae_cache  = meta_dir / f"vae_targets_{subj}.npy"

    # ── Mask ──────────────────────────────────────────────────────────────────
    if mask_path.exists():
        print(f"Loading mask from {mask_path}...")
        mask_img  = nib.load(mask_path)
        mask_data = np.transpose(mask_img.get_fdata(), (2, 1, 0))
        mask      = mask_data == 1
        print(f"  n_voxels in ROI: {mask.sum()}")
    else:
        print("WARNING: nsdgeneral.nii.gz not found — using fallback mask")
        mask = np.zeros((83, 104, 81), dtype=bool)
        mask[41, :, :] = True
        print(f"  n_voxels (fallback): {mask.sum()}")

    mask_flat = mask.reshape(-1)

    # ── Experiment design ─────────────────────────────────────────────────────
    ed              = loadmat(str(expdesign))
    masterordering  = ed["masterordering"].squeeze() - 1   # 0-indexed
    stim_ids_all    = masterordering[:30000]
    unique_stim_ids = np.unique(stim_ids_all)
    print(f"Total trials: {len(stim_ids_all)}, Unique stimuli: {len(unique_stim_ids)}")

    # ── Betas ─────────────────────────────────────────────────────────────────
    if betas_file.exists():
        print(f"\nLoading cached betas from {betas_file.name}...")
        betas_flat = np.load(betas_file)
        print(f"  Loaded betas: {betas_flat.shape}")
    else:
        print(f"\nLoading all 40 sessions for {subj}...")
        print(f"  Searching under: {root / 'betas'}")
        print(f"  Per-session cache: {sess_cache_dir}")
        all_betas = []
        for i in range(1, 41):
            sess      = f"session{str(i).zfill(2)}"
            sess_file = sess_cache_dir / f"{sess}.npy"

            # Already cached from a previous (possibly interrupted) run
            if sess_file.exists():
                betas_2d = np.load(sess_file)
                all_betas.append(betas_2d)
                print(f"  Resumed {sess} from cache: {betas_2d.shape}")
                continue

            beta_path = root / "betas" / f"betas_{sess}.hdf5"
            if not beta_path.exists():
                print(f"  WARNING: {sess} not found at {beta_path}")
                continue
            with h5py.File(beta_path, "r") as f:
                raw = f["/betas"][:]
            betas_2d = (raw.astype(np.float32) / 300.0).reshape(750, -1)[:, mask_flat]
            np.save(sess_file, betas_2d)   # cache immediately after loading
            all_betas.append(betas_2d)
            print(f"  Loaded {sess}: {betas_2d.shape}")

        if not all_betas:
            raise FileNotFoundError(
                f"No beta sessions found for {subj}.\n"
                f"Expected files at: {root / 'betas' / 'betas_session01.hdf5'}\n"
                f"Check your volume upload with: modal volume ls mindbridge-data betas"
            )

        betas_flat = np.concatenate(all_betas, axis=0)
        np.save(betas_file, betas_flat)
        print(f"Betas saved to {betas_file.name}: {betas_flat.shape}")

    n_trials, n_voxels = betas_flat.shape
    print(f"Total betas: {betas_flat.shape}")

    # ── Per-stimulus targets (CLIP / DINO / VAE) ──────────────────────────────
    if targets_file.exists():
        print(f"\nLoading cached targets from {targets_file.name}...")
        td          = np.load(targets_file)
        clip_unique = td["clip"]
        dino_unique = td["dino"]
        vae_unique  = td["vae"]
    else:
        print(f"\nComputing targets for {len(unique_stim_ids)} unique stimuli...")
        clip_unique, dino_unique, vae_unique = _compute_targets(
            unique_stim_ids, stim_path, device,
            clip_cache=clip_cache, dino_cache=dino_cache, vae_cache=vae_cache)
        np.savez(targets_file, clip=clip_unique, dino=dino_unique,
                 vae=vae_unique, unique_stim_ids=unique_stim_ids)
        print(f"Targets saved to {targets_file.name}")

    print(f"Targets: CLIP {clip_unique.shape}, DINO {dino_unique.shape}, VAE {vae_unique.shape}")

    # ── Expand to trial-level arrays ──────────────────────────────────────────
    if full_targets.exists():
        print(f"\nFull trial targets already exist, skipping expansion.")
    else:
        print("\nBuilding full trial-level target arrays...")
        stim_id_to_idx    = {int(sid): i for i, sid in enumerate(unique_stim_ids)}
        clip_targets_full = np.zeros((n_trials, D_CLIP),  dtype=np.float32)
        dino_targets_full = np.zeros((n_trials, D_DINO),  dtype=np.float32)
        vae_targets_full  = np.zeros((n_trials, *D_VAE),  dtype=np.float32)

        for trial_idx, sid in enumerate(stim_ids_all[:n_trials]):
            idx = stim_id_to_idx[int(sid)]
            clip_targets_full[trial_idx] = clip_unique[idx]
            dino_targets_full[trial_idx] = dino_unique[idx]
            vae_targets_full[trial_idx]  = vae_unique[idx]

        np.savez(full_targets,
                 clip=clip_targets_full,
                 dino=dino_targets_full,
                 vae=vae_targets_full)
        print(f"Full trial targets saved to {full_targets.name}")
        print(f"  CLIP {clip_targets_full.shape}, DINO {dino_targets_full.shape}, VAE {vae_targets_full.shape}")

    if torch.cuda.is_available():
        print(f"Available GPU VRAM: {torch.cuda.mem_get_info()[0]/1e9:.1f} GB")
    print("Ingestion complete.")


def _compute_targets(unique_stim_ids, stim_hdf5_path, device,
                     clip_cache=None, dino_cache=None, vae_cache=None):
    """
    Extract CLIP, DINO, VAE embeddings for every unique stimulus ID.
    Each model pass is cached individually — if a crash happens after CLIP
    but before DINO, only DINO and VAE will re-run on the next attempt.
    """
    n          = len(unique_stim_ids)
    batch_size = 16

    if not Path(stim_hdf5_path).exists():
        raise FileNotFoundError(f"Stimuli file not found: {stim_hdf5_path}")

    dino_tf = T.Compose([T.Resize(224), T.CenterCrop(224), T.ToTensor(),
                          T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    vae_tf  = T.Compose([T.Resize(512), T.CenterCrop(512), T.ToTensor(),
                          T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    def load_imgs(ids, i):
        with h5py.File(stim_hdf5_path, "r") as f:
            return [Image.fromarray(f["imgBrick"][int(sid)]) for sid in ids[i:i + batch_size]]

    # ── CLIP ──────────────────────────────────────────────────────────────────
    if clip_cache and Path(clip_cache).exists():
        print("CLIP targets already cached, loading...")
        clip_targets_arr = np.load(clip_cache)
    else:
        print("Running CLIP pass...")
        clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
            "ViT-L-14", pretrained="openai")
        clip_model = clip_model.to(device).eval()
        for p in clip_model.parameters():
            p.requires_grad_(False)
        clip_targets = []
        for i in range(0, n, batch_size):
            imgs    = load_imgs(unique_stim_ids, i)
            clip_in = torch.stack([clip_preprocess(im) for im in imgs]).to(device)
            with torch.no_grad():
                c = F.normalize(clip_model.encode_image(clip_in), dim=-1)
            clip_targets.append(c.cpu().numpy())
            if i % 320 == 0:
                print(f"  CLIP {i}/{n}")
        del clip_model; gc.collect(); torch.cuda.empty_cache()
        clip_targets_arr = np.concatenate(clip_targets)
        if clip_cache:
            np.save(clip_cache, clip_targets_arr)
            print(f"  CLIP targets cached to {Path(clip_cache).name}")

    # ── DINO ──────────────────────────────────────────────────────────────────
    if dino_cache and Path(dino_cache).exists():
        print("DINO targets already cached, loading...")
        dino_targets_arr = np.load(dino_cache)
    else:
        print("Running DINO pass...")
        dino_model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14")
        dino_model = dino_model.to(device).eval()
        for p in dino_model.parameters():
            p.requires_grad_(False)
        dino_targets = []
        for i in range(0, n, batch_size):
            imgs    = load_imgs(unique_stim_ids, i)
            dino_in = torch.stack([dino_tf(im) for im in imgs]).to(device)
            with torch.no_grad():
                d = dino_model(dino_in)
            dino_targets.append(d.cpu().numpy())
            if i % 320 == 0:
                print(f"  DINO {i}/{n}")
        del dino_model; gc.collect(); torch.cuda.empty_cache()
        dino_targets_arr = np.concatenate(dino_targets)
        if dino_cache:
            np.save(dino_cache, dino_targets_arr)
            print(f"  DINO targets cached to {Path(dino_cache).name}")

    # ── VAE ───────────────────────────────────────────────────────────────────
    if vae_cache and Path(vae_cache).exists():
        print("VAE targets already cached, loading...")
        vae_targets_arr = np.load(vae_cache)
    else:
        print("Running VAE pass...")
        vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")
        vae = vae.to(device).eval()
        for p in vae.parameters():
            p.requires_grad_(False)
        vae_targets = []
        for i in range(0, n, batch_size):
            imgs   = load_imgs(unique_stim_ids, i)
            vae_in = torch.stack([vae_tf(im) for im in imgs]).to(device)
            with torch.no_grad():
                v = vae.encode(vae_in).latent_dist.mean
            vae_targets.append(v.cpu().numpy())
            if i % 320 == 0:
                print(f"  VAE {i}/{n}")
        del vae; gc.collect(); torch.cuda.empty_cache()
        vae_targets_arr = np.concatenate(vae_targets)
        if vae_cache:
            np.save(vae_cache, vae_targets_arr)
            print(f"  VAE targets cached to {Path(vae_cache).name}")

    return clip_targets_arr, dino_targets_arr, vae_targets_arr


if __name__ == "__main__":
    # Local test run (still works without Modal)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--subj", default="subj01")
    parser.add_argument("--root", default="/mnt/mindbridge")
    args = parser.parse_args()
    run_ingestion(subj=args.subj, root=Path(args.root))
