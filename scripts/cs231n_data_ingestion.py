# -*- coding: utf-8 -*-
"""CS231n Data Ingestion - Multi-Session Production Sherlock Script"""

import sys
sys.path = [p for p in sys.path if "share/software" not in p and "jupyterlab" not in p]

import os
import gc
import h5py
import numpy as np
import nibabel as nib
from pathlib import Path
from scipy.io import loadmat

device = None  # set only when computing training targets

from paths import ensure_runtime_dirs, get_root
from download_nsd import ensure_nsd_data, parse_sessions, perception_beta_path

# =============================================================================
# PATHS
# =============================================================================
ROOT = get_root(None)
DATA_DIR = ROOT / "nsd_data"
META_DIR = ROOT / "nsd_meta"
ensure_runtime_dirs(ROOT)

# Subject is read from environment variable so the sbatch script can pass it in.
# Default to subj01 if not set.
SUBJ = os.environ.get("NSD_SUBJ", "subj01")
SESSIONS = parse_sessions(os.environ.get("NSD_SESSIONS", "all"), subj=SUBJ)
DOWNLOAD_STIMULI = os.environ.get("NSD_DOWNLOAD_STIMULI", "1") != "0"
DOWNLOAD_IMAGERY = os.environ.get("NSD_DOWNLOAD_IMAGERY", "1") != "0"
COMPUTE_TARGETS = os.environ.get("NSD_COMPUTE_TARGETS", "0") == "1"
print(f"Processing subject: {SUBJ}")
print(f"Root: {ROOT}")
print(f"Sessions: {SESSIONS[0]}-{SESSIONS[-1]} ({len(SESSIONS)} total)")
print(f"Compute targets: {COMPUTE_TARGETS}")

ensure_nsd_data(
    SUBJ,
    ROOT,
    sessions=SESSIONS,
    download_stimuli=DOWNLOAD_STIMULI,
    download_imagery=DOWNLOAD_IMAGERY,
)

MASK_LOCAL      = DATA_DIR / "nsddata" / "ppdata" / SUBJ / "func1pt8mm" / "roi" / "nsdgeneral.nii.gz"
EXPDESIGN_LOCAL = META_DIR / "nsd_expdesign.mat"
STIM_LOCAL      = META_DIR / "nsd_stimuli.hdf5"

assert EXPDESIGN_LOCAL.exists(), f"Missing experiment design: {EXPDESIGN_LOCAL}"
assert STIM_LOCAL.exists() or not DOWNLOAD_STIMULI, f"Missing stimuli HDF5: {STIM_LOCAL}"

# Per-subject output files
BETAS_FILE  = META_DIR / f"betas_flat_{SUBJ}.npy"       # (30000, n_voxels)
TARGETS_FILE = META_DIR / f"targets_{SUBJ}.npz"          # clip, dino, vae keyed by unique stim_id index

# Dimensions
D_CLIP = 768
D_DINO = 1024   # dinov2_vitl14 outputs 1024
D_VAE  = (4, 64, 64)

# =============================================================================
# LOAD MASK
# =============================================================================
if MASK_LOCAL.exists():
    print(f"Loading mask from {MASK_LOCAL}...")
    mask_img  = nib.load(MASK_LOCAL)
    mask_data = mask_img.get_fdata()
    mask_data = np.transpose(mask_data, (2, 1, 0))
    mask      = mask_data == 1
    print(f"  n_voxels in ROI: {mask.sum()}")
else:
    print("WARNING: nsdgeneral.nii.gz not found — using single-slice fallback mask")
    mask = np.zeros((83, 104, 81), dtype=bool)
    mask[41, :, :] = True
    print(f"  n_voxels (fallback): {mask.sum()}")

mask_flat = mask.reshape(-1)

# =============================================================================
# LOAD EXPERIMENT DESIGN
# =============================================================================
expdesign      = loadmat(str(EXPDESIGN_LOCAL))
masterordering = expdesign["masterordering"].squeeze() - 1  # 0-indexed

# =============================================================================
# LOAD ALL SESSION BETAS (skip if already cached)
# =============================================================================
if BETAS_FILE.exists():
    print(f"\nLoading cached betas from {BETAS_FILE.name}...")
    betas_flat = np.load(BETAS_FILE)
    print(f"  Loaded betas: {betas_flat.shape}")
else:
    print(f"\nLoading betas for {len(SESSIONS)} sessions ({SUBJ})...")
    all_betas = []
    for session in SESSIONS:
        beta_path = perception_beta_path(ROOT, SUBJ, session)
        if not beta_path.exists():
            print(f"  WARNING: session{session:02d} not found at {beta_path}, skipping")
            continue
        with h5py.File(beta_path, "r") as f:
            raw = f["/betas"][:]
        betas_4d = raw.astype(np.float32) / 300.0
        betas_2d = betas_4d.reshape(750, -1)[:, mask_flat]
        all_betas.append(betas_2d)
        print(f"  Loaded session{session:02d}: {betas_2d.shape}")

    betas_flat = np.concatenate(all_betas, axis=0)  # (30000, n_voxels)
    np.save(BETAS_FILE, betas_flat)
    print(f"Betas saved to {BETAS_FILE.name}: {betas_flat.shape}")

n_trials, n_voxels = betas_flat.shape
stim_ids_all = masterordering[:n_trials]
unique_stim_ids = np.unique(stim_ids_all)
print(f"Total trials: {len(stim_ids_all)}, Unique stimuli: {len(unique_stim_ids)}")
print(f"Total betas: {betas_flat.shape}")

# =============================================================================
# COMPUTE TARGETS FOR UNIQUE STIMULI (training only; skip for reconstruct prep)
# =============================================================================
def compute_targets(unique_stim_ids, stim_hdf5_path):
    """Extract CLIP, DINO, VAE targets for unique stimulus IDs only."""
    import ssl

    import torch
    import torch.nn.functional as F
    import torchvision.transforms as T
    from PIL import Image
    import open_clip
    from diffusers import AutoencoderKL

    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    n = len(unique_stim_ids)
    if not Path(stim_hdf5_path).exists():
        raise FileNotFoundError(f"Stimuli file not found: {stim_hdf5_path}")

    batch_size = 16

    dino_tf = T.Compose([T.Resize(224), T.CenterCrop(224), T.ToTensor(),
                          T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    vae_tf  = T.Compose([T.Resize(512), T.CenterCrop(512), T.ToTensor(),
                          T.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])

    def load_imgs(ids, i):
        with h5py.File(stim_hdf5_path, "r") as f:
            batch_ids = ids[i:i+batch_size]
            return [Image.fromarray(f["imgBrick"][int(sid)]) for sid in batch_ids]

    # --- CLIP ---
    print("Running CLIP pass...")
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="openai")
    clip_model = clip_model.to(device).eval()
    for p in clip_model.parameters(): p.requires_grad_(False)
    clip_targets = []
    for i in range(0, n, batch_size):
        imgs   = load_imgs(unique_stim_ids, i)
        clip_in = torch.stack([clip_preprocess(im) for im in imgs]).to(device)
        with torch.no_grad():
            c = F.normalize(clip_model.encode_image(clip_in), dim=-1)
        clip_targets.append(c.cpu().numpy())
        if i % 320 == 0: print(f"  CLIP {i}/{n}")
    del clip_model; gc.collect(); torch.cuda.empty_cache()

    # --- DINO ---
    print("Running DINO pass...")
    ssl._create_default_https_context = ssl._create_unverified_context
    dino_model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14")
    dino_model = dino_model.to(device).eval()
    for p in dino_model.parameters(): p.requires_grad_(False)
    dino_targets = []
    for i in range(0, n, batch_size):
        imgs    = load_imgs(unique_stim_ids, i)
        dino_in = torch.stack([dino_tf(im) for im in imgs]).to(device)
        with torch.no_grad():
            d = dino_model(dino_in)
        dino_targets.append(d.cpu().numpy())
        if i % 320 == 0: print(f"  DINO {i}/{n}")
    del dino_model; gc.collect(); torch.cuda.empty_cache()

    # --- VAE ---
    print("Running VAE pass...")
    vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")
    vae = vae.to(device).eval()
    for p in vae.parameters(): p.requires_grad_(False)
    vae_targets = []
    for i in range(0, n, batch_size):
        imgs   = load_imgs(unique_stim_ids, i)
        vae_in = torch.stack([vae_tf(im) for im in imgs]).to(device)
        with torch.no_grad():
            v = vae.encode(vae_in).latent_dist.mean
        vae_targets.append(v.cpu().numpy())
        if i % 320 == 0: print(f"  VAE {i}/{n}")
    del vae; gc.collect(); torch.cuda.empty_cache()

    return (np.concatenate(clip_targets),
            np.concatenate(dino_targets),
            np.concatenate(vae_targets))


if COMPUTE_TARGETS:
    if TARGETS_FILE.exists():
        print(f"\nLoading cached targets from {TARGETS_FILE.name}...")
        data        = np.load(TARGETS_FILE)
        clip_unique = data["clip"]
        dino_unique = data["dino"]
        vae_unique  = data["vae"]
    else:
        print(f"\nComputing targets for {len(unique_stim_ids)} unique stimuli...")
        clip_unique, dino_unique, vae_unique = compute_targets(unique_stim_ids, STIM_LOCAL)
        np.savez(TARGETS_FILE, clip=clip_unique, dino=dino_unique, vae=vae_unique,
                 unique_stim_ids=unique_stim_ids)
        print(f"Targets saved to {TARGETS_FILE.name}")

    print(f"Targets: CLIP {clip_unique.shape}, DINO {dino_unique.shape}, VAE {vae_unique.shape}")

    # Build full trial-level target arrays for training
    print("\nBuilding full trial-level target arrays...")
    stim_id_to_idx = {int(sid): i for i, sid in enumerate(unique_stim_ids)}

    clip_targets_full = np.zeros((n_trials, D_CLIP),  dtype=np.float32)
    dino_targets_full = np.zeros((n_trials, D_DINO),  dtype=np.float32)
    vae_targets_full  = np.zeros((n_trials, *D_VAE),  dtype=np.float32)

    for trial_idx, sid in enumerate(stim_ids_all[:n_trials]):
        idx = stim_id_to_idx[int(sid)]
        clip_targets_full[trial_idx] = clip_unique[idx]
        dino_targets_full[trial_idx] = dino_unique[idx]
        vae_targets_full[trial_idx]  = vae_unique[idx]

    FULL_TARGETS_FILE = META_DIR / f"targets_full_{SUBJ}.npz"
    np.savez(FULL_TARGETS_FILE,
             clip=clip_targets_full,
             dino=dino_targets_full,
             vae=vae_targets_full)
    print(f"Full trial targets saved to {FULL_TARGETS_FILE.name}")
    print(f"  CLIP {clip_targets_full.shape}, DINO {dino_targets_full.shape}, VAE {vae_targets_full.shape}")
else:
    print("\nSkipping CLIP/DINO/VAE targets (reconstruct only needs betas_flat).")

print("Ingestion complete.")
