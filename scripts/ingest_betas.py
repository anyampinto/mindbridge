# -*- coding: utf-8 -*-
"""Fast ingestion: flatten NSD betas only (for reconstruct with saved checkpoints)."""

import sys

sys.path = [p for p in sys.path if "share/software" not in p and "jupyterlab" not in p]

import os
import h5py
import numpy as np
import nibabel as nib
from pathlib import Path
from scipy.io import loadmat

from paths import ensure_runtime_dirs, get_root
from download_nsd import ensure_nsd_data, parse_sessions, perception_beta_path

ROOT = get_root(None)
DATA_DIR = ROOT / "nsd_data"
META_DIR = ROOT / "nsd_meta"
ensure_runtime_dirs(ROOT)

SUBJ = os.environ.get("NSD_SUBJ", "subj01")
SESSIONS = parse_sessions(os.environ.get("NSD_SESSIONS", "all"), subj=SUBJ)
DOWNLOAD_STIMULI = os.environ.get("NSD_DOWNLOAD_STIMULI", "1") != "0"
DOWNLOAD_IMAGERY = os.environ.get("NSD_DOWNLOAD_IMAGERY", "0") != "0"

print(f"Betas-only ingestion for {SUBJ}")
print(f"Root: {ROOT}")
print(f"Sessions: {SESSIONS[0]}-{SESSIONS[-1]} ({len(SESSIONS)} total)")

ensure_nsd_data(
    SUBJ,
    ROOT,
    sessions=SESSIONS,
    download_stimuli=DOWNLOAD_STIMULI,
    download_imagery=DOWNLOAD_IMAGERY,
)

MASK_LOCAL = DATA_DIR / "nsddata" / "ppdata" / SUBJ / "func1pt8mm" / "roi" / "nsdgeneral.nii.gz"
EXPDESIGN_LOCAL = META_DIR / "nsd_expdesign.mat"
BETAS_FILE = META_DIR / f"betas_flat_{SUBJ}.npy"

assert EXPDESIGN_LOCAL.exists(), f"Missing experiment design: {EXPDESIGN_LOCAL}"

if MASK_LOCAL.exists():
    print(f"Loading mask from {MASK_LOCAL}...")
    mask_img = nib.load(MASK_LOCAL)
    mask_data = np.transpose(mask_img.get_fdata(), (2, 1, 0))
    mask = mask_data == 1
    print(f"  n_voxels in ROI: {mask.sum()}")
else:
    print("WARNING: nsdgeneral.nii.gz not found — using single-slice fallback mask")
    mask = np.zeros((83, 104, 81), dtype=bool)
    mask[41, :, :] = True

mask_flat = mask.reshape(-1)

if BETAS_FILE.exists():
    print(f"\nLoading cached betas from {BETAS_FILE.name}...")
    betas_flat = np.load(BETAS_FILE)
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
        betas_2d = (raw.astype(np.float32) / 300.0).reshape(750, -1)[:, mask_flat]
        all_betas.append(betas_2d)
        print(f"  Loaded session{session:02d}: {betas_2d.shape}")

    if not all_betas:
        raise FileNotFoundError(f"No beta sessions found for {SUBJ}")

    betas_flat = np.concatenate(all_betas, axis=0)
    np.save(BETAS_FILE, betas_flat)
    print(f"Betas saved to {BETAS_FILE.name}: {betas_flat.shape}")

expdesign = loadmat(str(EXPDESIGN_LOCAL))
masterordering = expdesign["masterordering"].squeeze() - 1
n_trials = betas_flat.shape[0]
print(f"Total trials: {n_trials}, betas shape: {betas_flat.shape}")
print(f"Stimuli mapped: {len(masterordering[:n_trials])} trials")
print("Betas-only ingestion complete.")
