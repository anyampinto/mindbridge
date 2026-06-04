# -*- coding: utf-8 -*-
"""Trial-level NSD perception betas ↔ 73k VDVAE latents (31-layer flat)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import loadmat

from vdvae_brain_diffuser import NUM_LATENT_LAYERS, layer_flat_dims


def load_expdesign(root: Path) -> dict:
    path = root / "nsd_meta" / "nsd_expdesign.mat"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    return loadmat(str(path))


def trial_image_ids_73k(subj: str, n_trials: int, expdesign: dict) -> np.ndarray:
    """Map each row in betas_flat to a 0-based imgBrick index (73k space)."""
    subj_idx = int(subj.replace("subj", "")) - 1
    ordering_10k = expdesign["masterordering"].squeeze().astype(np.int64) - 1
    subjectim = expdesign["subjectim"][subj_idx, :].astype(np.int64) - 1
    return subjectim[ordering_10k[:n_trials]]


def load_trial_vdvae_targets(
    subj: str,
    root: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
  Returns (betas, latents, image_id_73k) aligned on perception trials.

  latents: (n_trials, D) float32 — all 31 VDVAE layers concatenated.
  """
    meta_dir = root / "nsd_meta"
    betas_path = meta_dir / f"betas_flat_{subj}.npy"
    lat_73k_path = meta_dir / "averaged" / "vdvae_latents_73k.npy"
    if not betas_path.exists():
        raise FileNotFoundError(betas_path)
    if not lat_73k_path.exists():
        raise FileNotFoundError(f"{lat_73k_path} — run vdvae_encoder_inference.py first")

    betas = np.load(betas_path).astype(np.float32)
    expdesign = load_expdesign(root)
    img_ids = trial_image_ids_73k(subj, len(betas), expdesign)
    latents_73k = np.load(lat_73k_path, mmap_mode="r")
    latents = np.array(latents_73k[img_ids], dtype=np.float32)
    return betas, latents, img_ids


def split_train_val_by_image(
    image_ids: np.ndarray,
    seed: int = 42,
    val_frac: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Hold out all trials of ~10% unique 73k images (no image in both train and val)."""
    image_ids = np.asarray(image_ids)
    unique = np.unique(image_ids)
    np.random.seed(seed)
    perm = np.random.permutation(len(unique))
    n_val = max(1, int(val_frac * len(unique)))
    val_images = unique[perm[:n_val]]
    val_mask = np.isin(image_ids, val_images)
    vl = np.flatnonzero(val_mask)
    tr = np.flatnonzero(~val_mask)
    assert not np.intersect1d(image_ids[tr], image_ids[vl]).size
    return tr, vl


def pick_val_trial_indices(
    n_trials: int,
    n_pick: int,
    *,
    seed: int = 42,
    pick_seed: int = 0,
    image_ids: np.ndarray | None = None,
) -> np.ndarray:
    """Val-split trial indices (image-level holdout when image_ids is given)."""
    if image_ids is not None:
        _, val_idx = split_train_val_by_image(image_ids, seed=seed)
    else:
        np.random.seed(seed)
        idx = np.random.permutation(n_trials)
        n_val = max(1, int(0.1 * n_trials))
        val_idx = idx[:n_val]
    rng = np.random.default_rng(pick_seed)
    pick = rng.choice(len(val_idx), size=min(n_pick, len(val_idx)), replace=False)
    return np.sort(val_idx[pick])


def layer_flat_dims_from_pack_or_ref(root: Path) -> np.ndarray | None:
    """Load per-layer flat dims; None if ref_stats.npz predates layer_flat_dims."""
    ref_npz = root / "vdvae" / "ref_stats.npz"
    if not ref_npz.exists():
        return None
    data = np.load(ref_npz, allow_pickle=True)
    if "layer_flat_dims" in data.files:
        dims = data["layer_flat_dims"].astype(np.int64)
        if dims.shape == (NUM_LATENT_LAYERS,):
            return dims
    return None
