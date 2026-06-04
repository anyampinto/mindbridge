# -*- coding: utf-8 -*-
"""Cache projector embeddings + load frozen dual contrastive for diffusion prior."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from paths import get_root, resolve_variant_checkpoint
from reconstruct_common import build_recon_model, normalize_variant
from vdvae_trial_data import load_expdesign, split_train_val_by_image, trial_image_ids_73k

DEFAULT_DUAL_RUN = "20260602_053625_train_dual_contrastive_subj01_150ep"
POOLED_PRIOR_SUBJ = "all_subjects"
PROJ_CACHE_NAME = "proj_embs_{subj}.npy"
CLIP_CACHE_NAME = "clip_embs_trials_{subj}.npy"
ALL_SUBJECTS = [f"subj{i:02d}" for i in range(1, 9)]


def dual_run_id_for_subj(run_id: str, subj: str) -> str:
    import re
    if re.search(r"subj\d+", run_id):
        return re.sub(r"subj\d+", subj, run_id, count=1)
    return run_id


def prior_ckpt_path(root: Path, subj: str, run_id: str | None = None) -> Path:
    if run_id:
        return root / "runs" / run_id / "checkpoints_prior" / subj / "best_prior.pt"
    if subj == POOLED_PRIOR_SUBJ:
        return root / "checkpoints_prior" / POOLED_PRIOR_SUBJ / "best_prior.pt"
    return root / "checkpoints_prior" / subj / "best_prior.pt"


@torch.no_grad()
def projector_embeddings(
    model: torch.nn.Module,
    betas_norm: torch.Tensor,
    *,
    batch_size: int = 256,
) -> np.ndarray:
    """Frozen dual model → L2-normalized 1024-d image projector vectors."""
    model.eval()
    out = []
    for i in range(0, len(betas_norm), batch_size):
        batch = betas_norm[i : i + batch_size]
        _, _, _, _, proj, _ = model(batch)
        out.append(F.normalize(proj.float(), dim=-1).cpu().numpy())
    return np.concatenate(out, axis=0).astype(np.float32)


def load_dual_for_prior(
    root: Path,
    subj: str,
    dual_run_id: str,
    device: torch.device,
    *,
    ckpt_prefer: str = "best_retrieval",
) -> tuple[torch.nn.Module, np.ndarray, np.ndarray]:
    ckpt_path = resolve_variant_checkpoint(
        root, "4H_CTR2", subj, dual_run_id, prefer=ckpt_prefer,
    )
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    vm = np.asarray(ckpt["voxel_mean"]).reshape(1, -1).astype(np.float32)
    vs = np.asarray(ckpt["voxel_std"]).reshape(1, -1).astype(np.float32)
    model = build_recon_model("4H_CTR2", int(ckpt["n_voxels"]), device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, vm, vs


def cache_perception_projector_pairs(
    subj: str,
    root: Path,
    *,
    dual_run_id: str = DEFAULT_DUAL_RUN,
    batch_size: int = 256,
    force: bool = False,
    ckpt_prefer: str = "best_retrieval",
) -> tuple[Path, Path]:
    """
    Cache (proj_emb, clip_emb) aligned to betas_flat rows.

    proj: nsd_meta/proj_embs_{subj}.npy  (N, 1024)
    clip: nsd_meta/clip_embs_trials_{subj}.npy  (N, 768) from targets_full
    """
    root = get_root(root)
    meta_dir = root / "nsd_meta"
    proj_path = meta_dir / PROJ_CACHE_NAME.format(subj=subj)
    clip_path = meta_dir / CLIP_CACHE_NAME.format(subj=subj)

    if proj_path.exists() and clip_path.exists() and not force:
        print(f"  Using cached {proj_path.name} and {clip_path.name}")
        return proj_path, clip_path

    betas = np.load(meta_dir / f"betas_flat_{subj}.npy").astype(np.float32)
    td = np.load(meta_dir / f"targets_full_{subj}.npz")
    clip = td["clip"].astype(np.float32)
    assert len(betas) == len(clip), f"betas {len(betas)} vs clip {len(clip)}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, vm, vs = load_dual_for_prior(
        root, subj, dual_run_id, device, ckpt_prefer=ckpt_prefer,
    )
    normed = (betas - vm) / vs
    betas_t = torch.tensor(normed, dtype=torch.float32, device=device)
    proj = projector_embeddings(model, betas_t, batch_size=batch_size)
    clip_n = clip / (np.linalg.norm(clip, axis=1, keepdims=True) + 1e-8)

    meta_dir.mkdir(parents=True, exist_ok=True)
    np.save(proj_path, proj)
    np.save(clip_path, clip_n.astype(np.float32))
    print(f"  Cached projector {proj.shape} → {proj_path}")
    print(f"  Cached CLIP targets {clip_n.shape} → {clip_path}")
    return proj_path, clip_path


def load_trial_splits(subj: str, root: Path, seed: int = 42, val_frac: float = 0.1):
    """Image-level train/val indices into betas_flat rows."""
    meta_dir = root / "nsd_meta"
    betas = np.load(meta_dir / f"betas_flat_{subj}.npy")
    exp = load_expdesign(root)
    img_ids = trial_image_ids_73k(subj, len(betas), exp)
    tr, vl = split_train_val_by_image(img_ids, seed=seed, val_frac=val_frac)
    return tr, vl, img_ids
