# -*- coding: utf-8 -*-
"""
Test-time brain-input ablations for MindBridge reconstruction / retrieval.

Modes:
  zeros       — all-zero beta vector (prior-only gatekeeper)
  mean        — training-set mean beta (constant input)
  random      — i.i.d. Gaussian matching train mean/std per voxel
  roi_shuffle — permute voxel order independently within each ROI group
"""

from __future__ import annotations

import numpy as np

from train_4head_roi import load_roi_indices

BRAIN_ABLATION_CHOICES = ("zeros", "mean", "random", "roi_shuffle")


def apply_brain_ablation(
    betas: np.ndarray,
    mode: str,
    *,
    root,
    subj: str,
    seed: int = 42,
    train_mean: np.ndarray | None = None,
    train_std: np.ndarray | None = None,
) -> np.ndarray:
    """
    Return corrupted betas for ablation inference. `betas` shape (B, n_voxels) or (n_voxels,).
    """
    if mode in ("", "none", None):
        return betas

    mode = mode.lower()
    if mode not in BRAIN_ABLATION_CHOICES:
        raise ValueError(f"brain_ablation must be one of {BRAIN_ABLATION_CHOICES}, got {mode!r}")

    single = betas.ndim == 1
    x = betas.astype(np.float32, copy=True)
    if single:
        x = x.reshape(1, -1)

    if mode == "zeros":
        out = np.zeros_like(x)
    elif mode == "mean":
        mu = _resolve_mean(x, train_mean)
        out = np.broadcast_to(mu, x.shape).copy()
    elif mode == "random":
        mu = _resolve_mean(x, train_mean)
        sig = _resolve_std(x, train_std)
        rng = np.random.default_rng(seed)
        out = rng.normal(loc=mu, scale=sig, size=x.shape).astype(np.float32)
    elif mode == "roi_shuffle":
        out = _roi_shuffle_batch(x, subj, root, seed=seed)
    else:
        raise ValueError(mode)

    return out[0] if single else out


def _resolve_mean(x: np.ndarray, train_mean: np.ndarray | None) -> np.ndarray:
    if train_mean is not None:
        mu = np.asarray(train_mean, dtype=np.float32).reshape(1, -1)
        if mu.shape[1] != x.shape[1]:
            raise ValueError(f"train_mean n_voxels={mu.shape[1]} != betas {x.shape[1]}")
        return mu
    return x.mean(axis=0, keepdims=True)


def _resolve_std(x: np.ndarray, train_std: np.ndarray | None) -> np.ndarray:
    if train_std is not None:
        sig = np.asarray(train_std, dtype=np.float32).reshape(1, -1)
        if sig.shape[1] != x.shape[1]:
            raise ValueError(f"train_std n_voxels={sig.shape[1]} != betas {x.shape[1]}")
        return np.maximum(sig, 1e-6)
    return x.std(axis=0, keepdims=True) + 1e-6


def _roi_shuffle_batch(
    x: np.ndarray,
    subj: str,
    root,
    *,
    seed: int,
) -> np.ndarray:
    early, mid, high = load_roi_indices(subj, root, n_voxels=x.shape[-1])
    groups = [early, mid, high]
    out = x.copy()
    rng = np.random.default_rng(seed)
    for b in range(out.shape[0]):
        for idx in groups:
            if idx.size < 2:
                continue
            perm = rng.permutation(idx.size)
            out[b, idx] = out[b, idx[perm]]
    return out


def ablation_out_subdir(base: str, ablation: str) -> str:
    tag = ablation.replace("_", "-")
    return f"{base}_abl_{tag}"
