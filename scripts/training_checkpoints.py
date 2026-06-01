"""Shared checkpoint helpers for variant training scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def retrieval_2way(
    pred: torch.Tensor,
    target: torch.Tensor,
    n_pairs: int = 10000,
    seed: int = 42,
) -> float:
    """
    2-way retrieval: fraction of random pairs (i, j) where
    cos(pred_i, target_i) > cos(pred_i, target_j). Chance = 0.5.

    pred/target are moved to CPU float32 (safe if passed from GPU).
    """
    pred = pred.detach().cpu().float()
    target = target.detach().cpu().float()
    pred = F.normalize(pred, dim=-1)
    target = F.normalize(target, dim=-1)
    n = pred.size(0)
    if n < 2:
        return 0.0

    rng = np.random.default_rng(seed)
    i = rng.integers(0, n, size=n_pairs)
    j = rng.integers(0, n, size=n_pairs)
    collide = i == j
    while collide.any():
        j[collide] = rng.integers(0, n, size=int(collide.sum()))
        collide = i == j

    sim_correct = (pred[i] * target[i]).sum(dim=-1)
    sim_wrong = (pred[i] * target[j]).sum(dim=-1)
    return float((sim_correct > sim_wrong).float().mean().item())


def compute_val_clip_cossim(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    predict_fn: Callable[[torch.nn.Module, torch.Tensor], torch.Tensor],
) -> float:
    """Mean cosine similarity between predicted and target CLIP embeddings on val set."""
    model.eval()
    sims: list[float] = []
    with torch.no_grad():
        for batch in val_loader:
            b, c = batch[0], batch[1]
            b = b.to(device, non_blocking=True)
            c = c.to(device, non_blocking=True)
            pc = predict_fn(model, b)
            sims.extend(
                F.cosine_similarity(
                    F.normalize(pc, dim=-1),
                    F.normalize(c, dim=-1),
                    dim=-1,
                ).cpu().tolist()
            )
    return float(np.mean(sims))


def _base_payload(
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    vm: np.ndarray,
    vs: np.ndarray,
    n_voxels: int,
    subj: str,
    variant: str,
    run_id: str,
    epochs: int,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "voxel_mean": vm,
        "voxel_std": vs,
        "n_voxels": n_voxels,
        "subj": subj,
        "variant": variant,
        "run_id": run_id,
        "epochs": epochs,
    }
    if extra:
        payload.update(extra)
    return payload


def maybe_save_best_clip(
    ckpt_path: Path,
    val_clip_cosim: float,
    best_clip_cosim: float,
    payload: dict[str, Any],
) -> float:
    """Save best_clip.pt if val_clip_cosim improved. Returns updated best_clip_cosim."""
    if val_clip_cosim > best_clip_cosim:
        best_clip_cosim = val_clip_cosim
        torch.save({**payload, "val_clip_cossim": val_clip_cosim}, ckpt_path / "best_clip.pt")
    return best_clip_cosim


def maybe_save_best_retrieval(
    ckpt_path: Path,
    val_retrieval_2way: float,
    best_retrieval_2way: float,
    payload: dict[str, Any],
) -> float:
    """Save best_retrieval.pt when val 2-way (projector) improves."""
    if val_retrieval_2way > best_retrieval_2way:
        best_retrieval_2way = val_retrieval_2way
        torch.save(
            {**payload, "val_retrieval_2way": val_retrieval_2way},
            ckpt_path / "best_retrieval.pt",
        )
    return best_retrieval_2way


def save_final_checkpoint(ckpt_path: Path, payload: dict[str, Any], epochs: int) -> Path:
    """Save weights after the full training schedule (epoch == epochs)."""
    out = ckpt_path / "final.pt"
    torch.save({**payload, "epoch": epochs}, out)
    return out
