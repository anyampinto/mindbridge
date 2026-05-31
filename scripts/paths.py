"""Shared path helpers for MindBridge on Modal, Sherlock, or local."""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_ROOT = "/mnt/mindbridge"
FALLBACK_ROOT = "/scratch/users/ampinto/mindbridge"
CHECKPOINT_NAME = "best_stage1.pt"


def get_root(root: Path | str | None = None) -> Path:
    if root is not None:
        return Path(root)
    env = os.environ.get("MINDBRIDGE_ROOT")
    if env:
        return Path(env)
    if Path(DEFAULT_ROOT).exists():
        return Path(DEFAULT_ROOT)
    return Path(FALLBACK_ROOT)


def checkpoint_dir(root: Path | str, subj: str) -> Path:
    return get_root(root) / "checkpoints" / subj


def checkpoint_file(root: Path | str, subj: str) -> Path:
    return checkpoint_dir(root, subj) / CHECKPOINT_NAME


def resolve_checkpoint(root: Path | str, subj: str) -> Path:
    """Return an existing checkpoint path (canonical or legacy flat upload)."""
    root = get_root(root)
    canonical = checkpoint_file(root, subj)
    if canonical.exists():
        return canonical

    legacy = root / "checkpoints" / subj
    if legacy.is_file():
        return legacy

    raise FileNotFoundError(
        f"No checkpoint for {subj}. Tried:\n"
        f"  - {canonical}\n"
        f"  - {legacy}"
    )


def ensure_runtime_dirs(root: Path | str, subj: str | None = None) -> None:
    root = get_root(root)
    for path in [
        root / "nsd_data",
        root / "nsd_meta",
        root / "checkpoints",
        root / "logs",
        root / "reconstructions",
        root / "hf_cache",
        root / "torch_cache",
    ]:
        path.mkdir(parents=True, exist_ok=True)
    if subj:
        checkpoint_dir(root, subj).mkdir(parents=True, exist_ok=True)
        (root / "logs" / subj).mkdir(parents=True, exist_ok=True)
        (root / "reconstructions" / subj).mkdir(parents=True, exist_ok=True)
