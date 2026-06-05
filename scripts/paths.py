"""Shared path helpers for MindBridge on Modal, Sherlock, or local."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_ROOT = "/mnt/mindbridge"
FALLBACK_ROOT = "/scratch/users/ampinto/mindbridge"
CHECKPOINT_NAME = "best_stage1.pt"
RUNS_DIR = "runs"
# Trial-level ingestion (betas_flat_*, targets_full_*) — do not write averaged artifacts here.
AVERAGED_SUBDIR = "averaged"


def get_root(root: Path | str | None = None) -> Path:
    if root is not None:
        return Path(root)
    env = os.environ.get("MINDBRIDGE_ROOT")
    if env:
        return Path(env)
    if Path(DEFAULT_ROOT).exists():
        return Path(DEFAULT_ROOT)
    return Path(FALLBACK_ROOT)


def make_run_id(label: str = "") -> str:
    """Unique run id: YYYYMMDD_HHMMSS_label (UTC). Never collides with prior runs."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    clean = re.sub(r"[^a-zA-Z0-9_-]+", "_", label).strip("_")
    return f"{ts}_{clean}" if clean else ts


def get_run_id(required: bool = False) -> str | None:
    return os.environ.get("MINDBRIDGE_RUN_ID") or (None if not required else _require_run_id())


def resolve_run_id(label: str = "") -> str:
    """Return existing MINDBRIDGE_RUN_ID or create and export a new one."""
    rid = os.environ.get("MINDBRIDGE_RUN_ID")
    if rid:
        return rid
    rid = make_run_id(label)
    os.environ["MINDBRIDGE_RUN_ID"] = rid
    return rid


def _require_run_id() -> str:
    rid = os.environ.get("MINDBRIDGE_RUN_ID")
    if not rid:
        raise ValueError(
            "MINDBRIDGE_RUN_ID is not set. All training/reconstruction outputs must "
            "live under runs/{run_id}/ to avoid overwriting prior work."
        )
    return rid


def run_root(root: Path | str | None = None, run_id: str | None = None) -> Path:
    root = get_root(root)
    rid = run_id or _require_run_id()
    path = root / RUNS_DIR / rid
    path.mkdir(parents=True, exist_ok=True)
    return path


def joint_ridge_ckpt_dir(root: Path | str, subj: str, run_id: str | None = None) -> Path:
    """Per-subject joint ridge weights (W_joint.npy, voxel stats, …)."""
    root = get_root(root)
    rid = run_id or os.environ.get("MINDBRIDGE_RUN_ID") or "20260601_053733_joint_ridge_all8"
    return run_root(root, rid) / "checkpoints_vJointRidge" / subj


def resolve_joint_ridge_run_id(run_id: str | None = None) -> str:
    if run_id:
        return run_id
    env = os.environ.get("MINDBRIDGE_RUN_ID")
    if env:
        return env
    return "20260601_053733_joint_ridge_all8"


def variant_ckpt_subdir(variant: str) -> str:
    """Filesystem folder under runs/{run_id}/ for a variant (may differ from variant.upper())."""
    overrides = {
        "4head_roi": "checkpoints_v4head_roi",
        "4H_roi": "checkpoints_v4head_roi",
        "4head_contrastive": "checkpoints_v4head_contrastive",
        "4H_ctr": "checkpoints_v4head_contrastive",
        "4H_CTR": "checkpoints_v4H_CTR",
        "4H_CTR2": "checkpoints_v4H_CTR2",
        "4H_CTR2_1H": "checkpoints_v4H_CTR2_1H",
        "dual_ctr": "checkpoints_v4H_CTR2",
        "4head_retrieval": "checkpoints_v4head_retrieval",
    }
    key = variant.strip()
    return overrides.get(key, f"checkpoints_v{key.upper()}")


def variant_ckpt_dir(
    root: Path | str,
    variant: str,
    subj: str,
    run_id: str | None = None,
) -> Path:
    path = run_root(root, run_id) / variant_ckpt_subdir(variant) / subj
    path.mkdir(parents=True, exist_ok=True)
    return path


def variant_log_dir(
    root: Path | str,
    variant: str,
    subj: str,
    run_id: str | None = None,
) -> Path:
    path = run_root(root, run_id) / f"logs_v{variant.upper()}" / subj
    path.mkdir(parents=True, exist_ok=True)
    return path


def reconstruction_run_dir(
    root: Path | str,
    subj: str,
    run_id: str | None = None,
) -> Path:
    path = run_root(root, run_id) / "reconstructions" / subj
    path.mkdir(parents=True, exist_ok=True)
    return path


def variant_checkpoint_file(
    root: Path | str,
    variant: str,
    subj: str,
    run_id: str | None = None,
    name: str = "final.pt",
) -> Path:
    return variant_ckpt_dir(root, variant, subj, run_id) / name


def resolve_variant_checkpoint(
    root: Path | str,
    variant: str,
    subj: str,
    run_id: str | None = None,
    prefer: str = "final",
) -> Path:
    """Resolve checkpoint: final.pt > best_clip.pt > legacy best.pt."""
    root = get_root(root)
    order = {
        "final": ("final.pt", "best_clip.pt", "best_retrieval.pt", "best_gen_proj.pt", "best.pt"),
        "best_clip": ("best_clip.pt", "final.pt", "best_retrieval.pt", "best_gen_proj.pt", "best.pt"),
        "best_retrieval": ("best_retrieval.pt", "best_gen_proj.pt", "best_clip.pt", "final.pt", "best.pt"),
        "best_gen_proj": ("best_gen_proj.pt", "best_retrieval.pt", "best_clip.pt", "final.pt", "best.pt"),
    }.get(prefer, (prefer, "final.pt", "best_clip.pt", "best_retrieval.pt", "best_gen_proj.pt", "best.pt"))

    if run_id or os.environ.get("MINDBRIDGE_RUN_ID"):
        for fname in order:
            path = variant_checkpoint_file(root, variant, subj, run_id, fname)
            if path.exists():
                return path

    legacy = root / f"checkpoints_v{variant.upper()}" / subj / "best.pt"
    if legacy.exists():
        print(f"  WARNING: using legacy checkpoint: {legacy}")
        return legacy

    rid = run_id or os.environ.get("MINDBRIDGE_RUN_ID", "?")
    tried = [str(root / RUNS_DIR / rid / variant_ckpt_subdir(variant) / subj / f) for f in order]
    raise FileNotFoundError(
        f"No checkpoint for {subj} variant {variant}. Tried:\n  " + "\n  ".join(tried)
    )


def resolve_variant_checkpoint_discover(
    root: Path | str,
    variant: str,
    subj: str,
    run_id: str | None = None,
    prefer: str = "best_retrieval",
) -> tuple[Path, str]:
    """
    Like resolve_variant_checkpoint, but if run_id misses, scan runs/*/checkpoints_* /subj/.
    Returns (path, resolved_run_id).
    """
    import re

    root = get_root(root)
    candidates: list[str] = []
    if run_id:
        candidates.append(run_id)
        if re.search(r"subj\d+", run_id):
            rewritten = re.sub(r"subj\d+", subj, run_id, count=1)
            if rewritten not in candidates:
                candidates.append(rewritten)

    for rid in candidates:
        try:
            return resolve_variant_checkpoint(root, variant, subj, rid, prefer), rid
        except FileNotFoundError:
            continue

    order = {
        "final": ("final.pt", "best_clip.pt", "best_retrieval.pt", "best_gen_proj.pt", "best.pt"),
        "best_clip": ("best_clip.pt", "final.pt", "best_retrieval.pt", "best_gen_proj.pt", "best.pt"),
        "best_retrieval": ("best_retrieval.pt", "best_gen_proj.pt", "best_clip.pt", "final.pt", "best.pt"),
        "best_gen_proj": ("best_gen_proj.pt", "best_retrieval.pt", "best_clip.pt", "final.pt", "best.pt"),
    }.get(prefer, (prefer, "final.pt", "best_clip.pt", "best_retrieval.pt", "best_gen_proj.pt", "best.pt"))
    subdir = variant_ckpt_subdir(variant)
    for run_dir in list_runs(root, variant=None):
        for fname in order:
            path = run_dir / subdir / subj / fname
            if path.exists():
                print(f"  Discovered {variant} ckpt for {subj}: {path} (run={run_dir.name})")
                return path, run_dir.name

    raise FileNotFoundError(
        f"No checkpoint for {subj} variant {variant} (scanned runs/*/{subdir}/)."
    )


def update_run_manifest(run_base: Path, **fields) -> None:
    manifest_path = run_base / "manifest.json"
    manifest: dict = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    for key, val in fields.items():
        if key in manifest and isinstance(manifest[key], dict) and isinstance(val, dict):
            manifest[key].update(val)
        else:
            manifest[key] = val
    manifest.setdefault("created_at", datetime.now(timezone.utc).isoformat())
    manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"  manifest: {manifest_path}")


def list_runs(root: Path | str | None = None, variant: str | None = None) -> list[Path]:
    root = get_root(root)
    runs_base = root / RUNS_DIR
    if not runs_base.exists():
        return []
    runs = sorted(runs_base.iterdir(), key=lambda p: p.name, reverse=True)
    if not variant:
        return [p for p in runs if p.is_dir()]
    suffix = f"checkpoints_v{variant.upper()}"
    return [p for p in runs if p.is_dir() and (p / suffix).exists()]


# --- legacy stage-1 paths (unchanged; do not write here from new runs) ---

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


def averaged_meta_dir(root: Path | str | None = None) -> Path:
    """Per-image averaged betas/targets (separate from trial-level nsd_meta files)."""
    path = get_root(root) / "nsd_meta" / AVERAGED_SUBDIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_runtime_dirs(root: Path | str, subj: str | None = None) -> None:
    root = get_root(root)
    for path in [
        root / "nsd_data",
        root / "nsd_meta",
        root / "nsd_meta" / AVERAGED_SUBDIR,
        root / "checkpoints",
        root / "logs",
        root / "reconstructions",
        root / RUNS_DIR,
        root / "hf_cache",
        root / "torch_cache",
    ]:
        path.mkdir(parents=True, exist_ok=True)
    if subj:
        checkpoint_dir(root, subj).mkdir(parents=True, exist_ok=True)
        (root / "logs" / subj).mkdir(parents=True, exist_ok=True)
        (root / "reconstructions" / subj).mkdir(parents=True, exist_ok=True)
