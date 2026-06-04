# -*- coding: utf-8 -*-
"""Paired (imagery, perception) betas for the same NSD-Imagery stimuli."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from paths import averaged_meta_dir, get_root


def _average_vis_imagery_betas(
    trials: list[dict],
    betas_720: np.ndarray,
) -> dict[tuple[str, int], np.ndarray]:
    """Mean vision-task betas during NSD-Imagery session, keyed by (set, stimulus_id)."""
    groups: dict[tuple[str, int], list[int]] = {}
    for t in trials:
        if t.get("task") != "vis":
            continue
        sid = t.get("stimulus_id")
        if sid is None:
            continue
        groups.setdefault((t["set"], int(sid)), []).append(int(t["beta_idx"]))
    return {k: betas_720[idxs].mean(0).astype(np.float32) for k, idxs in groups.items()}


def perception_beta_for_row(
    row: dict,
    *,
    perc_avg: np.ndarray,
    image_id_73k: np.ndarray,
    slot_10k: np.ndarray,
    vis_avg: dict[tuple[str, int], np.ndarray],
) -> np.ndarray | None:
    """NSD training perception average for one imagery-averaged row."""
    nid = row.get("nsd_id")
    if nid is not None and int(nid) > 0:
        hits = np.where(image_id_73k == int(nid))[0]
        if len(hits):
            return perc_avg[hits[0]]

    label = row.get("label") or ""
    m = re.search(r"shared(\d+)", label, re.I)
    if m:
        slot = int(m.group(1)) - 1
        hits = np.where(slot_10k == slot)[0]
        if len(hits):
            return perc_avg[hits[0]]

    key = (row.get("set"), row.get("stimulus_id"))
    if key[0] is not None and key[1] is not None:
        return vis_avg.get((str(key[0]), int(key[1])))
    return None


def load_paired_betas(
    subj: str,
    root: Path | str,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """
    Aligned pairs: imagery-averaged img-task betas ↔ perception betas (same content).

    Perception source priority:
      1. NSD training image average (betas_avg_perception) via nsd_id or shared10k slot
      2. Vision-task average from NSD-Imagery session (Set A / unmatched rows)
    """
    root = get_root(root)
    meta_path = root / "nsd_meta" / f"imagery_trial_meta_{subj}.json"
    img_path = root / "nsd_meta" / f"betas_avg_imagery_{subj}.npy"
    if not meta_path.exists() or not img_path.exists():
        raise FileNotFoundError(f"Run ingest_imagery for {subj} first")

    with open(meta_path) as f:
        meta = json.load(f)
    img_avg = np.load(img_path).astype(np.float32)
    rows = meta["averaged"]

    avg_dir = averaged_meta_dir(root)
    perc_avg = np.load(avg_dir / f"betas_avg_perception_{subj}.npy").astype(np.float32)
    td = np.load(avg_dir / f"targets_avg_{subj}.npz")
    i73 = td["image_id_73k"].astype(np.int64)
    slot_10k = td["slot_10k"].astype(np.int64) if "slot_10k" in td.files else None
    if slot_10k is None:
        slot_10k = np.array([], dtype=np.int64)

    full_path = root / "nsd_meta" / f"betas_flat_nsdimagery_{subj}.npy"
    if full_path.exists():
        betas_720 = np.load(full_path)
    else:
        raise FileNotFoundError(f"Missing {full_path} — run full imagery ingest")

    vis_avg = _average_vis_imagery_betas(meta["trials"], betas_720)

    img_out: list[np.ndarray] = []
    perc_out: list[np.ndarray] = []
    paired_rows: list[dict] = []

    for i, row in enumerate(rows):
        perc_b = perception_beta_for_row(
            row,
            perc_avg=perc_avg,
            image_id_73k=i73,
            slot_10k=slot_10k,
            vis_avg=vis_avg,
        )
        if perc_b is None:
            continue
        img_out.append(img_avg[i])
        perc_out.append(perc_b)
        paired_rows.append({**row, "pair_idx": len(paired_rows), "imagery_row": i})

    if not img_out:
        raise ValueError(f"No imagery↔perception pairs for {subj}")

    return np.stack(img_out), np.stack(perc_out), paired_rows


def perception_voxel_stats(subj: str, root: Path | str) -> tuple[np.ndarray, np.ndarray]:
    """Per-voxel mean/std from NSD perception train trials (matches MLP checkpoints)."""
    root = get_root(root)
    betas = np.load(root / "nsd_meta" / f"betas_flat_{subj}.npy").astype(np.float32)
    np.random.seed(42)
    idx = np.random.permutation(len(betas))
    n_train = int(0.9 * len(betas))
    tr = betas[idx[:n_train]]
    vm = tr.mean(0, keepdims=True)
    vs = tr.std(0, ddof=1, keepdims=True) + 1e-6
    return vm.astype(np.float32), vs.astype(np.float32)
