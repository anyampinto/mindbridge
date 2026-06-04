# -*- coding: utf-8 -*-
"""Ingest NSD-Imagery betas with trial metadata and per-stimulus averaging."""

import json
import os
import re
import sys

sys.path = [p for p in sys.path if "share/software" not in p and "jupyterlab" not in p]

import h5py
import nibabel as nib
import numpy as np
from pathlib import Path
from scipy.io import loadmat

from download_nsd import (
    download_imagery_betas,
    download_imagery_metadata,
    download_roi_masks,
    imagery_beta_path,
    require_aws_cli,
)
from paths import ensure_runtime_dirs, get_root

# Chronological GLMsingle run order (beta counts: 48,96,48,48, 96,48,48,96, 48,48,48,48).
RUN_SPECS = [
    ("vis", "A", "visA_dm.mat"),
    ("att", "A", "attA_dm.mat"),
    ("img", "A", "imgA_1_dm.mat"),
    ("img", "A", "imgA_2_dm.mat"),
    ("att", "B", "attB_dm.mat"),
    ("vis", "B", "visB_dm.mat"),
    ("img", "B", "imgB_1_dm.mat"),
    ("att", "C", "attC_dm.mat"),
    ("img", "B", "imgB_2_dm.mat"),
    ("vis", "C", "visC_dm.mat"),
    ("img", "C", "imgC_1_dm.mat"),
    ("img", "C", "imgC_2_dm.mat"),
]


def _mat_scalar(x) -> str:
    return str(np.asarray(x).item())


def _load_pair_lists(meta_dir: Path) -> dict[str, dict[str, dict]]:
    """Map set letter -> cue letter -> {stimulus_id, label}."""
    out: dict[str, dict[str, dict]] = {}
    for tag in ("A", "B", "C"):
        pair_list = loadmat(str(meta_dir / f"{tag}_pair_list.mat"))["pair_list"]
        out[tag] = {}
        for row in pair_list:
            sid = int(np.asarray(row[0]).item())
            label = _mat_scalar(row[1])
            cue = _mat_scalar(row[2])
            nsd_id = None
            m = re.search(r"nsd(\d+)", label)
            if m:
                nsd_id = int(m.group(1))
            out[tag][cue] = {
                "stimulus_id": sid,
                "label": label,
                "nsd_id": nsd_id,
            }
    return out


def _condit_entries(condit_list: np.ndarray) -> list[list[str]]:
    entries = []
    for row in condit_list:
        entries.append([_mat_scalar(x) for x in row.ravel()])
    return entries


def _cue_from_condit(entries: list[list[str]], col: int) -> str:
    if col < 0 or col >= len(entries):
        return ""
    return entries[col][0]


def _build_trial_table(meta_dir: Path) -> list[dict]:
    """Chronological trial metadata for all 720 GLMsingle betas."""
    dm_all = loadmat(str(meta_dir / "designmatrixGLMsingle.mat"))["stimulus"][0]
    pair_lists = _load_pair_lists(meta_dir)
    trials: list[dict] = []
    beta_idx = 0

    for run_idx, (task, stim_set, dm_file) in enumerate(RUN_SPECS):
        run_dm = np.asarray(dm_all[run_idx])
        local = loadmat(str(meta_dir / dm_file))
        condit_entries = _condit_entries(local["condit_list"])

        active_rows = np.where(run_dm.sum(axis=1) > 0)[0]
        used_cols = sorted({int(run_dm[row].argmax()) for row in active_rows})
        col_to_local = {c: i for i, c in enumerate(used_cols)}
        print(f"  run {run_idx:02d} {dm_file}: {len(active_rows)} betas, dm_cols={used_cols}")

        for row in active_rows:
            col = int(run_dm[row].argmax())
            local_col = col_to_local[col]
            cue = _cue_from_condit(condit_entries, local_col)
            stim_info = pair_lists.get(stim_set, {}).get(cue, {})
            trials.append(
                {
                    "beta_idx": beta_idx,
                    "run_idx": run_idx,
                    "run_name": dm_file.replace("_dm.mat", ""),
                    "task": task,
                    "set": stim_set,
                    "cond_col": col,
                    "cue": cue,
                    "stimulus_id": stim_info.get("stimulus_id"),
                    "label": stim_info.get("label"),
                    "nsd_id": stim_info.get("nsd_id"),
                }
            )
            beta_idx += 1

    if beta_idx != 720:
        raise ValueError(f"Expected 720 trials total, got {beta_idx}")
    return trials


def _load_mask(subj: str, root: Path) -> np.ndarray:
    mask_path = (
        root / "nsd_data" / "nsddata" / "ppdata" / subj
        / "func1pt8mm" / "roi" / "nsdgeneral.nii.gz"
    )
    if not mask_path.exists():
        raise FileNotFoundError(f"ROI mask not found: {mask_path}")
    mask_img = nib.load(str(mask_path))
    mask = np.transpose(mask_img.get_fdata(), (2, 1, 0)) == 1
    print(f"  n_voxels in ROI: {mask.sum()}")
    return mask.reshape(-1)


def _ensure_imagery_via_aws(subj: str, root: Path, *, force: bool = False) -> None:
    """Sync imagery metadata, ROI mask, and betas from S3 via AWS CLI."""
    if os.environ.get("NSD_SKIP_DOWNLOAD", "0") == "1":
        print("NSD_SKIP_DOWNLOAD=1 — skipping AWS downloads.")
        return

    require_aws_cli()
    print("Syncing NSD-Imagery metadata from S3...")
    download_imagery_metadata(root, force=force)
    print(f"Syncing ROI mask for {subj}...")
    download_roi_masks(subj, root, force=force)
    print(f"Syncing imagery betas for {subj}...")
    download_imagery_betas(subj, root, force=force)
    print("AWS sync complete.")


def _ensure_imagery_metadata(root: Path, *, force: bool = False) -> None:
    """Sync NSD-Imagery experiment metadata only (design matrices, pair lists, PNGs)."""
    if os.environ.get("NSD_SKIP_DOWNLOAD", "0") == "1":
        print("NSD_SKIP_DOWNLOAD=1 — skipping AWS downloads.")
        return
    require_aws_cli()
    print("Syncing NSD-Imagery metadata from S3...")
    download_imagery_metadata(root, force=force)


def _merge_averaged(
    root: Path,
    subj: str,
    sets_filter: frozenset[str],
    new_meta: list[dict],
    new_betas: np.ndarray,
) -> tuple[list[dict], np.ndarray]:
    """Replace rows for `sets_filter` and keep all other existing averaged stimuli."""
    meta_path = root / "nsd_meta" / f"imagery_trial_meta_{subj}.json"
    avg_path = root / "nsd_meta" / f"betas_avg_imagery_{subj}.npy"
    kept_meta: list[dict] = []
    kept_betas: list[np.ndarray] = []
    if meta_path.exists() and avg_path.exists():
        with open(meta_path) as f:
            old_payload = json.load(f)
        old_avg = np.load(avg_path)
        for i, row in enumerate(old_payload["averaged"]):
            if row["set"] in sets_filter:
                continue
            kept_meta.append(row)
            kept_betas.append(old_avg[i])
    merged_meta = kept_meta + new_meta
    merged_betas = np.vstack(kept_betas + [new_betas]) if kept_betas else new_betas
    order = sorted(
        range(len(merged_meta)),
        key=lambda i: (merged_meta[i]["set"], merged_meta[i]["stimulus_id"]),
    )
    return [merged_meta[i] for i in order], merged_betas[order]


def run_ingestion(
    subj: str = "subj01",
    root: Path | None = None,
    *,
    force: bool = False,
    sets: str | None = None,
) -> dict:
    root = get_root(root)
    ensure_runtime_dirs(root, subj)
    meta_dir = root / "nsd_meta" / "nsdimagery"
    out_dir = root / "nsd_meta"

    print(f"Imagery ingestion for {subj}")
    print(f"Root: {root}")

    sets_filter: frozenset[str] | None = None
    if sets:
        sets_filter = frozenset(s.strip().upper() for s in sets.split(",") if s.strip())
        print(f"  Partial ingest for sets: {sorted(sets_filter)}")
        _ensure_imagery_metadata(root, force=force)
    else:
        _ensure_imagery_via_aws(subj, root, force=force)

    meta_dir = root / "nsd_meta" / "nsdimagery"
    beta_path = imagery_beta_path(root, subj)
    assert meta_dir.exists(), f"Missing imagery metadata: {meta_dir}"
    if not beta_path.exists():
        if sets_filter:
            print("  Imagery betas missing — downloading HDF5...")
            require_aws_cli()
            download_imagery_betas(subj, root, force=force)
        else:
            raise FileNotFoundError(f"Missing imagery betas: {beta_path}")
    assert beta_path.exists(), f"Missing imagery betas: {beta_path}"

    mask_flat = _load_mask(subj, root)
    trials = _build_trial_table(meta_dir)

    print("Loading imagery betas HDF5...")
    with h5py.File(beta_path, "r") as f:
        raw = f["betas"][:]
    if raw.shape[0] != 720:
        raise ValueError(f"Expected 720 trials in HDF5, got {raw.shape[0]}")

    betas_2d = (raw.astype(np.float32) / 300.0).reshape(720, -1)[:, mask_flat]
    print(f"  Flat betas: {betas_2d.shape}")

    # Full chronological betas (skip on partial set ingest — flat files already exist)
    if not sets_filter:
        full_file = out_dir / f"betas_flat_nsdimagery_{subj}.npy"
        np.save(full_file, betas_2d)

        # Imagery-task trials only
        img_mask = np.array([t["task"] == "img" for t in trials])
        img_trials = [t for t in trials if t["task"] == "img"]
        img_betas = betas_2d[img_mask]
        img_file = out_dir / f"betas_flat_imagery_{subj}.npy"
        np.save(img_file, img_betas)
        print(f"  Imagery trials: {img_betas.shape}")
    else:
        img_trials = [t for t in trials if t["task"] == "img"]
        img_betas = betas_2d[[t["beta_idx"] for t in img_trials]]
        print(f"  Imagery trials (partial): {img_betas.shape}")

    # Per-stimulus average within imagery runs (18 conditions: 3 sets × 6 stimuli)
    groups: dict[tuple[str, int], list[int]] = {}
    for i, t in enumerate(img_trials):
        if sets_filter and t["set"] not in sets_filter:
            continue
        key = (t["set"], t["stimulus_id"])
        if key[1] is None:
            continue
        groups.setdefault(key, []).append(i)

    avg_rows = []
    avg_meta = []
    for (stim_set, sid) in sorted(groups.keys(), key=lambda x: (x[0], x[1])):
        idxs = groups[(stim_set, sid)]
        avg_rows.append(img_betas[idxs].mean(axis=0))
        sample = img_trials[idxs[0]]
        avg_meta.append(
            {
                "set": stim_set,
                "stimulus_id": sid,
                "cue": sample["cue"],
                "label": sample["label"],
                "nsd_id": sample["nsd_id"],
                "n_reps": len(idxs),
            }
        )

    if not avg_rows:
        raise ValueError(
            f"No averaged stimuli for sets={sorted(sets_filter) if sets_filter else 'all'}; "
            "check nsdimagery metadata (pair lists / design matrices)."
        )

    avg_betas = np.stack(avg_rows, axis=0)
    if sets_filter:
        avg_meta, avg_betas = _merge_averaged(root, subj, sets_filter, avg_meta, avg_betas)
        print(f"  Merged stimulus-averaged imagery: {avg_betas.shape}")
    else:
        print(f"  Stimulus-averaged imagery: {avg_betas.shape}")
    for m in avg_meta:
        print(f"    set={m['set']} id={m['stimulus_id']} cue={m['cue']} reps={m['n_reps']}")

    avg_file = out_dir / f"betas_avg_imagery_{subj}.npy"
    np.save(avg_file, avg_betas)

    meta_file = out_dir / f"imagery_trial_meta_{subj}.json"
    img_mask = np.array([t["task"] == "img" for t in trials])
    payload = {
        "subj": subj,
        "n_total": 720,
        "n_imagery": int(img_mask.sum()),
        "n_averaged": len(avg_meta),
        "trials": trials,
        "averaged": avg_meta,
    }
    with open(meta_file, "w") as f:
        json.dump(payload, f, indent=2)

    summary = {
        "subj": subj,
        "sets": sorted(sets_filter) if sets_filter else "all",
        "averaged": str(avg_file),
        "meta": str(meta_file),
        "shapes": {
            "full": list(betas_2d.shape),
            "averaged": list(avg_betas.shape),
        },
    }
    if not sets_filter:
        summary["full"] = str(out_dir / f"betas_flat_nsdimagery_{subj}.npy")
        summary["imagery"] = str(out_dir / f"betas_flat_imagery_{subj}.npy")
        summary["shapes"]["imagery"] = list(img_betas.shape)
    print("Imagery ingestion complete.")
    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--subj", default=os.environ.get("NSD_SUBJ", "subj01"))
    parser.add_argument("--root", default=os.environ.get("MINDBRIDGE_ROOT", "/mnt/mindbridge"))
    parser.add_argument("--force", action="store_true", help="Re-download from S3")
    parser.add_argument("--sets", default=None,
                        help="Comma-separated stimulus sets to ingest (e.g. C). Merges into existing averages.")
    args = parser.parse_args()
    run_ingestion(subj=args.subj, root=get_root(args.root), force=args.force, sets=args.sets)
