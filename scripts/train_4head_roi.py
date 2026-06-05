# -*- coding: utf-8 -*-
"""ROI voxel grouping for PCA/ridge ablations (early / mid / high visual field)."""

from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np

from paths import get_root


def _nsdgeneral_indices(root: Path, subj: str) -> np.ndarray:
    mask_path = (
        root / "nsddata" / "ppdata" / subj / "func1pt8mm" / "roi" / "nsdgeneral.nii.gz"
    )
    if not mask_path.exists():
        raise FileNotFoundError(f"Missing ROI mask: {mask_path}")
    mask = nib.load(str(mask_path)).get_fdata()
    return np.flatnonzero(mask > 0).astype(np.int64)


def _load_ecc_roi(root: Path, subj: str, labels: tuple[int, ...]) -> np.ndarray:
    """Voxel indices (within nsdgeneral layout) belonging to any of the given ROI labels."""
    roi_path = root / "nsddata" / "ppdata" / subj / "func1pt8mm" / "roi" / "prf-eccrois.nii.gz"
    if not roi_path.exists():
        raise FileNotFoundError(f"Missing prf-eccrois mask: {roi_path}")
    nsd_idx = _nsdgeneral_indices(root, subj)
    roi_vol = nib.load(str(roi_path)).get_fdata()
    flat = roi_vol.ravel()[nsd_idx]
    keep = np.zeros(flat.shape[0], dtype=bool)
    for lab in labels:
        keep |= flat == lab
    return np.flatnonzero(keep).astype(np.int64)


def load_roi_indices(
    subj: str,
    root: Path | str,
    *,
    n_voxels: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (early, mid, high) voxel index arrays within the nsdgeneral beta vector.

    Groups follow prf-eccrois (Brain-Diffuser convention):
      early — ecc05 + ecc10 (labels 1, 2)
      mid   — ecc20 (label 3)
      high  — ecc40 + ecc40+ (labels 4, 5)

    Falls back to three equal contiguous thirds of the nsdgeneral vector when
    prf-eccrois is unavailable (sufficient for shuffle ablation at test time).
    """
    root = get_root(root)
    cache = root / "nsd_meta" / f"roi_voxel_groups_{subj}.json"
    if cache.exists():
        data = json.loads(cache.read_text())
        return (
            np.asarray(data["early_idx"], dtype=np.int64),
            np.asarray(data["mid_idx"], dtype=np.int64),
            np.asarray(data["high_idx"], dtype=np.int64),
        )

    roi_path = root / "nsddata" / "ppdata" / subj / "func1pt8mm" / "roi" / "prf-eccrois.nii.gz"
    if roi_path.exists():
        early = _load_ecc_roi(root, subj, (1, 2))
        mid = _load_ecc_roi(root, subj, (3,))
        high = _load_ecc_roi(root, subj, (4, 5))
        source = "prf-eccrois"
    else:
        try:
            n = len(_nsdgeneral_indices(root, subj))
        except FileNotFoundError:
            if n_voxels is None:
                raise
            n = n_voxels
            source = "contiguous_thirds_from_betas"
            print(f"  Warning: nsdgeneral mask missing — ROI shuffle uses {n} voxels from betas")
        else:
            source = "contiguous_thirds_fallback"
            print(f"  Warning: {roi_path.name} missing — ROI shuffle uses {source}")
        t = max(n // 3, 1)
        early = np.arange(0, t, dtype=np.int64)
        mid = np.arange(t, 2 * t, dtype=np.int64)
        high = np.arange(2 * t, n, dtype=np.int64)

    payload = {
        "subj": subj,
        "source": source,
        "early_idx": early.tolist(),
        "mid_idx": mid.tolist(),
        "high_idx": high.tolist(),
        "n_early": int(early.size),
        "n_mid": int(mid.size),
        "n_high": int(high.size),
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(payload, indent=2))
    print(f"  Cached ROI groups → {cache}")
    return early, mid, high
