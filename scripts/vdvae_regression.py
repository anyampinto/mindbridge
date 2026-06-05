# -*- coding: utf-8 -*-
"""
Ridge regression: fMRI betas → VDVAE latents (all 31 layers, flat 91k-d).

Default: **unaveraged perception trials** — betas_flat_{subj}.npy (one row per trial),
VDVAE targets from vdvae_latents_73k.npy via nsd_expdesign (one latent per trial image).
Val split: **held-out 73k images** (~10% unique ids, seed 42) — not random trials.

Legacy (--data averaged --pca-sweep): image-averaged betas only; not used for recon.

Saves to nsd_meta/vdvae_ridge_trials_{subj}.npz (required by reconstruct_vdvae_dual).

Usage:
    python vdvae_regression.py --subj subj01 --root /mnt/mindbridge
    python vdvae_regression.py --subj subj01 --data averaged --pca-sweep
"""

from __future__ import annotations

import argparse
import json
import sys

sys.path = [p for p in sys.path if "share/software" not in p and "jupyterlab" not in p]

import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, normalize

from paths import get_root, resolve_run_id, run_root, update_run_manifest
from train_4head_roi import load_roi_indices
from vdvae_brain_diffuser import NUM_LATENT_LAYERS
from vdvae_trial_data import (
    layer_flat_dims_from_pack_or_ref,
    load_trial_vdvae_targets,
    split_train_val_by_image,
)

ALL_SUBJ = [f"subj{i:02d}" for i in range(1, 9)]
VARIANT = "VDVAE"
PCA_COMPONENTS = (100, 200, 500)
BASELINE_ALPHA = 50000.0
PCA_ALPHA = 1e4
# Layers 0–5 are well-predicted; 6–30 use training mean at decode (selective reconstruction).
N_EARLY_LAYERS = 6


def early_flat_dim(layer_dims: np.ndarray, n_early: int = N_EARLY_LAYERS) -> int:
    return int(layer_dims[:n_early].sum())


def cosine_sim(pred: np.ndarray, target: np.ndarray) -> float:
    pn = normalize(pred, norm="l2")
    tn = normalize(target, norm="l2")
    return float((pn * tn).sum(axis=1).mean())


def latent_metrics(pred: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    mse = float(np.mean((pred - target) ** 2))
    return mse, cosine_sim(pred, target)


def per_layer_cosine(
    pred: np.ndarray, target: np.ndarray, layer_dims: np.ndarray,
) -> list[float]:
    """Val cosine per VDVAE layer (31 hierarchical latents)."""
    out: list[float] = []
    offset = 0
    for i in range(NUM_LATENT_LAYERS):
        d = int(layer_dims[i])
        cos = cosine_sim(pred[:, offset:offset + d], target[:, offset:offset + d])
        out.append(cos)
        offset += d
    return out


def split_train_val(n: int, seed: int = 42, val_frac: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """Random trial split (leaky if multiple trials per image). Prefer split_train_val_by_image."""
    np.random.seed(seed)
    idx = np.random.permutation(n)
    n_val = max(1, int(val_frac * n))
    return idx[n_val:], idx[:n_val]


def normalize_betas(
    Xtr: np.ndarray, Xvl: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-voxel z-score from train rows only (betas already /300 at ingest).

    Matches brain-diffuser vdvae_regression.py except ddof=1 and 1e-6 floor on std.
    """
    vm = Xtr.mean(axis=0, keepdims=True)
    vs = Xtr.std(axis=0, ddof=1, keepdims=True) + 1e-6
    return (Xtr - vm) / vs, (Xvl - vm) / vs, vm, vs


def normalize_betas_subset(
    Xtr: np.ndarray, Xvl: np.ndarray, cols: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sub_tr = Xtr[:, cols]
    sub_vl = Xvl[:, cols]
    vm = sub_tr.mean(0, keepdims=True)
    vs = sub_tr.std(0, keepdims=True) + 1e-6
    return (sub_tr - vm) / vs, (sub_vl - vm) / vs, vm, vs


def component_groups(n_components: int) -> tuple[slice, slice, slice]:
    n1 = n_components // 3
    n2 = n_components // 3
    return slice(0, n1), slice(n1, n1 + n2), slice(n1 + n2, n_components)


def pca_to_latents(pred_pca: np.ndarray, pack: dict) -> np.ndarray:
    whiten = bool(int(pack.get("whiten", 0)))
    pred_w = pred_pca @ pack["pca_components"].astype(np.float32) + pack["pca_mean"].astype(np.float32)
    if whiten:
        return pred_w * pack["latent_scale"].astype(np.float32) + pack["latent_mean"].astype(np.float32)
    return pred_w


def predict_vdvae_latents(betas: np.ndarray, pack: dict) -> np.ndarray:
    """Inference helper used by reconstruct_vdvae_dual.py."""
    mode = str(pack.get("mode", "global_raw"))
    if mode in ("global_raw", "trials_global_raw"):
        x = (betas.astype(np.float32) - pack["voxel_mean"]) / pack["voxel_std"]
        return x @ pack["W"].T + pack["b"]

    x = (betas.astype(np.float32) - pack["voxel_mean"]) / pack["voxel_std"]
    if mode == "global_pca":
        pred_pca = x @ pack["W"].T + pack["b"]
        return pca_to_latents(pred_pca, pack)

    if mode == "trials_early_pca":
        pred_pca = x @ pack["W"].T + pack["b"]
        comps = pack["early_pca_components"].astype(np.float32)
        mean = pack["early_pca_mean"].astype(np.float32)
        early_lat = pred_pca @ comps + mean
        early_dim = int(pack.get("early_flat_dim", early_lat.shape[1]))
        late_mean = pack.get("latent_mean_late")
        if late_mean is None and "train_latent_mean" in pack:
            late_mean = pack["train_latent_mean"][early_dim:]
        layer_dims = pack.get("layer_flat_dims")
        if late_mean is not None and layer_dims is not None:
            full_dim = int(np.asarray(layer_dims, dtype=np.int64).sum())
            out = np.zeros((len(betas), full_dim), dtype=np.float32)
            ed = min(early_dim, early_lat.shape[1])
            out[:, :ed] = early_lat[:, :ed]
            out[:, early_dim:] = np.asarray(late_mean, dtype=np.float32)
            return out
        return early_lat

    if mode == "roi_pca":
        n_comp = int(pack["n_components"])
        pred_pca = np.zeros((len(betas), n_comp), dtype=np.float32)
        for group in ("early", "mid", "high"):
            cols = pack[f"{group}_voxel_idx"].astype(np.int64)
            g0, g1 = int(pack[f"{group}_comp_start"]), int(pack[f"{group}_comp_end"])
            xg = (betas[:, cols].astype(np.float32) - pack[f"{group}_voxel_mean"]) / pack[f"{group}_voxel_std"]
            pred_pca[:, g0:g1] = xg @ pack[f"W_{group}"].T + pack[f"b_{group}"]
        return pca_to_latents(pred_pca, pack)

    raise ValueError(f"Unknown ridge mode: {mode}")


def apply_hybrid_latents(
    pred: np.ndarray,
    pack: dict,
    *,
    n_early: int | None = None,
) -> np.ndarray:
    """
    Layers 0..n_early-1: ridge prediction; layers n_early..30: training-set mean.
    """
    n_early = n_early if n_early is not None else int(pack.get("n_early_layers", N_EARLY_LAYERS))
    layer_dims = pack.get("layer_flat_dims")
    if layer_dims is None:
        raise KeyError("ridge pack missing layer_flat_dims")
    layer_dims = np.asarray(layer_dims, dtype=np.int64)
    early_dim = early_flat_dim(layer_dims, n_early)

    late_mean = pack.get("latent_mean_late")
    if late_mean is None:
        full_mean = pack.get("train_latent_mean")
        if full_mean is not None:
            late_mean = full_mean[early_dim:]
        else:
            raise KeyError(
                "ridge pack missing latent_mean_late — re-run vdvae_regression.py or "
                "call ensure_hybrid_latent_stats()"
            )

    out = pred.astype(np.float32, copy=True)
    out[:, early_dim:] = np.asarray(late_mean, dtype=np.float32)
    return out


def predict_vdvae_latents_hybrid(
    betas: np.ndarray,
    pack: dict,
    *,
    n_early: int | None = None,
) -> np.ndarray:
    pred = predict_vdvae_latents(betas, pack)
    return apply_hybrid_latents(pred, pack, n_early=n_early)


def ensure_hybrid_latent_stats(pack: dict, root: Path, subj: str) -> dict:
    """Fill latent_mean_late from training trials if missing (no re-fit)."""
    if "latent_mean_late" in pack:
        return pack
    from vdvae_trial_data import load_trial_vdvae_targets

    betas, latents, img_ids = load_trial_vdvae_targets(subj, root)
    tr, _ = split_train_val_by_image(img_ids)
    layer_dims = pack.get("layer_flat_dims")
    if layer_dims is None:
        layer_dims = layer_flat_dims_from_pack_or_ref(root)
    if layer_dims is None:
        raise FileNotFoundError("Need layer_flat_dims in ridge pack or vdvae/ref_stats.npz")
    layer_dims = np.asarray(layer_dims, dtype=np.int64)
    n_early = int(pack.get("n_early_layers", N_EARLY_LAYERS))
    early_dim = early_flat_dim(layer_dims, n_early)
    pack = dict(pack)
    pack["layer_flat_dims"] = layer_dims.astype(np.int32)
    pack["n_early_layers"] = np.int32(n_early)
    pack["early_flat_dim"] = np.int32(early_dim)
    pack["latent_mean_late"] = latents[tr].mean(0)[early_dim:].astype(np.float32)
    return pack


def fit_global_raw(
    Xtr_n: np.ndarray, Xvl_n: np.ndarray, Ytr: np.ndarray, Yvl: np.ndarray, *, alpha: float,
    mode: str = "global_raw",
) -> tuple[np.ndarray, dict]:
    ridge = Ridge(alpha=alpha, fit_intercept=True).fit(Xtr_n, Ytr)
    pred_vl = ridge.predict(Xvl_n)
    pack = {
        "mode": mode,
        "W": ridge.coef_.astype(np.float32),
        "b": ridge.intercept_.astype(np.float32),
        "alpha": np.float32(alpha),
        "n_layers": np.int32(NUM_LATENT_LAYERS),
    }
    return pred_vl, pack


def fit_global_pca(
    Xtr_n: np.ndarray,
    Xvl_n: np.ndarray,
    Ytr: np.ndarray,
    Yvl: np.ndarray,
    *,
    n_components: int,
    whiten: bool,
    alpha: float,
) -> tuple[np.ndarray, dict]:
    if whiten:
        scaler = StandardScaler()
        Y_fit = scaler.fit_transform(Ytr)
        latent_mean = scaler.mean_.astype(np.float32)
        latent_scale = scaler.scale_.astype(np.float32)
    else:
        Y_fit = Ytr
        latent_mean = Ytr.mean(0).astype(np.float32)
        latent_scale = np.ones(Ytr.shape[1], dtype=np.float32)

    pca = PCA(n_components=n_components, random_state=42)
    Y_pca_tr = pca.fit_transform(Y_fit)
    ridge = Ridge(alpha=alpha, fit_intercept=True).fit(Xtr_n, Y_pca_tr)
    pred_pca_vl = ridge.predict(Xvl_n)

    pack = {
        "mode": "global_pca",
        "whiten": np.int8(1 if whiten else 0),
        "n_components": np.int32(n_components),
        "latent_mean": latent_mean,
        "latent_scale": latent_scale,
        "pca_components": pca.components_.astype(np.float32),
        "pca_mean": pca.mean_.astype(np.float32),
        "W": ridge.coef_.astype(np.float32),
        "b": ridge.intercept_.astype(np.float32),
        "alpha": np.float32(alpha),
        "n_layers": np.int32(NUM_LATENT_LAYERS),
    }
    pred_vl = pca_to_latents(pred_pca_vl, pack)
    return pred_vl, pack


def fit_roi_pca(
    Xtr: np.ndarray,
    Xvl: np.ndarray,
    Ytr: np.ndarray,
    Yvl: np.ndarray,
    *,
    n_components: int,
    early_idx: np.ndarray,
    mid_idx: np.ndarray,
    high_idx: np.ndarray,
    alpha: float,
    vm: np.ndarray,
    vs: np.ndarray,
) -> tuple[np.ndarray, dict]:
    scaler = StandardScaler()
    Y_w_tr = scaler.fit_transform(Ytr)
    pca = PCA(n_components=n_components, random_state=42)
    Y_pca_tr = pca.fit_transform(Y_w_tr)

    e_sl, m_sl, h_sl = component_groups(n_components)
    groups = {
        "early": (early_idx, e_sl),
        "mid": (mid_idx, m_sl),
        "high": (high_idx, h_sl),
    }

    pack: dict = {
        "mode": "roi_pca",
        "whiten": np.int8(1),
        "n_components": np.int32(n_components),
        "latent_mean": scaler.mean_.astype(np.float32),
        "latent_scale": scaler.scale_.astype(np.float32),
        "pca_components": pca.components_.astype(np.float32),
        "pca_mean": pca.mean_.astype(np.float32),
        "voxel_mean": vm.astype(np.float32),
        "voxel_std": vs.astype(np.float32),
        "alpha": np.float32(alpha),
        "n_layers": np.int32(NUM_LATENT_LAYERS),
    }

    pred_pca_vl = np.zeros((len(Xvl), n_components), dtype=np.float32)
    for name, (vox_idx, comp_sl) in groups.items():
        Xtr_g, Xvl_g, vm_g, vs_g = normalize_betas_subset(Xtr, Xvl, vox_idx)
        ridge = Ridge(alpha=alpha, fit_intercept=True).fit(
            Xtr_g, Y_pca_tr[:, comp_sl],
        )
        pred_pca_vl[:, comp_sl] = ridge.predict(Xvl_g)
        pack[f"{name}_voxel_idx"] = vox_idx.astype(np.int32)
        pack[f"{name}_comp_start"] = np.int32(comp_sl.start)
        pack[f"{name}_comp_end"] = np.int32(comp_sl.stop)
        pack[f"{name}_voxel_mean"] = vm_g.astype(np.float32)
        pack[f"{name}_voxel_std"] = vs_g.astype(np.float32)
        pack[f"W_{name}"] = ridge.coef_.astype(np.float32)
        pack[f"b_{name}"] = ridge.intercept_.astype(np.float32)

    pred_vl = pca_to_latents(pred_pca_vl, pack)
    return pred_vl, pack


def save_ridge_pack(path: Path, pack: dict, *, vm: np.ndarray, vs: np.ndarray, latent_dim: int) -> None:
    out = dict(pack)
    if "voxel_mean" not in out:
        out["voxel_mean"] = vm.astype(np.float32)
        out["voxel_std"] = vs.astype(np.float32)
    out["latent_dim"] = np.int32(latent_dim)
    np.savez(path, **out)


def run_subject_trials(subj: str, root: Path, run_id: str | None = None, *, alpha: float = BASELINE_ALPHA) -> dict:
    """Ridge on trial-level betas → full 31-layer VDVAE latents."""
    if run_id:
        import os
        os.environ["MINDBRIDGE_RUN_ID"] = run_id
    run_id = resolve_run_id(f"vdvae_ridge_trials_{subj}")
    run_base = run_root(root, run_id)
    ckpt_dir = run_base / f"checkpoints_v{VARIANT}" / subj
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    betas, latents, img_ids = load_trial_vdvae_targets(subj, root)
    layer_dims = layer_flat_dims_from_pack_or_ref(root)

    tr, vl = split_train_val_by_image(img_ids)
    _, vl_leaky = split_train_val(len(betas))
    Xtr, Xvl = betas[tr], betas[vl]
    Ytr, Yvl = latents[tr], latents[vl]
    Xtr_n, Xvl_n, vm, vs = normalize_betas(Xtr, Xvl)

    print(f"\n{'='*60}")
    print(f"  VDVAE Ridge (trials, {NUM_LATENT_LAYERS} layers) — {subj}")
    print(f"  run={run_id}  α={alpha:g}")
    print(f"  betas {betas.shape} → latents {latents.shape}")
    print(
        f"  split: image-level (73k id)  train={len(tr)} trials / "
        f"{len(np.unique(img_ids[tr]))} images"
    )
    print(
        f"  val={len(vl)} trials / {len(np.unique(img_ids[vl]))} held-out images"
    )
    print(f"{'='*60}")

    pred_vl, pack = fit_global_raw(
        Xtr_n, Xvl_n, Ytr, Yvl, alpha=alpha, mode="trials_global_raw",
    )
    if layer_dims is not None:
        pack["layer_flat_dims"] = layer_dims.astype(np.int32)
        n_early = N_EARLY_LAYERS
        early_dim = early_flat_dim(layer_dims, n_early)
        train_mean = Ytr.mean(0).astype(np.float32)
        pack["n_early_layers"] = np.int32(n_early)
        pack["early_flat_dim"] = np.int32(early_dim)
        pack["train_latent_mean"] = train_mean
        pack["latent_mean_late"] = train_mean[early_dim:]
    pack["data_source"] = np.array("trials")
    pack["val_split"] = np.array("image_73k")
    pack["voxel_mean"] = vm.astype(np.float32)
    pack["voxel_std"] = vs.astype(np.float32)

    mse, cos = latent_metrics(pred_vl, Yvl)
    pred_leaky = predict_vdvae_latents(betas[vl_leaky], pack)
    _, cos_leaky = latent_metrics(pred_leaky, latents[vl_leaky])
    mse_h, cos_h = mse, cos
    if layer_dims is not None:
        pred_hybrid_vl = apply_hybrid_latents(pred_vl, pack)
        mse_h, cos_h = latent_metrics(pred_hybrid_vl, Yvl)
    layer_cos: list[float] | None = None
    if layer_dims is not None:
        layer_cos = per_layer_cosine(pred_vl, Yvl, layer_dims)
    print(f"\n  Val MSE: {mse:.4f}  Val cosine (held-out images): {cos:.4f}")
    print(
        f"  Val cosine (trial split, same images in train — leaky): {cos_leaky:.4f}  "
        f"({len(vl_leaky)} trials)"
    )
    if layer_dims is not None:
        print(
            f"  Val hybrid (layers 0-{N_EARLY_LAYERS - 1} pred, "
            f"{N_EARLY_LAYERS}-30 train mean): cosine={cos_h:.4f}  MSE={mse_h:.4f}"
        )
    if layer_cos is not None:
        print(f"  Per-layer val cosine (layers 0..{NUM_LATENT_LAYERS - 1}):")
        for i, lc in enumerate(layer_cos):
            print(f"    layer {i:2d}: {lc:.4f}")
        print(f"  Mean per-layer cosine: {float(np.mean(layer_cos)):.4f}")
    else:
        print("  (per-layer metrics skipped — run vdvae_write_ref_dims.py on GPU)")

    out_path = root / "nsd_meta" / f"vdvae_ridge_trials_{subj}.npz"
    save_ridge_pack(out_path, pack, vm=vm, vs=vs, latent_dim=latents.shape[1])
    np.savez(ckpt_dir / "ridge_trials.npz", **np.load(out_path))

    results = {
        "subj": subj,
        "run_id": run_id,
        "data_source": "trials",
        "n_layers": NUM_LATENT_LAYERS,
        "latent_dim": int(latents.shape[1]),
        "alpha": alpha,
        "val_split": "image_73k",
        "n_train_trials": int(len(tr)),
        "n_val_trials": int(len(vl)),
        "n_train_images": int(len(np.unique(img_ids[tr]))),
        "n_val_images": int(len(np.unique(img_ids[vl]))),
        "n_train": int(len(tr)),
        "n_val": int(len(vl)),
        "val_mse": mse,
        "val_cosine": cos,
        "val_cosine_trial_leaky": cos_leaky,
        "val_cosine_hybrid": float(cos_h) if layer_dims is not None else None,
        "val_mse_hybrid": float(mse_h) if layer_dims is not None else None,
        "n_early_layers": N_EARLY_LAYERS,
        "mean_layer_cosine": float(np.mean(layer_cos)) if layer_cos else None,
        "per_layer_cosine": layer_cos,
        "ridge_path": str(out_path),
    }

    update_run_manifest(run_base, run_id=run_id, variant=VARIANT, subjects={subj: results})

    result_path = root / "results" / f"vdvae_ridge_trials_{subj}.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(results, indent=2))
    print(f"  Saved {out_path}")
    print(f"  Saved {result_path}")
    return results


def run_subject_averaged_pca(subj: str, root: Path, run_id: str | None = None) -> dict:
    """Legacy: averaged betas + PCA/ROI sweep → vdvae_ridge_pca_{subj}.npz."""
    avg_dir = root / "nsd_meta" / "averaged"
    betas_path = avg_dir / f"betas_avg_perception_{subj}.npy"
    lat_path = avg_dir / f"vdvae_latents_{subj}.npy"
    if not betas_path.exists():
        raise FileNotFoundError(betas_path)
    if not lat_path.exists():
        raise FileNotFoundError(f"{lat_path} — run vdvae_encoder_inference.py first")

    if run_id:
        import os
        os.environ["MINDBRIDGE_RUN_ID"] = run_id
    run_id = resolve_run_id(f"vdvae_ridge_avg_{subj}")
    run_base = run_root(root, run_id)
    ckpt_dir = run_base / f"checkpoints_v{VARIANT}" / subj
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    betas = np.load(betas_path).astype(np.float32)
    latents = np.load(lat_path).astype(np.float32)
    tr, vl = split_train_val(len(betas))
    Xtr, Xvl = betas[tr], betas[vl]
    Ytr, Yvl = latents[tr], latents[vl]
    Xtr_n, Xvl_n, vm, vs = normalize_betas(Xtr, Xvl)

    print(f"\n{'='*60}")
    print(f"  VDVAE Ridge PCA sweep (averaged) — {subj}")
    print(f"{'='*60}")

    results: dict = {"conditions": {}, "data_source": "averaged"}

    pred_vl, pack_raw = fit_global_raw(Xtr_n, Xvl_n, Ytr, Yvl, alpha=BASELINE_ALPHA)
    mse, cos = latent_metrics(pred_vl, Yvl)
    results["conditions"]["global_raw"] = {"val_cosine": cos, "val_mse": mse}
    save_ridge_pack(avg_dir / f"vdvae_ridge_{subj}.npz", pack_raw, vm=vm, vs=vs, latent_dim=latents.shape[1])
    print(f"  [baseline raw] val cosine={cos:.4f}")

    best_w_pca = (-1.0, None, None)
    for n in PCA_COMPONENTS:
        pred_vl, pack = fit_global_pca(
            Xtr_n, Xvl_n, Ytr, Yvl, n_components=n, whiten=True, alpha=PCA_ALPHA,
        )
        _, cos = latent_metrics(pred_vl, Yvl)
        if cos > best_w_pca[0]:
            best_w_pca = (cos, n, pack)
        print(f"  whiten+PCA n={n}: cosine={cos:.4f}")

    early_idx, mid_idx, high_idx = load_roi_indices(subj, root)
    best_roi = (-1.0, None, None)
    for n in PCA_COMPONENTS:
        pred_vl, pack = fit_roi_pca(
            Xtr, Xvl, Ytr, Yvl, n_components=n,
            early_idx=early_idx, mid_idx=mid_idx, high_idx=high_idx,
            alpha=PCA_ALPHA, vm=vm, vs=vs,
        )
        _, cos = latent_metrics(pred_vl, Yvl)
        if cos > best_roi[0]:
            best_roi = (cos, n, pack)

    candidates = []
    if best_w_pca[2] is not None:
        candidates.append(("global_whiten_pca", best_w_pca[0], best_w_pca[2]))
    if best_roi[2] is not None:
        candidates.append(("roi_whiten_pca", best_roi[0], best_roi[2]))
    best_name, best_cos, best_pack = max(candidates, key=lambda x: x[1])

    pca_path = avg_dir / f"vdvae_ridge_pca_{subj}.npz"
    save_ridge_pack(pca_path, best_pack, vm=vm, vs=vs, latent_dim=latents.shape[1])
    results.update({"best_model": best_name, "best_val_cosine": best_cos, "pca_ridge_path": str(pca_path)})

    (root / "results" / f"vdvae_ridge_{subj}.json").write_text(json.dumps(results, indent=2))
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description="Ridge: betas → VDVAE latents (31 layers).")
    ap.add_argument("--root", type=str, default=None)
    ap.add_argument("--subj", type=str, default="subj01")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--run-id", type=str, default="")
    ap.add_argument(
        "--data", choices=("trials", "averaged"), default="trials",
        help="trials=betas_flat + 73k latents; averaged=unique images only",
    )
    ap.add_argument("--pca-sweep", action="store_true", help="PCA/ROI sweep (requires --data averaged)")
    ap.add_argument("--alpha", type=float, default=BASELINE_ALPHA, help="Ridge alpha for trial mode")
    args = ap.parse_args()

    if args.pca_sweep and args.data != "averaged":
        ap.error("--pca-sweep requires --data averaged")

    root = get_root(args.root)
    run_id = args.run_id or None
    subjects = ALL_SUBJ if args.all else [args.subj]

    for subj in subjects:
        if args.data == "trials":
            run_subject_trials(subj, root, run_id=run_id, alpha=args.alpha)
        elif args.pca_sweep:
            run_subject_averaged_pca(subj, root, run_id=run_id)
        else:
            raise SystemExit("Use --data trials (default) or --data averaged --pca-sweep")


if __name__ == "__main__":
    main()
