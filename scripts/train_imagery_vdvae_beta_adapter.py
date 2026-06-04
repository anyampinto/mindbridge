# -*- coding: utf-8 -*-
"""
Beta adapter LOO-trained for VDVAE Stage-1 early layers (0–5), init from CLIP adapter.

Targets: GT VDVAE latents from vdvae_latents_73k (NSD id) or on-the-fly encode of imagery PNGs.
Loss: 1 − cosine on early flat dims after VDVAE ridge on adapted raw betas.

Optional combined finetune: 0.5 × VDVAE_early + 0.5 × CLIP (joint ridge vs GT CLIP).

Per-fold eval (optional): VDVAE S1 + hybrid decode; recon Kneeland for
  Stage 2A joint ridge @ 0.5, Stage 2B prior τ=1.5 @ 0.3.

Usage:
  python train_imagery_vdvae_beta_adapter.py --subj subj01 --eval-recon
  python train_imagery_vdvae_beta_adapter.py --subj subj01 --combined
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys

sys.path = [p for p in sys.path if "share/software" not in p and "jupyterlab" not in p]

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

from eval_imagery_crossdecode import (
    _eval_row_indices,
    _find_imagery_png,
    gt_clip_for_imagery_rows,
    predict_joint_ridge_clip,
    _load_joint_ridge_pack,
)
from paths import get_root, resolve_joint_ridge_run_id, resolve_run_id, run_root, update_run_manifest
from reconstruct_vdvae_dual import (
    load_prior_bundle,
    load_vdvae_ridge,
    reconstruct_stimuli,
    select_set_b_eval_indices,
)
from reconstruct_imagery_small import load_imagery_gt_image, load_vd_image_variation_pipe
from train_imagery_beta_adapter import BetaAdapter, normalize_raw
from vdvae_brain_diffuser import (
    capture_ref_stats,
    decode_flat_latents,
    download_vdvae_weights,
    encode_flat_latents,
    layer_flat_dims,
    load_ema_vae,
)
from vdvae_regression import N_EARLY_LAYERS, early_flat_dim, ensure_hybrid_latent_stats

VARIANT = "ImgVdvaeBetaAdapter"
DEFAULT_CLIP_ADAPTER_RUN = "20260603_230807_imagery_beta_adapter_subj01"
DEFAULT_JOINT_RIDGE_RUN = "20260601_053733_joint_ridge_all8"
DEFAULT_DUAL_RUN = "20260602_053625_train_dual_contrastive_subj01_150ep"


def load_clip_adapter_ckpt(root: Path, subj: str, run_id: str) -> dict:
    path = (
        root / "runs" / run_id
        / "checkpoints_vImgBetaAdapter" / subj / "best_adapter.pt"
    )
    if not path.exists():
        raise FileNotFoundError(f"CLIP adapter not found: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def imagery_row_to_nsd_index(row: dict, image_id_73k: np.ndarray, slot_10k: np.ndarray) -> int | None:
    nid = row.get("nsd_id")
    if nid is not None and int(nid) > 0:
        return int(nid)
    label = row.get("label") or ""
    import re
    m = re.search(r"shared(\d+)", label, re.I)
    if m and slot_10k.size:
        slot = int(m.group(1)) - 1
        hits = np.where(slot_10k == slot)[0]
        if len(hits):
            return int(image_id_73k[hits[0]])
    return None


def load_kneeland_vdvae_targets(
    subj: str,
    root: Path,
    *,
    device: torch.device,
    encode_missing: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[dict], list[int]]:
    """12 Set A+B imagery betas and full GT VDVAE latents (flat 91k)."""
    meta_path = root / "nsd_meta" / f"imagery_trial_meta_{subj}.json"
    meta = json.loads(meta_path.read_text())
    rows_all = meta["averaged"]
    indices = _eval_row_indices(rows_all, "kneeland")
    betas = np.load(root / "nsd_meta" / f"betas_avg_imagery_{subj}.npy")[indices].astype(np.float32)

    lat_73k_path = root / "nsd_meta" / "averaged" / "vdvae_latents_73k.npy"
    if not lat_73k_path.exists():
        raise FileNotFoundError(f"Missing {lat_73k_path} — run vdvae-encode")
    lat_73k = np.load(lat_73k_path, mmap_mode="r")

    td = np.load(root / "nsd_meta" / "averaged" / f"targets_avg_{subj}.npz")
    i73 = td["image_id_73k"].astype(np.int64)
    slot_10k = td["slot_10k"].astype(np.int64) if "slot_10k" in td.files else np.array([], dtype=np.int64)

    ema_vae = preprocess_fn = None
    gt_latents = []
    selected_rows = [rows_all[i] for i in indices]

    for row in selected_rows:
        idx73 = imagery_row_to_nsd_index(row, i73, slot_10k)
        if idx73 is not None:
            gt_latents.append(np.array(lat_73k[idx73], dtype=np.float32))
            continue
        if not encode_missing:
            raise ValueError(f"No 73k index for set={row.get('set')} cue={row.get('cue')}")
        if ema_vae is None:
            download_vdvae_weights(root)
            ema_vae, preprocess_fn, _ = load_ema_vae(root, device)
        label = row.get("label", "")
        png = _find_imagery_png(root, label)
        if png is None:
            raise FileNotFoundError(f"Imagery PNG missing for {label!r}")
        from PIL import Image
        img = np.array(Image.open(png).convert("RGB"))[None]
        enc = encode_flat_latents(ema_vae, preprocess_fn, img, batch_size=1, device=device)
        gt_latents.append(enc[0].astype(np.float32))

    return betas, np.stack(gt_latents), selected_rows, indices


def ridge_pack_torch(ridge: dict, device: torch.device) -> dict:
    return {
        "W": torch.tensor(ridge["W"], dtype=torch.float32, device=device),
        "b": torch.tensor(ridge["b"], dtype=torch.float32, device=device),
        "voxel_mean": torch.tensor(ridge["voxel_mean"], dtype=torch.float32, device=device),
        "voxel_std": torch.tensor(ridge["voxel_std"], dtype=torch.float32, device=device),
        "early_dim": early_flat_dim(
            np.asarray(ridge["layer_flat_dims"], dtype=np.int64), N_EARLY_LAYERS,
        ),
    }


def predict_vdvae_early_torch(raw_betas: torch.Tensor, rp: dict) -> torch.Tensor:
    x = (raw_betas - rp["voxel_mean"]) / rp["voxel_std"]
    full = x @ rp["W"].T + rp["b"]
    return full[:, : rp["early_dim"]]


def joint_ridge_clip_torch(raw_betas: torch.Tensor, jp: dict) -> torch.Tensor:
    vm = jp["voxel_mean"].reshape(1, -1)
    vs = jp["voxel_std"].reshape(1, -1)
    x = (raw_betas - vm) / vs
    pred = x @ jp["W"].T + jp["b"]
    clip = pred[:, :768] * jp["clip_std"] + jp["clip_mean"]
    return F.normalize(clip.float(), dim=-1)


def adapter_to_raw(
    adapter: BetaAdapter,
    img_raw: torch.Tensor,
    vm: torch.Tensor,
    vs: torch.Tensor,
) -> torch.Tensor:
    x = (img_raw - vm) / vs
    out_n = adapter(x)
    return out_n * vs + vm


def cosine_regression_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return 1.0 - F.cosine_similarity(pred, target, dim=-1).mean()


def train_fold(
    adapter: BetaAdapter,
    *,
    img_raw: torch.Tensor,
    gt_early: torch.Tensor,
    gt_clip: torch.Tensor | None,
    train_idx: torch.Tensor,
    adapter_vm: torch.Tensor,
    adapter_vs: torch.Tensor,
    rp: dict,
    jp: dict | None,
    lr: float,
    epochs: int,
    weight_decay: float,
    combined: bool,
) -> list[float]:
    opt = torch.optim.AdamW(adapter.parameters(), lr=lr, weight_decay=weight_decay)
    losses: list[float] = []
    for _ in range(epochs):
        adapter.train()
        opt.zero_grad()
        adapted = adapter_to_raw(adapter, img_raw[train_idx], adapter_vm, adapter_vs)
        pred_e = predict_vdvae_early_torch(adapted, rp)
        loss = cosine_regression_loss(pred_e, gt_early[train_idx])
        if combined and jp is not None and gt_clip is not None:
            pred_c = joint_ridge_clip_torch(adapted, jp)
            loss = 0.5 * loss + 0.5 * cosine_regression_loss(pred_c, gt_clip[train_idx])
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
    return losses


@torch.no_grad()
def eval_early_cosine(
    adapter: BetaAdapter,
    img_raw: torch.Tensor,
    gt_early: torch.Tensor,
    eval_idx: np.ndarray,
    adapter_vm: torch.Tensor,
    adapter_vs: torch.Tensor,
    rp: dict,
) -> float:
    adapter.eval()
    adapted = adapter_to_raw(adapter, img_raw[eval_idx], adapter_vm, adapter_vs)
    pred_e = predict_vdvae_early_torch(adapted, rp)
    return float(F.cosine_similarity(pred_e, gt_early[eval_idx], dim=-1).mean().item())


@torch.no_grad()
def eval_fold_recon_kneeland(
    adapter: BetaAdapter,
    *,
    subj: str,
    root: Path,
    img_raw_np: np.ndarray,
    meta: dict,
    kneeland_idx: list[int],
    adapter_ckpt: dict,
    ridge: dict,
    joint_pack: dict,
    dual_model,
    dual_vm: np.ndarray,
    dual_vs: np.ndarray,
    prior_bundle: dict,
    ema_vae,
    ref_stats,
    layer_dims: np.ndarray,
    vd_pipe,
    device: torch.device,
    row_idx_set_b: list[int],
    vd_strength_jr: float = 0.5,
    vd_strength_pr: float = 0.3,
    prior_temperature: float = 1.5,
) -> dict:
    from eval_recon_kneeland import eval_recon_kneeland
    from train_imagery_beta_adapter import apply_adapter

    fold_ckpt = {
        **adapter_ckpt,
        "model_state": {k: v.cpu() for k, v in adapter.state_dict().items()},
    }
    betas_adapted = apply_adapter(img_raw_np, fold_ckpt, device=device)

    gt_i, labels_i = [], []
    for i in row_idx_set_b:
        row = meta["averaged"][i]
        gt_i.append(load_imagery_gt_image(
            root, row.get("label", ""), cue=str(row.get("cue", "")), size=256,
        ))
        labels_i.append(f"{row.get('cue', '?')}")

    pos_in_kneeland = [kneeland_idx.index(i) for i in row_idx_set_b]
    betas_b = betas_adapted[pos_in_kneeland]

    out: dict = {}
    for label, stage2, vd_s, prior_b, prior_t in (
        ("joint_ridge_vd05", "joint_ridge", vd_strength_jr, None, 1.0),
        ("prior_t15_vd03", "prior", vd_strength_pr, prior_bundle, prior_temperature),
    ):
        s1, s2 = reconstruct_stimuli(
            betas_batch=betas_b,
            gt_imgs=gt_i,
            labels=labels_i,
            ridge=ridge,
            ema_vae=ema_vae,
            ref_stats=ref_stats,
            layer_dims=layer_dims,
            dual_model=dual_model,
            variant="4H_CTR2",
            vd_pipe=vd_pipe,
            device=device,
            vd_strength=vd_s,
            vd_steps=50,
            hybrid_layers=True,
            root=root,
            subj=subj,
            stage2_source=stage2,
            prior_bundle=prior_b,
            prior_temperature=prior_t,
            dual_vm=dual_vm,
            dual_vs=dual_vs,
            joint_ridge_pack=joint_pack if stage2 == "joint_ridge" else None,
        )
        k2 = eval_recon_kneeland(
            s2, row_idx_set_b, subj, root, subset="kneeland", n_pairs=1000, device=device,
        )
        out[label] = k2
        print(f"      recon Kneeland [{label}]: {k2['kneeland_2wc']:.4f} ({100*k2['kneeland_2wc']:.1f}%)")
    return out


def run_training(
    subj: str,
    root: Path,
    *,
    run_id: str | None = None,
    clip_adapter_run_id: str = DEFAULT_CLIP_ADAPTER_RUN,
    joint_ridge_run_id: str = DEFAULT_JOINT_RIDGE_RUN,
    dual_run_id: str = DEFAULT_DUAL_RUN,
    lr: float = 1e-4,
    epochs: int = 20,
    weight_decay: float = 0.1,
    combined: bool = False,
    eval_recon: bool = False,
    seed: int = 42,
) -> dict:
    if run_id:
        os.environ["MINDBRIDGE_RUN_ID"] = run_id
    tag = f"imagery_vdvae_beta_adapter_{subj}" + ("_combined" if combined else "")
    run_id = resolve_run_id(tag)
    run_base = run_root(root, run_id)
    ckpt_dir = run_base / f"checkpoints_v{VARIANT}" / subj
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("VDVAE beta adapter training requires CUDA.")

    init_ckpt = load_clip_adapter_ckpt(root, subj, clip_adapter_run_id)
    init_state = copy.deepcopy(init_ckpt["model_state"])
    vm_np = np.asarray(init_ckpt["voxel_mean"]).reshape(1, -1).astype(np.float32)
    vs_np = np.asarray(init_ckpt["voxel_std"]).reshape(1, -1).astype(np.float32)
    n_voxels = int(init_ckpt["n_voxels"])
    hidden = int(init_ckpt.get("hidden", 1024))

    img_raw_np, gt_latents, kneeland_rows, kneeland_idx = load_kneeland_vdvae_targets(
        subj, root, device=device,
    )
    ridge = load_vdvae_ridge(root, subj)
    ridge = ensure_hybrid_latent_stats(dict(ridge), root, subj)
    layer_dims = np.asarray(ridge["layer_flat_dims"], dtype=np.int64)
    early_dim = early_flat_dim(layer_dims, N_EARLY_LAYERS)
    gt_early_np = gt_latents[:, :early_dim].astype(np.float32)

    img_t = torch.tensor(img_raw_np, dtype=torch.float32, device=device)
    gt_early = torch.tensor(gt_early_np, dtype=torch.float32, device=device)
    adapter_vm = torch.tensor(vm_np, dtype=torch.float32, device=device)
    adapter_vs = torch.tensor(vs_np, dtype=torch.float32, device=device)
    rp = ridge_pack_torch(ridge, device)

    jp = None
    gt_clip_t = None
    if combined:
        jr_pack = _load_joint_ridge_pack(root, subj, resolve_joint_ridge_run_id(joint_ridge_run_id))
        jp = {
            "W": torch.tensor(jr_pack["W"], dtype=torch.float32, device=device),
            "b": torch.tensor(jr_pack["b"], dtype=torch.float32, device=device),
            "voxel_mean": torch.tensor(jr_pack["voxel_mean"], dtype=torch.float32, device=device),
            "voxel_std": torch.tensor(jr_pack["voxel_std"], dtype=torch.float32, device=device),
            "clip_mean": torch.tensor(jr_pack["clip_mean"], dtype=torch.float32, device=device),
            "clip_std": float(jr_pack["clip_std"]),
        }
        gt_clip_np = gt_clip_for_imagery_rows(root, subj, kneeland_rows, device=device)
        gt_clip_t = torch.tensor(gt_clip_np, dtype=torch.float32, device=device)

    n_stim = len(img_raw_np)
    print(f"\n{'='*60}")
    print(f"  VDVAE beta adapter LOO — {subj}  combined={combined}")
    print(f"  run={run_id}  stimuli={n_stim}  early_dim={early_dim}")
    print(f"  init from CLIP adapter: {clip_adapter_run_id}")
    print(f"{'='*60}")

    # Optional recon eval backends (lazy load once)
    recon_backends = None
    if eval_recon:
        from paths import resolve_variant_checkpoint
        from reconstruct_common import build_recon_model

        download_vdvae_weights(root)
        ema_vae, preprocess_fn, _ = load_ema_vae(root, device)
        ref_stats = capture_ref_stats(ema_vae, preprocess_fn, device)
        vd_pipe = load_vd_image_variation_pipe(device)
        dual_ckpt = resolve_variant_checkpoint(root, "4H_CTR2", subj, dual_run_id, prefer="best_retrieval")
        dck = torch.load(dual_ckpt, map_location=device, weights_only=False)
        dual_vm = np.asarray(dck["voxel_mean"]).reshape(1, -1).astype(np.float32)
        dual_vs = np.asarray(dck["voxel_std"]).reshape(1, -1).astype(np.float32)
        dual_model = build_recon_model("4H_CTR2", int(dck["n_voxels"]), device)
        dual_model.load_state_dict(dck["model_state"])
        dual_model.eval()
        prior_bundle = load_prior_bundle(root, subj, device, None)
        joint_pack = _load_joint_ridge_pack(root, subj, resolve_joint_ridge_run_id(joint_ridge_run_id))
        meta = json.loads((root / "nsd_meta" / f"imagery_trial_meta_{subj}.json").read_text())
        row_idx_set_b = select_set_b_eval_indices(meta)
        recon_backends = dict(
            ema_vae=ema_vae, ref_stats=ref_stats, layer_dims=layer_dims,
            vd_pipe=vd_pipe, dual_model=dual_model, dual_vm=dual_vm, dual_vs=dual_vs,
            prior_bundle=prior_bundle, joint_pack=joint_pack, meta=meta,
            row_idx_set_b=row_idx_set_b,
        )

    fold_results: list[dict] = []
    for holdout in range(n_stim):
        tr = np.array([i for i in range(n_stim) if i != holdout], dtype=np.int64)
        adapter = BetaAdapter(n_voxels, hidden=hidden, dropout=float(init_ckpt.get("dropout", 0.1)))
        adapter.load_state_dict(init_state)
        adapter.to(device)

        loss_hist = train_fold(
            adapter,
            img_raw=img_t,
            gt_early=gt_early,
            gt_clip=gt_clip_t,
            train_idx=torch.tensor(tr, dtype=torch.long, device=device),
            adapter_vm=adapter_vm,
            adapter_vs=adapter_vs,
            rp=rp,
            jp=jp,
            lr=lr,
            epochs=epochs,
            weight_decay=weight_decay,
            combined=combined,
        )
        cos = eval_early_cosine(adapter, img_t, gt_early, np.array([holdout]), adapter_vm, adapter_vs, rp)
        row = kneeland_rows[holdout]
        fold_entry = {
            "holdout": f"set{row['set']}_cue{row['cue']}",
            "early_cosine": cos,
            "final_loss": loss_hist[-1] if loss_hist else float("nan"),
        }
        print(f"  LOO holdout {fold_entry['holdout']}: early_cos={cos:.4f}")

        if eval_recon and recon_backends is not None:
            print("    recon Kneeland eval...")
            rek = eval_fold_recon_kneeland(
                adapter,
                subj=subj,
                root=root,
                img_raw_np=img_raw_np,
                meta=recon_backends["meta"],
                kneeland_idx=kneeland_idx,
                adapter_ckpt=init_ckpt,
                ridge=ridge,
                joint_pack=recon_backends["joint_pack"],
                dual_model=recon_backends["dual_model"],
                dual_vm=recon_backends["dual_vm"],
                dual_vs=recon_backends["dual_vs"],
                prior_bundle=recon_backends["prior_bundle"],
                ema_vae=recon_backends["ema_vae"],
                ref_stats=recon_backends["ref_stats"],
                layer_dims=recon_backends["layer_dims"],
                vd_pipe=recon_backends["vd_pipe"],
                device=device,
                row_idx_set_b=recon_backends["row_idx_set_b"],
            )
            fold_entry["recon_kneeland"] = rek

        fold_results.append(fold_entry)

    mean_early = float(np.mean([f["early_cosine"] for f in fold_results]))
    print(f"\n  LOO mean early-layer cosine: {mean_early:.4f}")

    # Full-set train (init from CLIP adapter)
    torch.manual_seed(seed)
    adapter_full = BetaAdapter(n_voxels, hidden=hidden, dropout=float(init_ckpt.get("dropout", 0.1)))
    adapter_full.load_state_dict(init_state)
    adapter_full.to(device)
    all_idx = torch.arange(n_stim, device=device)
    train_fold(
        adapter_full,
        img_raw=img_t,
        gt_early=gt_early,
        gt_clip=gt_clip_t,
        train_idx=all_idx,
        adapter_vm=adapter_vm,
        adapter_vs=adapter_vs,
        rp=rp,
        jp=jp,
        lr=lr,
        epochs=epochs,
        weight_decay=weight_decay,
        combined=combined,
    )
    full_cos = eval_early_cosine(
        adapter_full, img_t, gt_early, np.arange(n_stim), adapter_vm, adapter_vs, rp,
    )
    print(f"  Full-set early cosine: {full_cos:.4f}")

    ckpt_path = ckpt_dir / "best_adapter.pt"
    payload = {
        "model_state": adapter_full.state_dict(),
        "n_voxels": n_voxels,
        "hidden": hidden,
        "dropout": float(init_ckpt.get("dropout", 0.1)),
        "voxel_mean": vm_np,
        "voxel_std": vs_np,
        "align_space": "vdvae_early",
        "combined_clip_ridge": combined,
        "clip_adapter_run_id": clip_adapter_run_id,
        "n_stimuli": n_stim,
        "early_dim": early_dim,
        "loo_mean_early_cosine": mean_early,
        "full_early_cosine": full_cos,
        "loo_folds": fold_results,
        "variant": VARIANT,
        "run_id": run_id,
        "subj": subj,
    }
    torch.save(payload, ckpt_path)
    print(f"  Saved {ckpt_path}")

    full_recon = None
    if eval_recon and recon_backends is not None:
        print("\n  Full adapter recon Kneeland...")
        full_recon = eval_fold_recon_kneeland(
            adapter_full,
            subj=subj,
            root=root,
            img_raw_np=img_raw_np,
            meta=recon_backends["meta"],
            kneeland_idx=kneeland_idx,
            adapter_ckpt=init_ckpt,
            ridge=ridge,
            joint_pack=recon_backends["joint_pack"],
            dual_model=recon_backends["dual_model"],
            dual_vm=recon_backends["dual_vm"],
            dual_vs=recon_backends["dual_vs"],
            prior_bundle=recon_backends["prior_bundle"],
            ema_vae=recon_backends["ema_vae"],
            ref_stats=recon_backends["ref_stats"],
            layer_dims=recon_backends["layer_dims"],
            vd_pipe=recon_backends["vd_pipe"],
            device=device,
            row_idx_set_b=recon_backends["row_idx_set_b"],
        )

    results = {
        "subj": subj,
        "run_id": run_id,
        "combined": combined,
        "loo_mean_early_cosine": mean_early,
        "full_early_cosine": full_cos,
        "checkpoint": str(ckpt_path),
        "clip_adapter_init": clip_adapter_run_id,
        "full_recon_kneeland": full_recon,
    }
    update_run_manifest(run_base, run_id=run_id, variant=VARIANT, subjects={subj: results})
    out_json = root / "results" / f"imagery_vdvae_beta_adapter_{subj}{'_combined' if combined else ''}.json"
    out_json.write_text(json.dumps({**results, "loo_folds": fold_results}, indent=2))
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description="VDVAE early-layer imagery beta adapter (LOO)")
    ap.add_argument("--subj", default=os.environ.get("NSD_SUBJ", "subj01"))
    ap.add_argument("--root", default=os.environ.get("MINDBRIDGE_ROOT", "/mnt/mindbridge"))
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--clip-adapter-run-id", default=DEFAULT_CLIP_ADAPTER_RUN)
    ap.add_argument("--joint-ridge-run-id", default=DEFAULT_JOINT_RIDGE_RUN)
    ap.add_argument("--dual-run-id", default=DEFAULT_DUAL_RUN)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--combined", action="store_true", help="0.5 VDVAE + 0.5 joint-ridge CLIP loss")
    ap.add_argument("--eval-recon", action="store_true", help="Per-fold + full recon Kneeland (slow)")
    args = ap.parse_args()

    run_training(
        args.subj,
        get_root(args.root),
        run_id=args.run_id,
        clip_adapter_run_id=args.clip_adapter_run_id,
        joint_ridge_run_id=args.joint_ridge_run_id,
        dual_run_id=args.dual_run_id,
        lr=args.lr,
        epochs=args.epochs,
        weight_decay=args.weight_decay,
        combined=args.combined,
        eval_recon=args.eval_recon,
    )


if __name__ == "__main__":
    main()
