# -*- coding: utf-8 -*-
"""Imagery cross-decode helpers: Kneeland row selection, GT CLIP, joint ridge."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from paths import get_root, joint_ridge_ckpt_dir, resolve_variant_checkpoint
from reconstruct_common import build_recon_model, forward_recon, normalize_variant, projector_to_clip768
from reconstruct_imagery_small import find_imagery_png, load_imagery_gt_image, select_imagery_set_indices
from reconstruct_vdvae_dual import select_set_b_eval_indices
from retrieval_metrics import retrieval_2way, retrieval_2way_kneeland_subset


def _clip_image_features(model, pixel_values: torch.Tensor) -> torch.Tensor:
    feat = model.get_image_features(pixel_values=pixel_values)
    if not isinstance(feat, torch.Tensor):
        if hasattr(feat, "image_embeds"):
            feat = feat.image_embeds
        elif hasattr(feat, "pooler_output") and feat.pooler_output is not None:
            feat = feat.pooler_output
        else:
            feat = feat[0]
    return feat


def _as_clip_tensor(feat: torch.Tensor) -> torch.Tensor:
    return F.normalize(feat.float(), dim=-1)


def _eval_row_indices(rows: list[dict], subset: str) -> list[int]:
    subset = subset.lower()
    if subset == "kneeland":
        return select_imagery_set_indices({"averaged": rows}, "A", n=0, all_valid=True) + select_set_b_eval_indices(
            {"averaged": rows},
        )
    if subset.startswith("set_"):
        letter = subset.split("_", 1)[1].upper()
        return select_imagery_set_indices({"averaged": rows}, letter, n=0, all_valid=True)
    raise ValueError(f"Unknown subset {subset!r}")


def _find_imagery_png(root: Path, label: str) -> Path | None:
    return find_imagery_png(root, label)


@torch.no_grad()
def gt_clip_for_imagery_rows(
    root: Path | str,
    subj: str,
    rows: list[dict],
    *,
    device: torch.device | None = None,
) -> np.ndarray:
    """(N, 768) CLIP image embeddings for imagery GT images (Set A PNG or placeholder)."""
    root = get_root(root)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from transformers import CLIPModel, CLIPProcessor

    repo = "openai/clip-vit-large-patch14"
    model = CLIPModel.from_pretrained(repo).to(device).eval()
    processor = CLIPProcessor.from_pretrained(repo)
    imgs = [
        load_imagery_gt_image(root, row.get("label", ""), cue=str(row.get("cue", "")), size=256)
        for row in rows
    ]
    embs = []
    batch_size = 8
    for i in range(0, len(imgs), batch_size):
        batch = imgs[i : i + batch_size]
        pixel = processor(images=batch, return_tensors="pt")["pixel_values"].to(device)
        feat = _as_clip_tensor(_clip_image_features(model, pixel))
        embs.append(feat.cpu().numpy())
    return np.concatenate(embs, axis=0).astype(np.float32)


def _load_joint_ridge_pack(root: Path | str, subj: str, run_id: str | None = None) -> dict:
    root = get_root(root)
    ckpt_dir = joint_ridge_ckpt_dir(root, subj, run_id)
    W = np.load(ckpt_dir / "W_joint.npy").astype(np.float32)
    b = np.load(ckpt_dir / "b_joint.npy").astype(np.float32)
    vm = np.load(ckpt_dir / "voxel_mean.npy").astype(np.float32)
    vs = np.load(ckpt_dir / "voxel_std.npy").astype(np.float32)
    clip_mean = np.load(ckpt_dir / "clip_mean.npy").astype(np.float32)
    clip_std = np.load(ckpt_dir / "clip_std.npy").astype(np.float32)
    return {
        "W": W,
        "b": b,
        "voxel_mean": vm,
        "voxel_std": vs,
        "clip_mean": clip_mean,
        "clip_std": clip_std,
        "ckpt_dir": str(ckpt_dir),
    }


def predict_joint_ridge_clip(betas: np.ndarray, pack: dict) -> np.ndarray:
    vm = np.asarray(pack["voxel_mean"]).reshape(1, -1)
    vs = np.asarray(pack["voxel_std"]).reshape(1, -1)
    x = (betas.astype(np.float32) - vm) / vs
    pred = x @ pack["W"].T + pack["b"]
    clip = pred[:, :768] * pack["clip_std"] + pack["clip_mean"]
    clip = clip / (np.linalg.norm(clip, axis=1, keepdims=True) + 1e-8)
    return clip.astype(np.float32)


@torch.no_grad()
def eval_subj(
    subj: str,
    root: Path | str,
    variant: str,
    run_id: str,
    ckpt_prefer: str = "best_retrieval",
    *,
    subset: str = "kneeland",
    head: str = "projector",
    n_pairs: int = 1000,
    metric: str = "kneeland",
    beta_adapter_run_id: str | None = None,
    clip_source: str = "regression",
) -> dict:
    """Embedding-space Kneeland 2WC for a trained decoder variant."""
    root = get_root(root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    meta = json.loads((root / "nsd_meta" / f"imagery_trial_meta_{subj}.json").read_text())
    rows = meta["averaged"]
    row_idx = _eval_row_indices(rows, subset)
    betas = np.load(root / "nsd_meta" / f"betas_avg_imagery_{subj}.npy")[row_idx].astype(np.float32)

    if beta_adapter_run_id:
        from train_imagery_beta_adapter import apply_imagery_beta_adapter

        betas = apply_imagery_beta_adapter(betas, root, beta_adapter_run_id, subj, device=device)

    gallery_rows = [rows[i] for i in row_idx]
    gt_clip = gt_clip_for_imagery_rows(root, subj, gallery_rows, device=device)

    variant_n = normalize_variant(variant)
    if variant_n == "joint_ridge":
        jp = _load_joint_ridge_pack(root, subj, run_id)
        pred = predict_joint_ridge_clip(betas, jp)
        key = "joint_ridge_clip"
    else:
        ckpt = torch.load(
            resolve_variant_checkpoint(root, variant_n, subj, run_id, prefer=ckpt_prefer),
            map_location=device,
            weights_only=False,
        )
        model = build_recon_model(variant_n, int(ckpt["n_voxels"]), device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        vm = torch.tensor(ckpt["voxel_mean"], dtype=torch.float32, device=device)
        vs = torch.tensor(ckpt["voxel_std"], dtype=torch.float32, device=device)
        b_t = torch.tensor(betas, dtype=torch.float32, device=device)
        b_n = (b_t - vm) / vs
        if head == "projector" and variant_n in ("4H_CTR", "4H_CTR2", "4H_CTR2_1H"):
            out = model(b_n)
            pred_t = projector_to_clip768(out[4], device)
            key = "projector_clip768"
        else:
            pred_t, _ = forward_recon(model, variant_n, b_n, clip_source=clip_source, device=device)
            key = "regression_clip" if head != "projector" else "projector_clip768"
        pred = pred_t.cpu().numpy()

    gt_t = torch.tensor(gt_clip)
    pred_t = torch.tensor(pred)
    k2 = retrieval_2way_kneeland_subset(pred_t, gt_t, list(range(len(row_idx))), n_pairs=n_pairs)
    if metric != "kneeland":
        k2 = {"kneeland_2wc": retrieval_2way(pred_t, gt_t, n_pairs=n_pairs)}
    return {key: {"kneeland_2wc": k2}}


def main() -> None:
    import argparse
    import os

    ap = argparse.ArgumentParser(description="Imagery cross-decode Kneeland eval")
    ap.add_argument("--subj", default=os.environ.get("NSD_SUBJ", "subj01"))
    ap.add_argument("--root", default=os.environ.get("MINDBRIDGE_ROOT", "/mnt/mindbridge"))
    ap.add_argument("--variant", default="4H_CTR2")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--ckpt-prefer", default="best_retrieval")
    ap.add_argument("--subset", default="kneeland")
    ap.add_argument("--head", default="projector")
    ap.add_argument("--metric", default="kneeland")
    ap.add_argument("--n-pairs", type=int, default=1000)
    ap.add_argument("--beta-adapter-run-id", default="")
    args = ap.parse_args()

    res = eval_subj(
        args.subj,
        args.root,
        args.variant,
        args.run_id,
        args.ckpt_prefer,
        subset=args.subset,
        head=args.head,
        n_pairs=args.n_pairs,
        metric=args.metric,
        beta_adapter_run_id=args.beta_adapter_run_id or None,
    )
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
