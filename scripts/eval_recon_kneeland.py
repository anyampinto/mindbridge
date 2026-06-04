# -*- coding: utf-8 -*-
"""
Kneeland 2WC on reconstruction CLIP embeddings.

Two protocols:
  - legacy_gt_distractor: cos(recon_i, gt_i) > cos(recon_i, gt_j)  [old MindBridge default]
  - paper_recon_distractor (Appendix A.2): cos(recon_i, gt_i) > cos(recon_j, gt_i)
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path = [p for p in sys.path if "share/software" not in p and "jupyterlab" not in p]

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from eval_imagery_crossdecode import _eval_row_indices, gt_clip_for_imagery_rows
from paths import get_root
from reconstruct_imagery_small import load_imagery_gt_image
from reconstruct_imagery_small import select_imagery_set_indices
from reconstruct_vdvae_dual import select_set_b_eval_indices
from retrieval_metrics import (
    retrieval_2way_kneeland_paper,
    retrieval_2way_kneeland_paper_multisample,
    retrieval_2way_kneeland_subset,
)


def encode_images_clip(
    images: list[Image.Image],
    device: torch.device,
    batch_size: int = 8,
) -> np.ndarray:
    import torch.nn.functional as F
    from transformers import CLIPModel, CLIPProcessor

    from eval_imagery_crossdecode import _as_clip_tensor

    repo = "openai/clip-vit-large-patch14"
    model = CLIPModel.from_pretrained(repo).to(device).eval()
    processor = CLIPProcessor.from_pretrained(repo)
    embs = []
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]
            pixel = processor(images=batch, return_tensors="pt")["pixel_values"].to(device)
            feat = F.normalize(_as_clip_tensor(model.get_image_features(pixel_values=pixel)), dim=-1)
            embs.append(feat.cpu().numpy())
    return np.concatenate(embs, axis=0).astype(np.float32)


def load_imagery_rows(subj: str, root: Path) -> list[dict]:
    meta = json.loads((root / "nsd_meta" / f"imagery_trial_meta_{subj}.json").read_text())
    return meta["averaged"]


def load_kneeland_stage2_recons(
    out_dir: Path,
    rows: list[dict],
    gallery_idx: list[int],
    *,
    require_all: bool = False,
) -> tuple[list[Image.Image], list[int]]:
    """Load gallery recons; return (images, gallery row indices that exist)."""
    imgs: list[Image.Image] = []
    available: list[int] = []
    for gi in gallery_idx:
        row = rows[gi]
        cue = row.get("cue", "?")
        sid = str(row.get("set", "A"))
        p = out_dir / f"stage2_set{sid}" / f"{cue}_stage2.png"
        if not p.exists():
            if require_all:
                raise FileNotFoundError(f"Missing stage2 PNG for gallery stim: {p}")
            print(f"  Warning: no recon for gallery cue={cue} set={sid} ({p.name}), excluded from 2WC gallery")
            continue
        imgs.append(Image.open(p).convert("RGB"))
        available.append(gi)
    return imgs, available


def _gallery_positions(query_row_indices: list[int], gallery_idx: list[int]) -> list[int]:
    key_to_pos = {gallery_idx[j]: j for j in range(len(gallery_idx))}
    return [key_to_pos[qi] for qi in query_row_indices]


def eval_recon_kneeland(
    recon_images: list[Image.Image],
    query_row_indices: list[int],
    subj: str,
    root: Path,
    *,
    subset: str = "kneeland",
    n_pairs: int = 1000,
    device: torch.device | None = None,
    gallery_recon_images: list[Image.Image] | None = None,
    gallery_row_indices: list[int] | None = None,
    recon_samples_per_stim: list[list[Image.Image]] | None = None,
    seed: int = 42,
) -> dict:
    """
    Evaluate query reconstructions with legacy + paper Kneeland 2WC.

    gallery_recon_images: all gallery recons (e.g. 12) in same order as gallery_row_indices.
    If omitted, paper protocol uses only query recons as the distractor pool (underestimates n_g).
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = load_imagery_rows(subj, root)
    gallery_idx = gallery_row_indices or _eval_row_indices(rows, subset)
    gallery_rows = [rows[i] for i in gallery_idx]
    gt_gallery = gt_clip_for_imagery_rows(root, subj, gallery_rows, device=device)
    gt_t = torch.tensor(gt_gallery)

    pred_query = torch.tensor(encode_images_clip(recon_images, device))
    gallery_pos = _gallery_positions(query_row_indices, gallery_idx)

    k2_legacy = retrieval_2way_kneeland_subset(
        pred_query, gt_t, gallery_pos, n_pairs=n_pairs, seed=seed,
    )

    if gallery_recon_images is not None:
        recon_gallery = torch.tensor(encode_images_clip(gallery_recon_images, device))
    else:
        recon_gallery = pred_query
        gallery_idx = query_row_indices

    qpos = _gallery_positions(query_row_indices, gallery_idx)

    if recon_samples_per_stim is not None:
        if len(recon_samples_per_stim) != len(gallery_idx):
            raise ValueError("recon_samples_per_stim must match gallery size")
        k = len(recon_samples_per_stim[0])
        stacked = []
        for stim_imgs in recon_samples_per_stim:
            stacked.append(encode_images_clip(stim_imgs, device))
        recon_samples = torch.tensor(np.stack(stacked, axis=0))  # (n_g, K, d)
        k2_paper = retrieval_2way_kneeland_paper_multisample(
            recon_samples, gt_t, qpos, n_pairs=n_pairs, seed=seed,
        )
    else:
        k2_paper = retrieval_2way_kneeland_paper(
            recon_gallery, gt_t, qpos, n_pairs=n_pairs, seed=seed,
        )

    return {
        "subset": subset,
        "query_row_indices": query_row_indices,
        "n_gallery": len(gallery_idx),
        "kneeland_2wc_gt_distractor": k2_legacy,
        "kneeland_2wc_paper": k2_paper,
        "kneeland_2wc": k2_legacy["kneeland_2wc"],
        "kneeland_2wc_paper_value": k2_paper["kneeland_2wc"],
    }


def load_multisample_gallery_recons(
    seed_dirs: list[Path],
    rows: list[dict],
    gallery_idx: list[int],
) -> list[list[Image.Image]]:
    """Per gallery stimulus: K recon PNGs (one per seed dir), same order as gallery_idx."""
    out: list[list[Image.Image]] = []
    for gi in gallery_idx:
        row = rows[gi]
        cue = row.get("cue", "?")
        sid = str(row.get("set", "A"))
        per_seed: list[Image.Image] = []
        for sd in seed_dirs:
            p = sd / f"stage2_set{sid}" / f"{cue}_stage2.png"
            if not p.exists():
                raise FileNotFoundError(f"Missing multisample recon: {p}")
            per_seed.append(Image.open(p).convert("RGB"))
        out.append(per_seed)
    return out


def eval_kneeland_multisample_from_seed_dirs(
    seed_dirs: list[Path],
    subj: str,
    root: Path,
    *,
    n_pairs: int = 1000,
    seed: int = 42,
    device: torch.device | None = None,
) -> dict:
    """Paper (+ legacy) 2WC using K reconstructions per stimulus (K = len(seed_dirs))."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = load_imagery_rows(subj, root)
    meta = {"averaged": rows}
    gallery_idx_full = _eval_row_indices(rows, "kneeland")
    samples = load_multisample_gallery_recons(seed_dirs, rows, gallery_idx_full)
    gallery_idx_avail = gallery_idx_full

    gallery_rows = [rows[i] for i in gallery_idx_avail]
    gt_gallery = gt_clip_for_imagery_rows(root, subj, gallery_rows, device=device)
    gt_t = torch.tensor(gt_gallery)

    stacked = []
    for stim_imgs in samples:
        stacked.append(encode_images_clip(stim_imgs, device))
    recon_samples = torch.tensor(np.stack(stacked, axis=0))

    row_a = select_imagery_set_indices(meta, "A", n=0, all_valid=True)
    row_b = select_set_b_eval_indices(meta)

    k2_paper_a = retrieval_2way_kneeland_paper_multisample(
        recon_samples, gt_t, _gallery_positions(row_a, gallery_idx_avail), n_pairs=n_pairs, seed=seed,
    )
    k2_paper_b = retrieval_2way_kneeland_paper_multisample(
        recon_samples, gt_t, _gallery_positions(row_b, gallery_idx_avail), n_pairs=n_pairs, seed=seed,
    )
    avg_paper = (k2_paper_a["kneeland_2wc"] + k2_paper_b["kneeland_2wc"]) / 2.0

    single_recons, _ = load_kneeland_stage2_recons(seed_dirs[0], rows, gallery_idx_avail)
    s2a = [
        Image.open(seed_dirs[0] / "stage2_setA" / f"{rows[ri]['cue']}_stage2.png").convert("RGB")
        for ri in row_a
    ]
    s2b = [
        Image.open(seed_dirs[0] / "stage2_setB" / f"{rows[ri]['cue']}_stage2.png").convert("RGB")
        for ri in row_b
    ]
    k2a = eval_recon_kneeland(
        s2a, row_a, subj, root, subset="kneeland", n_pairs=n_pairs, device=device,
        gallery_recon_images=single_recons, gallery_row_indices=gallery_idx_avail, seed=seed,
    )
    k2b = eval_recon_kneeland(
        s2b, row_b, subj, root, subset="kneeland", n_pairs=n_pairs, device=device,
        gallery_recon_images=single_recons, gallery_row_indices=gallery_idx_avail, seed=seed,
    )

    return {
        "n_seeds": len(seed_dirs),
        "seed_dirs": [str(d) for d in seed_dirs],
        "n_gallery": len(gallery_idx_avail),
        "kneeland_2wc_paper_multisample_avg_ab": avg_paper,
        "set_a_paper_multisample": k2_paper_a,
        "set_b_paper_multisample": k2_paper_b,
        "set_a_legacy_seed0": k2a["kneeland_2wc_gt_distractor"],
        "set_b_legacy_seed0": k2b["kneeland_2wc_gt_distractor"],
        "avg_ab_legacy_seed0": (
            k2a["kneeland_2wc_gt_distractor"]["kneeland_2wc"]
            + k2b["kneeland_2wc_gt_distractor"]["kneeland_2wc"]
        ) / 2.0,
    }


def eval_kneeland_ab_from_run_dir(
    out_dir: Path,
    subj: str,
    root: Path,
    *,
    n_pairs: int = 1000,
    device: torch.device | None = None,
    seed: int = 42,
) -> dict:
    """Legacy + paper 2WC for Set A, Set B, and A+B average (12-gallery recons)."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = load_imagery_rows(subj, root)
    meta = {"averaged": rows}
    gallery_idx = _eval_row_indices(rows, "kneeland")
    row_a = select_imagery_set_indices(meta, "A", n=0, all_valid=True)
    row_b = select_set_b_eval_indices(meta)

    gallery_recons, gallery_idx_avail = load_kneeland_stage2_recons(out_dir, rows, gallery_idx)
    if len(gallery_idx_avail) < 2:
        raise FileNotFoundError(f"Need ≥2 gallery recons under {out_dir}, found {len(gallery_idx_avail)}")

    def _query_imgs(row_indices: list[int]) -> list[Image.Image]:
        out: list[Image.Image] = []
        for ri in row_indices:
            row = rows[ri]
            cue = row.get("cue", "?")
            sid = str(row.get("set", "A"))
            p = out_dir / f"stage2_set{sid}" / f"{cue}_stage2.png"
            out.append(Image.open(p).convert("RGB"))
        return out

    k2a = eval_recon_kneeland(
        _query_imgs(row_a), row_a, subj, root,
        subset="kneeland", n_pairs=n_pairs, device=device,
        gallery_recon_images=gallery_recons, gallery_row_indices=gallery_idx_avail, seed=seed,
    )
    k2b = eval_recon_kneeland(
        _query_imgs(row_b), row_b, subj, root,
        subset="kneeland", n_pairs=n_pairs, device=device,
        gallery_recon_images=gallery_recons, gallery_row_indices=gallery_idx_avail, seed=seed,
    )

    def _avg(key: str) -> float:
        return (
            k2a[key]["kneeland_2wc"] + k2b[key]["kneeland_2wc"]
        ) / 2.0

    return {
        "out_dir": str(out_dir),
        "n_gallery_recons": len(gallery_idx_avail),
        "gallery_row_indices": gallery_idx_avail,
        "set_a": k2a,
        "set_b": k2b,
        "avg_ab_gt_distractor": _avg("kneeland_2wc_gt_distractor"),
        "avg_ab_paper": _avg("kneeland_2wc_paper"),
        "n_pairs": n_pairs,
        "seed": seed,
        "note": "11/12 gallery if Set-B cue T missing; paper uses recon-distractor protocol",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subj", default="subj01")
    ap.add_argument("--root", default=None)
    ap.add_argument("--images-npy", default=None)
    ap.add_argument("--query-indices", default=None)
    ap.add_argument("--run-dir", default=None, help="Kneeland A+B run dir with stage2_setA/B")
    ap.add_argument("--subset", default="kneeland")
    ap.add_argument("--n-pairs", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    root = get_root(args.root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.run_dir:
        res = eval_kneeland_ab_from_run_dir(
            Path(args.run_dir), args.subj, root, n_pairs=args.n_pairs, device=device, seed=args.seed,
        )
        print(json.dumps(res, indent=2))
        print(f"\nKNEELAND_2WC_GT_DIST_AVG_AB={res['avg_ab_gt_distractor']:.6f}")
        print(f"KNEELAND_2WC_PAPER_AVG_AB={res['avg_ab_paper']:.6f}")
        return

    if not args.images_npy or not args.query_indices:
        raise ValueError("Provide --run-dir or both --images-npy and --query-indices")

    arr = np.load(args.images_npy)
    images = [Image.fromarray(arr[i]).convert("RGB") for i in range(len(arr))]
    qidx = [int(x) for x in args.query_indices.split(",")]
    res = eval_recon_kneeland(
        images, qidx, args.subj, root, subset=args.subset, n_pairs=args.n_pairs, device=device,
        seed=args.seed,
    )
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
