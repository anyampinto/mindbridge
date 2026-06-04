# -*- coding: utf-8 -*-
"""
Two-stage reconstruction: perception val + imagery Set B.

Stage 1: predicted VAE latents -> SD VAE decoder
Stage 2 (--decoder):
  vd        — Versatile Diffusion img2img (dual contrastive img_proj, default)
  kandinsky — Kandinsky 2.2 prior (768-d regression CLIP) + decoder img2img

Modes:
  --mode perception  — 3 val images (averaged perception betas, 10% split seed 42)
  --mode imagery     — Set B cues (default 3 for kandinsky, all 5 for vd dual)
  --mode both

Outputs (kandinsky):
  reconstructions/{subj}/kandinsky/kandinsky_perception_val3_grid.png
  reconstructions/{subj}/kandinsky/kandinsky_imagery_setB3_grid.png
  results/kandinsky_recon_clip2wc_{subj}.json

Usage:
  python reconstruct.py --subj subj01 --decoder kandinsky --mode both
  python reconstruct.py --subj subj01 --decoder vd --mode both
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path = [p for p in sys.path if "share/software" not in p and "jupyterlab" not in p]

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image

from paths import ensure_runtime_dirs, get_root, resolve_checkpoint, resolve_variant_checkpoint, variant_ckpt_dir
from reconstruct_common import (
    CLIP_SOURCE_CHOICES,
    DEFAULT_RUN_IDS,
    VARIANT_CHOICES,
    build_recon_model,
    forward_recon,
    normalize_variant,
    projector_to_clip768,
)
from reconstruct_imagery_small import (
    eval_recon_clip_2wc,
    load_imagery_gt_image,
    load_kandinsky_pipelines,
    load_nsd_image,
    load_vd_image_variation_pipe,
    print_clip_2wc_metrics,
    save_grid,
    select_imagery_set_indices,
    select_set_b_indices,
    stage1_vae_decode,
    stage2_kandinsky_img2img,
    stage2_vd_img2img,
)
from train_dual_contrastive import MindBridgeContrastive

_OLD_SENTINEL = "clip_head.weight"
_CTR_SENTINEL = "projector.net.0.weight"
DEFAULT_DUAL_RUN = "20260602_053625_train_dual_contrastive_subj01_150ep"
DEFAULT_CTR_RUN = "20260602_034428_train_contrastive_retrieval_subj01_150ep"
DECODER_CHOICES = ("vd", "kandinsky")
IMAGERY_SUBSET_CHOICES = ("set_a", "set_b", "set_c")


def find_latest_contrastive_checkpoint(
    root: Path, subj: str,
) -> tuple[Path, str, dict, str] | None:
    runs_dir = root / "runs"
    if not runs_dir.exists():
        return None

    candidates: list[tuple[str, Path, dict, str]] = []
    for run_dir in runs_dir.iterdir():
        if not run_dir.is_dir():
            continue
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.exists():
            continue
        try:
            manifest = json.loads(manifest_path.read_text())
        except json.JSONDecodeError:
            continue
        variant = str(manifest.get("variant", ""))
        if not variant.startswith("4H_CTR"):
            continue
        if subj not in manifest.get("subjects", {}):
            continue
        ckpt_dir = run_dir / f"checkpoints_v{variant}" / subj
        ckpt_path = None
        for name in ("best_retrieval.pt", "best_clip.pt", "final.pt"):
            p = ckpt_dir / name
            if p.exists():
                ckpt_path = p
                break
        if ckpt_path:
            candidates.append((run_dir.name, ckpt_path, manifest, variant))

    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    run_name, ckpt_path, manifest, variant = candidates[-1]
    return ckpt_path, run_name, manifest.get("subjects", {}).get(subj, {}), variant


def resolve_best_checkpoint(
    root: Path, subj: str, run_id: str | None = None,
) -> tuple[Path, str, dict, str]:
    if run_id:
        for variant in ("4H_CTR2", "4H_CTR", "4H_CTR3"):
            for name in ("best_retrieval.pt", "best_clip.pt", "final.pt"):
                p = variant_ckpt_dir(root, variant, subj, run_id) / name
                if p.exists():
                    mpath = root / "runs" / run_id / "manifest.json"
                    metrics = {}
                    if mpath.exists():
                        metrics = json.loads(mpath.read_text()).get("subjects", {}).get(subj, {})
                    return p, f"run-id {run_id} ({name})", metrics, variant

    result = find_latest_contrastive_checkpoint(root, subj)
    if result:
        ckpt_path, run_name, subj_metrics, variant = result
        return ckpt_path, f"latest contrastive ({run_name})", subj_metrics, variant

    for legacy_name in ("best_retrieval.pt", "best_clip.pt"):
        legacy = root / "checkpoints" / subj / legacy_name
        if legacy.exists():
            return legacy, f"legacy {legacy_name}", {}, "legacy"

    ckpt_path = resolve_checkpoint(root, subj)
    return ckpt_path, f"legacy {ckpt_path.name}", {}, "legacy"


def load_dual_contrastive_model(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt["model_state"]
    if _OLD_SENTINEL in state and _CTR_SENTINEL not in state:
        raise RuntimeError(f"Pre-contrastive checkpoint at {ckpt_path}; train dual contrastive first.")
    if _CTR_SENTINEL not in state:
        raise RuntimeError(f"Not a dual contrastive checkpoint: {ckpt_path}")
    model = MindBridgeContrastive(n_voxels=int(ckpt["n_voxels"])).to(device)
    model.load_state_dict(state)
    model.eval()
    return model, ckpt


def load_variant_model(
    root: Path,
    subj: str,
    variant: str,
    run_id: str,
    ckpt_prefer: str,
    device: torch.device,
) -> tuple[torch.nn.Module, dict, Path, str]:
    ckpt_path = resolve_variant_checkpoint(root, variant, subj, run_id, prefer=ckpt_prefer)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_recon_model(variant, int(ckpt["n_voxels"]), device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    mpath = root / "runs" / run_id / "manifest.json"
    metrics = {}
    if mpath.exists():
        metrics = json.loads(mpath.read_text()).get("subjects", {}).get(subj, {})
    return model, ckpt, ckpt_path, metrics


def _perception_val_indices(n_images: int, n_pick: int, seed: int) -> np.ndarray:
    np.random.seed(42)
    perm = np.random.permutation(n_images)
    n_val = max(1, int(0.1 * n_images))
    val_idx = perm[:n_val]
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(val_idx), size=min(n_pick, len(val_idx)), replace=False)
    return np.sort(val_idx[pick])


def _load_perception_batch(
    root: Path, subj: str, indices: np.ndarray,
) -> tuple[np.ndarray, list[Image.Image], list[str]]:
    avg_betas = np.load(root / "nsd_meta" / "averaged" / f"betas_avg_perception_{subj}.npy")
    td = np.load(root / "nsd_meta" / "averaged" / f"targets_avg_{subj}.npz")
    stim_ids = td["image_id_73k"][indices].astype(int)
    betas = avg_betas[indices]
    gt_imgs, labels = [], []
    with h5py.File(root / "nsd_meta" / "nsd_stimuli.hdf5", "r") as f:
        for i, sid in enumerate(stim_ids):
            img = Image.fromarray(f["imgBrick"][int(sid)]).convert("RGB")
            gt_imgs.append(img.resize((256, 256), Image.LANCZOS))
            labels.append(f"val{i}\nnsd{int(sid)}")
    return betas, gt_imgs, labels


def _load_imagery_batch(
    root: Path,
    subj: str,
    subset: str,
    n: int,
) -> tuple[np.ndarray, list[Image.Image], list[str], list[dict]]:
    """Load averaged imagery betas + GT images for set_a, set_b, or set_c."""
    meta_path = root / "nsd_meta" / f"imagery_trial_meta_{subj}.json"
    avg_path = root / "nsd_meta" / f"betas_avg_imagery_{subj}.npy"
    with open(meta_path) as f:
        meta = json.load(f)
    betas_avg = np.load(avg_path)

    set_id = subset.replace("set_", "").upper()
    require_nsd = set_id == "B"
    rows = select_imagery_set_indices(
        meta, set_id, n=n, all_valid=(n <= 0), require_nsd=require_nsd,
    )
    selected = [meta["averaged"][i] for i in rows]
    betas = betas_avg[rows]
    gt_imgs, labels = [], []
    for row in selected:
        label = row.get("label") or ""
        cue = row.get("cue", "?")
        nsd_id = row.get("nsd_id")
        if set_id == "B" and nsd_id not in (None, 0):
            gt_imgs.append(load_nsd_image(root, int(nsd_id)))
            labels.append(f"{cue}\nnsd{int(nsd_id)}")
        else:
            gt_imgs.append(load_imagery_gt_image(root, label, cue=cue))
            labels.append(f"{cue}\n{label}")
    return betas, gt_imgs, labels, selected


def _load_imagery_setb_batch(
    root: Path, subj: str, n: int,
) -> tuple[np.ndarray, list[Image.Image], list[str], list[dict]]:
    return _load_imagery_batch(root, subj, "set_b", n)


@torch.no_grad()
def reconstruct_batch_vd(
    model: MindBridgeContrastive,
    betas: np.ndarray,
    voxel_mean: np.ndarray,
    voxel_std: np.ndarray,
    device: torch.device,
    *,
    vd_strength: float,
    vd_steps: int,
    seed: int,
    text_weight: float,
    use_text_cond: bool,
) -> tuple[list[Image.Image], list[Image.Image]]:
    normed = (betas - voxel_mean) / voxel_std
    betas_t = torch.tensor(normed, dtype=torch.float32, device=device)
    _ci, _ct, _dino, pred_vae, img_proj, text_proj = model(betas_t)
    clip_768 = projector_to_clip768(img_proj, device)

    from diffusers import AutoencoderKL

    vae = AutoencoderKL.from_pretrained(
        "CompVis/stable-diffusion-v1-4", subfolder="vae", torch_dtype=torch.float16,
    ).to(device).eval()
    stage1_imgs = stage1_vae_decode(vae, pred_vae, device)
    del vae
    torch.cuda.empty_cache()

    vd_pipe = load_vd_image_variation_pipe(device)
    vd_pipe.set_progress_bar_config(disable=True)
    gen = torch.Generator(device=device).manual_seed(seed)

    stage2_imgs = []
    for i in range(len(stage1_imgs)):
        emb = clip_768[i]
        if use_text_cond and text_weight > 0:
            emb = F.normalize(
                (1 - text_weight) * emb + text_weight * text_proj[i].float(), dim=-1,
            )
        stage2_imgs.append(
            stage2_vd_img2img(
                vd_pipe, stage1_imgs[i], emb,
                strength=vd_strength, num_inference_steps=vd_steps, generator=gen,
            )
        )

    del vd_pipe
    torch.cuda.empty_cache()
    return stage1_imgs, stage2_imgs


@torch.no_grad()
def reconstruct_batch_kandinsky(
    model: torch.nn.Module,
    variant: str,
    betas: np.ndarray,
    voxel_mean: np.ndarray,
    voxel_std: np.ndarray,
    device: torch.device,
    *,
    clip_source: str,
    seed: int,
    prior_strength: float,
    prior_steps: int,
    img2img_strength: float,
    decoder_steps: int,
) -> tuple[list[Image.Image], list[Image.Image]]:
    normed = (betas - voxel_mean) / voxel_std
    betas_t = torch.tensor(normed, dtype=torch.float32, device=device)
    pred_clip, pred_vae = forward_recon(
        model, variant, betas_t, clip_source=clip_source, device=device,
    )

    from diffusers import AutoencoderKL

    vae = AutoencoderKL.from_pretrained(
        "CompVis/stable-diffusion-v1-4", subfolder="vae", torch_dtype=torch.float16,
    ).to(device).eval()
    stage1_imgs = stage1_vae_decode(vae, pred_vae, device)
    del vae
    torch.cuda.empty_cache()

    prior_pipe, decoder_pipe = load_kandinsky_pipelines(device)
    prior_pipe.set_progress_bar_config(disable=True)
    decoder_pipe.set_progress_bar_config(disable=True)
    gen = torch.Generator(device=device).manual_seed(seed)

    stage2_imgs = []
    for i in range(len(stage1_imgs)):
        stage2_imgs.append(
            stage2_kandinsky_img2img(
                prior_pipe, decoder_pipe, stage1_imgs[i], pred_clip[i],
                prior_strength=prior_strength,
                prior_steps=prior_steps,
                img2img_strength=img2img_strength,
                decoder_steps=decoder_steps,
                generator=gen,
            )
        )

    del prior_pipe, decoder_pipe
    torch.cuda.empty_cache()
    return stage1_imgs, stage2_imgs


def run_mode(
    mode: str,
    *,
    root: Path,
    subj: str,
    model,
    ckpt: dict,
    device: torch.device,
    out_dir: Path,
    variant: str,
    decoder: str,
    imagery_subset: str,
    n_perception: int,
    n_imagery: int,
    vd_strength: float,
    vd_steps: int,
    seed: int,
    text_weight: float,
    use_text_cond: bool,
    clip_source: str,
    prior_strength: float,
    prior_steps: int,
    kandinsky_strength: float,
    kandinsky_steps: int,
    n_pairs_2wc: int,
    skip_clip_2wc: bool,
) -> dict:
    voxel_mean = np.asarray(ckpt["voxel_mean"]).reshape(1, -1)
    voxel_std = np.asarray(ckpt["voxel_std"]).reshape(1, -1)

    result: dict = {"mode": mode, "imagery_subset": imagery_subset}

    if mode == "perception":
        avg_path = root / "nsd_meta" / "averaged" / f"betas_avg_perception_{subj}.npy"
        n_images = len(np.load(avg_path, mmap_mode="r"))
        idx = _perception_val_indices(n_images, n_perception, seed)
        betas, gt_imgs, labels = _load_perception_batch(root, subj, idx)
        result["indices"] = idx.tolist()
        prefix = "kandinsky" if decoder == "kandinsky" else "dual"
        grid_name = f"{prefix}_perception_val{n_perception}_grid.png"
    else:
        n_pick = n_imagery if n_imagery > 0 else 0
        betas, gt_imgs, labels, selected = _load_imagery_batch(root, subj, imagery_subset, n_pick)
        result["stimuli"] = [
            {
                "set": r.get("set"),
                "cue": r["cue"],
                "nsd_id": r.get("nsd_id"),
                "label": r.get("label"),
            }
            for r in selected
        ]
        prefix = "kandinsky" if decoder == "kandinsky" else "dual"
        set_tag = imagery_subset.replace("set_", "set").upper()
        n_tag = len(selected) if n_pick <= 0 else n_pick
        grid_name = f"{prefix}_imagery_{set_tag}{n_tag}_grid.png"

    print(f"\n{'=' * 60}\n  [{mode}] n={len(gt_imgs)}  labels={labels}\n{'=' * 60}")

    if decoder == "kandinsky":
        stage1_imgs, stage2_imgs = reconstruct_batch_kandinsky(
            model, variant, betas, voxel_mean, voxel_std, device,
            clip_source=clip_source,
            seed=seed,
            prior_strength=prior_strength,
            prior_steps=prior_steps,
            img2img_strength=kandinsky_strength,
            decoder_steps=kandinsky_steps,
        )
        stage2_label = "Stage 2 (Kandinsky img2img)"
        title = f"Kandinsky 2.2 {mode} — {subj} ({variant}, regression CLIP→prior)"
    else:
        stage1_imgs, stage2_imgs = reconstruct_batch_vd(
            model, betas, voxel_mean, voxel_std, device,
            vd_strength=vd_strength,
            vd_steps=vd_steps,
            seed=seed,
            text_weight=text_weight,
            use_text_cond=use_text_cond,
        )
        stage2_label = "Stage 2 (dual VD)"
        title = f"Dual CTR2 {mode} — {subj} ({variant}, img_proj VD)"

    grid_path = out_dir / grid_name
    save_grid(
        gt_imgs, stage1_imgs, stage2_imgs, labels, grid_path,
        title=title, stage2_label=stage2_label,
    )
    result["grid"] = str(grid_path)

    if skip_clip_2wc:
        return result

    print(f"\n  CLIP 2WC on {mode} reconstructions (ViT-L/14 image encoder vs GT images)...")
    m1 = eval_recon_clip_2wc(stage1_imgs, gt_imgs, device, n_pairs=n_pairs_2wc, labels=labels)
    m2 = eval_recon_clip_2wc(stage2_imgs, gt_imgs, device, n_pairs=n_pairs_2wc, labels=labels)
    print_clip_2wc_metrics(m1, "Stage 1 (VAE)")
    print_clip_2wc_metrics(m2, stage2_label)

    result["clip_2wc"] = {"stage1": m1, "stage2": m2}
    return result


MODE_CHOICES = (
    "perception", "imagery", "both",
    "imagery_set_a", "imagery_set_c", "imagery_set_ac",
)


def _modes_and_subsets(mode: str, imagery_subset: str) -> list[tuple[str, str]]:
    """Return (run_key, imagery_subset) pairs to execute."""
    if mode == "perception":
        return [("perception", "")]
    if mode == "imagery":
        return [("imagery", imagery_subset)]
    if mode == "both":
        return [("perception", ""), ("imagery", imagery_subset)]
    if mode == "imagery_set_a":
        return [("imagery", "set_a")]
    if mode == "imagery_set_c":
        return [("imagery", "set_c")]
    if mode == "imagery_set_ac":
        return [("imagery", "set_a"), ("imagery", "set_c")]
    raise ValueError(f"Unknown mode {mode!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subj", default=os.environ.get("NSD_SUBJ", "subj01"))
    parser.add_argument("--root", default=os.environ.get("MINDBRIDGE_ROOT", "/mnt/mindbridge"))
    parser.add_argument("--decoder", default="vd", choices=DECODER_CHOICES)
    parser.add_argument("--mode", default="both", choices=MODE_CHOICES)
    parser.add_argument(
        "--imagery-subset", default="set_b", choices=IMAGERY_SUBSET_CHOICES,
        help="Imagery set when --mode imagery or both (default set_b)",
    )
    parser.add_argument("--variant", default="4H_CTR", choices=VARIANT_CHOICES)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--ckpt-prefer", default="best_retrieval",
                        choices=("final", "best_clip", "best_retrieval"))
    parser.add_argument("--clip-source", default=None, choices=CLIP_SOURCE_CHOICES)
    parser.add_argument("--n-perception", type=int, default=3)
    parser.add_argument("--n-imagery", type=int, default=3,
                        help="Set-B stimuli count (0 = all valid; default 3 for kandinsky)")
    parser.add_argument("--vd-strength", type=float, default=0.6)
    parser.add_argument("--vd-steps", type=int, default=20)
    parser.add_argument("--prior-strength", type=float, default=0.25,
                        help="Kandinsky prior emb2emb strength on brain CLIP")
    parser.add_argument("--prior-steps", type=int, default=25)
    parser.add_argument("--kandinsky-strength", type=float, default=0.3,
                        help="Kandinsky decoder img2img strength")
    parser.add_argument("--kandinsky-steps", type=int, default=50)
    parser.add_argument("--text-weight", type=float, default=0.0)
    parser.add_argument("--n-pairs-2wc", type=int, default=10000)
    parser.add_argument("--skip-clip-2wc", action="store_true", help="Skip CLIP 2WC metrics")
    parser.add_argument("--eval-clip-2wc", action="store_true", help="Compute CLIP 2WC (default except imagery_set_a/c/ac)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    skip_clip_2wc = args.skip_clip_2wc or (
        not args.eval_clip_2wc
        and args.mode in ("imagery_set_a", "imagery_set_c", "imagery_set_ac")
    )

    root = get_root(args.root)
    subj = args.subj
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    variant = normalize_variant(args.variant)
    use_text_cond = args.text_weight > 0

    if args.decoder == "kandinsky":
        run_id = args.run_id or DEFAULT_RUN_IDS.get(variant, DEFAULT_CTR_RUN)
        clip_source = args.clip_source or "regression"
        n_imagery = args.n_imagery
    else:
        run_id = args.run_id or DEFAULT_DUAL_RUN
        clip_source = args.clip_source or "projector"
        if args.mode in ("imagery_set_a", "imagery_set_c", "imagery_set_ac"):
            n_imagery = 0  # all stimuli in set
        else:
            n_imagery = args.n_imagery if args.n_imagery != 3 else 0  # dual set_b: all valid

    os.environ["TRANSFORMERS_CACHE"] = str(root / "hf_cache")
    os.environ["HF_HOME"] = str(root / "hf_cache")
    os.environ["TORCH_HOME"] = str(root / "torch_cache")

    ensure_runtime_dirs(root, subj)
    out_dir = root / "reconstructions" / subj
    if args.decoder == "kandinsky":
        out_dir = out_dir / "kandinsky"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.decoder == "kandinsky":
        model, ckpt, ckpt_path, subj_metrics = load_variant_model(
            root, subj, variant, run_id, args.ckpt_prefer, device,
        )
        ckpt_desc = f"{variant} run-id {run_id} ({ckpt_path.name})"
    else:
        ckpt_path, ckpt_desc, subj_metrics, variant = resolve_best_checkpoint(
            root, subj, run_id=args.run_id or None,
        )
        model, ckpt = load_dual_contrastive_model(ckpt_path, device)

    print(f"\nDecoder:  {args.decoder}")
    print(f"Checkpoint: {ckpt_desc}")
    print(f"  {ckpt_path}")
    print(f"  variant={variant}  epoch={ckpt.get('epoch')}  clip_source={clip_source}")

    run_plan = _modes_and_subsets(args.mode, args.imagery_subset)
    all_results: dict = {
        "subj": subj,
        "decoder": args.decoder,
        "variant": variant,
        "run_id": run_id,
        "checkpoint": str(ckpt_path),
        "clip_source": clip_source,
        "training_metrics": subj_metrics,
    }
    if args.decoder == "kandinsky":
        all_results.update({
            "prior_strength": args.prior_strength,
            "prior_steps": args.prior_steps,
            "kandinsky_strength": args.kandinsky_strength,
            "kandinsky_steps": args.kandinsky_steps,
        })
    else:
        all_results.update({
            "vd_strength": args.vd_strength,
            "vd_steps": args.vd_steps,
        })

    for run_key, subset in run_plan:
        result_key = run_key if not subset or subset == args.imagery_subset else f"{run_key}_{subset}"
        all_results[result_key] = run_mode(
            run_key,
            root=root,
            subj=subj,
            model=model,
            ckpt=ckpt,
            device=device,
            out_dir=out_dir,
            variant=variant,
            decoder=args.decoder,
            imagery_subset=subset or args.imagery_subset,
            n_perception=args.n_perception,
            n_imagery=n_imagery,
            vd_strength=args.vd_strength,
            vd_steps=args.vd_steps,
            seed=args.seed,
            text_weight=args.text_weight,
            use_text_cond=use_text_cond,
            clip_source=clip_source,
            prior_strength=args.prior_strength,
            prior_steps=args.prior_steps,
            kandinsky_strength=args.kandinsky_strength,
            kandinsky_steps=args.kandinsky_steps,
            n_pairs_2wc=args.n_pairs_2wc,
            skip_clip_2wc=skip_clip_2wc,
        )

    if not skip_clip_2wc:
        results_dir = root / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        if args.mode in ("imagery_set_a", "imagery_set_c", "imagery_set_ac"):
            metrics_name = f"dual_recon_clip2wc_{subj}_{args.mode}.json"
        elif args.decoder == "kandinsky":
            metrics_name = f"kandinsky_recon_clip2wc_{subj}.json"
        else:
            metrics_name = f"dual_recon_clip2wc_{subj}.json"
        metrics_path = results_dir / metrics_name
        metrics_path.write_text(json.dumps(all_results, indent=2))
        print(f"\nSaved CLIP 2WC metrics: {metrics_path}")

        imagery_keys = [k for k in all_results if k.startswith("imagery")]
        if imagery_keys or "perception" in all_results:
            print("\n=== Summary CLIP 2WC (image recon vs GT stimulus) ===")
            for key in imagery_keys + (["perception"] if "perception" in all_results else []):
                if key not in all_results or "clip_2wc" not in all_results[key]:
                    continue
                s2 = all_results[key]["clip_2wc"]["stage2"]["clip_2wc"]
                s1 = all_results[key]["clip_2wc"]["stage1"]["clip_2wc"]
                print(f"  {key:20s}  Stage1={s1:.3f}  Stage2={s2:.3f}")

    print(f"\nGrids → {out_dir}")


if __name__ == "__main__":
    main()
