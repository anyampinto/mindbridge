# -*- coding: utf-8 -*-
"""
Brain-diffuser VDVAE Stage 1 + MindBridge dual-contrastive VD Stage 2.

Stage 1 (default vdvae_ridge): perception-trained VDVAE ridge on betas → VDVAE decoder.
Stage 1 (mlp_vae): same dual MLP vae_head as 4H training → SD 1.4 VAE decode (better for raw imagery).
Stage 2: CLIP 768 → Versatile Diffusion img2img. Sources:
         - joint_ridge: linear CLIP+DINO ridge (62.9% imagery embedding Kneeland)
         - projector / prior / regression: dual-contrastive MLP paths

Reconstructs:
  - 3 perception val images (10% split, seed 42)
  - Set-B imagery cues W/K/B/C/D (5 stimuli)

Outputs:
  reconstructions/{subj}/vdvae_dual/vdvae_perception_val3_grid.png
  reconstructions/{subj}/vdvae_dual/vdvae_imagery_setB5_grid.png

Usage:
    python reconstruct_vdvae_dual.py --subj subj01 --root /mnt/mindbridge \\
        --run-id 20260602_053625_train_dual_contrastive_subj01_150ep
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path = [p for p in sys.path if "share/software" not in p and "jupyterlab" not in p]

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image

from paths import get_root, resolve_joint_ridge_run_id, resolve_variant_checkpoint
from reconstruct_common import (
    build_recon_model,
    forward_recon,
    normalize_variant,
    projector_to_clip768,
)
from reconstruct_imagery_small import (
    VD_DEFAULT_VISION_MIX,
    load_imagery_gt_image,
    save_grid,
    select_imagery_set_indices,
)
from vd_hf_dual import (
    COCORetrievalBank,
    load_coco_retrieval_bank,
    load_hf_vd_dual_guided_pipe,
    load_hf_vd_image_variation_pipe,
    retrieve_top1_captions,
    stage2_hf_vd_dual_brain_text_img2img,
    stage2_hf_vd_dual_img2img,
    stage2_hf_vd_dual_text_only_img2img,
    stage2_hf_vd_img2img,
)
from vd_brain_diffuser import download_vd_pretrained
from vdvae_brain_diffuser import (
    capture_ref_stats,
    decode_flat_latents,
    download_vdvae_weights,
    layer_flat_dims,
    load_ema_vae,
)
from vdvae_trial_data import load_trial_vdvae_targets, pick_val_trial_indices

SET_B_EVAL_CUES = ("W", "K", "B", "C", "D")
DEFAULT_DUAL_RUN = "20260602_053625_train_dual_contrastive_subj01_150ep"
DEFAULT_BETA_ADAPTER_RUN = "20260603_230807_imagery_beta_adapter_subj01"
STAGE2_SOURCES = ("projector", "prior", "prior_blend", "regression", "joint_ridge")
STAGE1_SOURCES = ("vdvae_ridge", "mlp_vae", "neutral")


def resolve_stage1_sources(
    stage1_source: str = "vdvae_ridge",
    *,
    stage1_source_perception: str | None = None,
    stage1_source_imagery: str | None = None,
) -> tuple[str, str]:
    """Per-modality Stage 1; falls back to stage1_source when a split is omitted."""
    s1p = stage1_source_perception or stage1_source
    s1i = stage1_source_imagery or stage1_source
    for label, s in (("perception", s1p), ("imagery", s1i)):
        if s not in STAGE1_SOURCES:
            raise ValueError(f"stage1_source_{label} must be one of {STAGE1_SOURCES}, got {s!r}")
    return s1p, s1i
TEXT_SOURCES = ("", "mlp_text_hf_dual", "mlp_text_only", "caption_retrieval")
DEFAULT_TEXT_TO_IMAGE_STRENGTH = {"caption_retrieval": 0.4, "mlp_text_hf_dual": 0.15, "mlp_text_only": 1.0}
IMAGERY_SET_CHOICES = ("set_b", "set_a", "kneeland")
# Imagery CLIP for VD must come from dual-contrastive MLP (not joint ridge / COCO captions).
DUAL_MLP_VARIANTS = frozenset({"4H_CTR", "4H_CTR2"})


def require_dual_mlp_imagery_path(
    *,
    variant: str,
    stage2_source: str,
    text_source: str,
    eval_recon_kneeland: bool,
) -> None:
    if not eval_recon_kneeland:
        return
    if stage2_source == "joint_ridge":
        raise ValueError(
            "joint_ridge bypasses the dual contrastive MLP; use projector | prior | prior_blend "
            "with variant 4H_CTR2"
        )
    if text_source == "caption_retrieval":
        raise ValueError("caption_retrieval uses COCO captions, not brain→MLP embeddings")
    v = normalize_variant(variant)
    if v not in DUAL_MLP_VARIANTS:
        raise ValueError(f"Imagery eval requires dual MLP variant {DUAL_MLP_VARIANTS}, got {v}")


# HF-only Kneeland recon: Set A + Set B only (no Set C). All paths use 4H_CTR2 MLP.
KNEELAND_HF_CONFIG: dict[str, dict] = {
    "A": {"text_source": "", "stage2_source": "projector", "vd_strength": 0.12},
    "B": {"text_source": "", "stage2_source": "prior", "vd_strength": 0.3},
}
DEFAULT_JOINT_RIDGE_RUN = "20260601_053733_joint_ridge_all8"


def save_stage1_grid(
    gt_imgs: list[Image.Image],
    stage1_imgs: list[Image.Image],
    labels: list[str],
    save_path: Path,
    *,
    title: str,
) -> None:
    n = len(gt_imgs)
    fig, axes = plt.subplots(2, n, figsize=(n * 2.8, 5.5))
    if n == 1:
        axes = np.array([[axes[0]], [axes[1]]])
    for col in range(n):
        axes[0, col].imshow(gt_imgs[col])
        axes[0, col].set_title(labels[col], fontsize=8)
        axes[1, col].imshow(stage1_imgs[col])
        for row in range(2):
            axes[row, col].axis("off")
    axes[0, 0].set_ylabel("Ground truth", fontsize=9, rotation=90, labelpad=10)
    axes[1, 0].set_ylabel("Stage 1 (VDVAE)", fontsize=9, rotation=90, labelpad=10)
    plt.suptitle(title, fontsize=12, y=1.02)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {save_path}")


def save_stage2_grid(
    gt_imgs: list[Image.Image],
    stage2_imgs: list[Image.Image],
    labels: list[str],
    save_path: Path,
    *,
    title: str,
) -> None:
    n = len(gt_imgs)
    fig, axes = plt.subplots(2, n, figsize=(n * 2.8, 5.5))
    if n == 1:
        axes = np.array([[axes[0]], [axes[1]]])
    for col in range(n):
        axes[0, col].imshow(gt_imgs[col])
        axes[0, col].set_title(labels[col], fontsize=8)
        axes[1, col].imshow(stage2_imgs[col])
        for row in range(2):
            axes[row, col].axis("off")
    axes[0, 0].set_ylabel("Ground truth", fontsize=9, rotation=90, labelpad=10)
    axes[1, 0].set_ylabel("Stage 2 (dual VD)", fontsize=9, rotation=90, labelpad=10)
    plt.suptitle(title, fontsize=12, y=1.02)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {save_path}")


def load_vdvae_ridge(root: Path, subj: str) -> dict:
    path = root / "nsd_meta" / f"vdvae_ridge_trials_{subj}.npz"
    if not path.exists():
        raise FileNotFoundError(f"{path} — run vdvae_regression.py (trial-level) first")
    return dict(np.load(path))


@torch.no_grad()
def stage1_mlp_vae_images(
    betas_batch: np.ndarray,
    *,
    dual_model,
    variant: str,
    dual_vm: np.ndarray,
    dual_vs: np.ndarray,
    device: torch.device,
    out_size: int = 512,
) -> list[Image.Image]:
    """Stage 1 from frozen dual MLP vae_head (SD VAE latents), not VDVAE ridge."""
    from diffusers import AutoencoderKL
    from reconstruct_imagery_small import stage1_vae_decode

    betas_n = (betas_batch.astype(np.float32) - dual_vm) / dual_vs
    betas_t = torch.tensor(betas_n, dtype=torch.float32, device=device)
    _, pred_vae = forward_recon(
        dual_model, variant, betas_t, clip_source="regression", device=device,
    )
    vae = AutoencoderKL.from_pretrained(
        "CompVis/stable-diffusion-v1-4", subfolder="vae", torch_dtype=torch.float16,
    ).to(device).eval()
    imgs = stage1_vae_decode(vae, pred_vae, device)
    del vae
    torch.cuda.empty_cache()
    if out_size != 256:
        imgs = [im.resize((out_size, out_size), Image.LANCZOS) for im in imgs]
    return imgs


def stage1_neutral_images(n: int, *, out_size: int = 512) -> list[Image.Image]:
    base = Image.new("RGB", (out_size, out_size), (128, 128, 128))
    return [base.copy() for _ in range(n)]


def predict_vdvae_latents(
    betas: np.ndarray,
    ridge: dict,
    *,
    hybrid: bool = True,
    root: Path | None = None,
    subj: str = "",
) -> np.ndarray:
    from vdvae_regression import (
        ensure_hybrid_latent_stats,
        predict_vdvae_latents as _predict_full,
        predict_vdvae_latents_hybrid,
    )
    if not hybrid:
        return _predict_full(betas, ridge)
    pack = ridge
    if "latent_mean_late" not in pack and root is not None and subj:
        pack = ensure_hybrid_latent_stats(dict(ridge), root, subj)
    return predict_vdvae_latents_hybrid(betas, pack)


def select_set_b_gallery_indices(meta: dict) -> list[int]:
    """All 6 Set-B Kneeland gallery stimuli (incl. cue T; nsd_id may be 0)."""
    from reconstruct_imagery_small import select_imagery_set_indices

    return select_imagery_set_indices(meta, "B", n=0, all_valid=True, require_nsd=False)


def select_set_b_eval_indices(meta: dict) -> list[int]:
    rows = [
        i for i, row in enumerate(meta["averaged"])
        if row.get("set") == "B"
        and str(row.get("cue", "")).upper() in SET_B_EVAL_CUES
    ]
    if len(rows) != len(SET_B_EVAL_CUES):
        found = [meta["averaged"][i].get("cue") for i in rows]
        print(f"  Warning: expected {len(SET_B_EVAL_CUES)} Set-B cues, found {len(rows)}: {found}")
    if not rows:
        raise ValueError(f"No Set-B eval cues {SET_B_EVAL_CUES} in imagery metadata")
    return rows


@torch.no_grad()
def stage2_clip_embeddings(
    betas_batch: np.ndarray,
    *,
    dual_model,
    variant: str,
    device: torch.device,
    stage2_source: str,
    prior_bundle: dict | None = None,
    prior_temperature: float = 1.0,
    dual_vm: np.ndarray | None = None,
    dual_vs: np.ndarray | None = None,
    joint_ridge_pack: dict | None = None,
    prior_seed: int | None = None,
    prior_blend_weight: float = 0.85,
) -> torch.Tensor:
    """(B, 768) CLIP for VD Stage 2."""
    if stage2_source == "joint_ridge":
        if joint_ridge_pack is None:
            raise ValueError("stage2_source=joint_ridge requires joint_ridge_pack")
        from eval_imagery_crossdecode import predict_joint_ridge_clip

        clip_np = predict_joint_ridge_clip(betas_batch.astype(np.float32), joint_ridge_pack)
        print("  Stage2 CLIP: joint ridge (CLIP 768-d, same as 62.9% embedding Kneeland)")
        return torch.tensor(clip_np, dtype=torch.float32, device=device)

    betas_t = torch.tensor(betas_batch, dtype=torch.float32, device=device)
    if dual_vm is not None and dual_vs is not None:
        betas_t = torch.tensor(
            (betas_batch.astype(np.float32) - dual_vm) / dual_vs,
            dtype=torch.float32,
            device=device,
        )

    _, _, _, _, proj, _ = dual_model(betas_t)
    proj = F.normalize(proj.float(), dim=-1)

    if stage2_source == "prior":
        if prior_bundle is None:
            raise ValueError("stage2_source=prior requires prior_bundle")
        from models.diffusion_prior import ddim_sample

        prior = prior_bundle["prior"]
        schedule = prior_bundle["schedule"]
        ddim_steps = int(prior_bundle.get("ddim_steps", 50))
        # VD image_embeddings: diffusion prior DDIM sample (on-manifold CLIP), NOT visual_projection.
        temp = float(prior_temperature)
        clip_for_vd = ddim_sample(
            prior, schedule, proj, ddim_steps=ddim_steps, temperature=temp,
            seed=prior_seed,
        )
        raw_proj_clip = F.normalize(projector_to_clip768(proj, device).float(), dim=-1)
        diag = F.cosine_similarity(clip_for_vd, raw_proj_clip, dim=-1)
        temp_tag = f" τ={temp}" if temp != 1.0 else ""
        print(
            f"  Stage2 CLIP: diffusion prior (DDIM-{ddim_steps}{temp_tag})  "
            f"NOT visual_projection  |  cosim(prior, raw_proj→768) mean={diag.mean():.4f}"
        )
        return clip_for_vd

    if stage2_source == "prior_blend":
        if prior_bundle is None:
            raise ValueError("stage2_source=prior_blend requires prior_bundle")
        from models.diffusion_prior import ddim_sample

        prior = prior_bundle["prior"]
        schedule = prior_bundle["schedule"]
        ddim_steps = int(prior_bundle.get("ddim_steps", 50))
        temp = float(prior_temperature)
        prior_clip = ddim_sample(
            prior, schedule, proj, ddim_steps=ddim_steps, temperature=temp, seed=prior_seed,
        )
        proj_clip = F.normalize(projector_to_clip768(proj, device).float(), dim=-1)
        w = float(np.clip(prior_blend_weight, 0.0, 1.0))
        clip_for_vd = F.normalize(w * prior_clip + (1.0 - w) * proj_clip, dim=-1)
        print(
            f"  Stage2 CLIP: prior_blend w={w:.2f} (DDIM-τ={temp} prior + {1-w:.2f}× projector→768)"
        )
        return clip_for_vd

    if stage2_source == "projector":
        # Off-manifold path: 1024-d projector → frozen visual_projection → 768-d
        clip_for_vd = F.normalize(projector_to_clip768(proj, device).float(), dim=-1)
        print("  Stage2 CLIP: projector → visual_projection (no diffusion prior)")
        return clip_for_vd

    pred_clip, _ = forward_recon(
        dual_model, variant, betas_t, clip_source="regression", device=device,
    )
    print("  Stage2 CLIP: regression head")
    return F.normalize(pred_clip.float(), dim=-1)


@torch.no_grad()
def stage2_text_clip_mlp(
    betas_batch: np.ndarray,
    *,
    dual_model,
    dual_vm: np.ndarray,
    dual_vs: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    """(B, 768) CLIP-text space from dual-contrastive text_projector."""
    betas_t = torch.tensor(
        (betas_batch.astype(np.float32) - dual_vm) / dual_vs,
        dtype=torch.float32,
        device=device,
    )
    *_, text_proj = dual_model(betas_t)
    text_clip = F.normalize(text_proj.float(), dim=-1)
    print("  Stage2 text: MLP text_projector (768-d, dual-guided VD)")
    return text_clip


def load_prior_bundle(
    root: Path,
    subj: str,
    device: torch.device,
    prior_run_id: str | None = None,
    *,
    prior_subj: str | None = None,
) -> dict:
    import sys as _sys
    _repo = Path(__file__).resolve().parent.parent
    if str(_repo) not in _sys.path:
        _sys.path.insert(0, str(_repo))

    from diffusion_prior_io import prior_ckpt_path
    from models.diffusion_prior import DDPMSchedule, DiffusionPrior

    psubj = prior_subj or subj
    path = prior_ckpt_path(root, psubj, prior_run_id)
    if not path.exists():
        legacy = root / "checkpoints_prior" / subj / "best_prior.pt"
        path = legacy if legacy.exists() else path
    if not path.exists():
        raise FileNotFoundError(f"No prior checkpoint at {path}")

    ckpt = torch.load(path, map_location=device, weights_only=False)
    prior = DiffusionPrior().to(device)
    prior.load_state_dict(ckpt["model_state"])
    prior.eval()
    n_steps = int(ckpt.get("num_timesteps", 1000))
    schedule = DDPMSchedule(num_timesteps=n_steps).to(device)
    print(f"  Prior: {path}  val_cosim={ckpt.get('val_cosim', '?')}  ddim={ckpt.get('ddim_steps', 50)}")
    return {
        "prior": prior,
        "schedule": schedule,
        "ddim_steps": int(ckpt.get("ddim_steps", 50)),
        "path": str(path),
    }


def reconstruct_stimuli(
    *,
    betas_batch: np.ndarray,
    gt_imgs: list[Image.Image],
    labels: list[str],
    ridge: dict,
    ema_vae,
    ref_stats,
    layer_dims: np.ndarray,
    dual_model,
    variant: str,
    vd_pipe,
    device: torch.device,
    vd_strength: float,
    vd_steps: int,
    guidance_scale: float = 7.5,
    decode_batch: int = 4,
    hybrid_layers: bool = True,
    root: Path | None = None,
    subj: str = "",
    stage2_source: str = "projector",
    prior_bundle: dict | None = None,
    prior_temperature: float = 1.0,
    dual_vm: np.ndarray | None = None,
    dual_vs: np.ndarray | None = None,
    joint_ridge_pack: dict | None = None,
    text_source: str = "",
    vd_vision_mix: float = VD_DEFAULT_VISION_MIX,
    coco_bank: COCORetrievalBank | None = None,
    text_to_image_strength: float | None = None,
    brain_token_weight: float = 0.85,
    prior_seed: int | None = None,
    vd_seed: int | None = None,
    prior_blend_weight: float = 0.85,
    stage1_source: str = "vdvae_ridge",
) -> tuple[list[Image.Image], list[Image.Image]]:
    if stage1_source == "mlp_vae":
        if dual_model is None or dual_vm is None or dual_vs is None:
            raise ValueError("stage1_source=mlp_vae requires dual_model + voxel norm stats")
        stage1_imgs = stage1_mlp_vae_images(
            betas_batch,
            dual_model=dual_model,
            variant=variant,
            dual_vm=dual_vm,
            dual_vs=dual_vs,
            device=device,
            out_size=512,
        )
    elif stage1_source == "neutral":
        stage1_imgs = stage1_neutral_images(len(betas_batch), out_size=512)
    else:
        pred_lat = predict_vdvae_latents(
            betas_batch, ridge, hybrid=hybrid_layers, root=root, subj=subj,
        )
        stage1_imgs = decode_flat_latents(
            ema_vae, pred_lat, ref_stats, layer_dims=layer_dims,
            out_size=512, batch_size=decode_batch, device=device,
        )

    # (B, 768) brain CLIP fed to VD as encoder_hidden_states (image_embeddings inside stage2_vd_img2img)
    clip_for_vd = stage2_clip_embeddings(
        betas_batch,
        dual_model=dual_model,
        variant=variant,
        device=device,
        stage2_source=stage2_source,
        prior_bundle=prior_bundle,
        prior_temperature=prior_temperature,
        dual_vm=dual_vm,
        dual_vs=dual_vs,
        joint_ridge_pack=joint_ridge_pack,
        prior_seed=prior_seed,
        prior_blend_weight=prior_blend_weight,
    )

    text_for_vd = None
    retrieved: list[str] = []
    retrieval_scores: list[float] = []
    if text_source in ("mlp_text_hf_dual", "mlp_text_only", "caption_retrieval"):
        if dual_model is None or dual_vm is None or dual_vs is None:
            raise ValueError(f"text_source={text_source} requires dual contrastive checkpoint")
        text_for_vd = stage2_text_clip_mlp(
            betas_batch, dual_model=dual_model, dual_vm=dual_vm, dual_vs=dual_vs, device=device,
        )
        if text_source == "caption_retrieval":
            if coco_bank is None:
                raise ValueError("caption_retrieval requires coco_bank")
            retrieved, _, scores = retrieve_top1_captions(text_for_vd, coco_bank, device=device)
            retrieval_scores = scores.tolist()
            for lab, cap, sc in zip(labels, retrieved, retrieval_scores):
                print(f"    [{lab}] retrieved: {cap!r}  cos={sc:.4f}")

    txt_strength = text_to_image_strength
    if txt_strength is None and text_source in DEFAULT_TEXT_TO_IMAGE_STRENGTH:
        txt_strength = DEFAULT_TEXT_TO_IMAGE_STRENGTH[text_source]

    stage2_imgs: list[Image.Image] = []
    for i, s1 in enumerate(stage1_imgs):
        gen = None
        if vd_seed is not None:
            gen = torch.Generator(device=device)
            gen.manual_seed(int(vd_seed) + i)
        if text_source == "caption_retrieval":
            s2 = stage2_hf_vd_dual_img2img(
                vd_pipe, s1, clip_for_vd[i], retrieved[i],
                strength=vd_strength,
                num_inference_steps=vd_steps,
                guidance_scale=guidance_scale,
                text_to_image_strength=float(txt_strength if txt_strength is not None else 0.4),
                generator=gen,
            )
        elif text_source == "mlp_text_hf_dual":
            s2 = stage2_hf_vd_dual_brain_text_img2img(
                vd_pipe, s1, clip_for_vd[i], text_for_vd[i],
                strength=vd_strength,
                num_inference_steps=vd_steps,
                guidance_scale=guidance_scale,
                text_to_image_strength=float(txt_strength if txt_strength is not None else 0.15),
                brain_token_weight=brain_token_weight,
                generator=gen,
            )
        elif text_source == "mlp_text_only":
            s2 = stage2_hf_vd_dual_text_only_img2img(
                vd_pipe, s1, text_for_vd[i],
                strength=vd_strength,
                num_inference_steps=vd_steps,
                guidance_scale=guidance_scale,
                brain_token_weight=brain_token_weight,
                generator=gen,
            )
        else:
            s2 = stage2_hf_vd_img2img(
                vd_pipe, s1, clip_for_vd[i],
                strength=vd_strength,
                num_inference_steps=vd_steps,
                guidance_scale=guidance_scale,
                generator=gen,
            )
        stage2_imgs.append(s2)
    return stage1_imgs, stage2_imgs


def _load_vd_pipe(
    text_source: str,
    device: torch.device,
    root: Path,
    *,
    cache: dict[str, object],
    coco_holder: list[COCORetrievalBank | None],
) -> object:
    if text_source in cache:
        return cache[text_source]
    if text_source in ("caption_retrieval", "mlp_text_hf_dual", "mlp_text_only"):
        if text_source == "caption_retrieval" and coco_holder[0] is None:
            print("  Loading HF VersatileDiffusionDualGuided + COCO caption retrieval bank...")
            coco_holder[0] = load_coco_retrieval_bank(root)
        elif text_source == "mlp_text_only":
            print("  Loading HF VersatileDiffusionDualGuided (Set-C text-only)...")
        elif text_source == "mlp_text_hf_dual":
            print("  Loading HF VersatileDiffusionDualGuided (brain text + prior)...")
        else:
            print("  Loading HF VersatileDiffusionDualGuided...")
        pipe = load_hf_vd_dual_guided_pipe(device)
    else:
        print("  Loading HF VersatileDiffusionImageVariation (vision-only prior)...")
        pipe = load_hf_vd_image_variation_pipe(device)
    cache[text_source] = pipe
    return pipe


def _reconstruct_imagery_set(
    *,
    set_id: str,
    text_source: str,
    meta: dict,
    betas_avg: np.ndarray,
    out_dir: Path,
    subj: str,
    tag: str,
    results: dict,
    eval_recon_kneeland: bool,
    kneeland_n_pairs: int,
    device: torch.device,
    root: Path,
    vd_cache: dict[str, object],
    coco_holder: list[COCORetrievalBank | None],
    ridge,
    ema_vae,
    ref_stats,
    layer_dims,
    dual_model,
    variant: str,
    stage2_source: str,
    prior_bundle,
    prior_temperature: float,
    dual_vm,
    dual_vs,
    joint_ridge_pack,
    vd_strength: float,
    vd_steps: int,
    guidance_scale: float,
    hybrid_layers: bool,
    vd_vision_mix: float,
    text_to_image_strength: float | None,
    brain_token_weight: float,
    prior_seed: int | None,
    vd_seed: int | None,
    beta_adapter_run_id: str | None,
    prior_blend_weight: float = 0.85,
    kneeland_full_set_b: bool = False,
    stage1_source: str = "vdvae_ridge",
    beta_adapter_train_subj: str = "subj01",
) -> None:
    adapter_train = beta_adapter_train_subj or os.environ.get("ADAPTER_TRAIN_SUBJ", subj)
    set_id = set_id.upper()
    if set_id == "B":
        row_idx = (
            select_set_b_gallery_indices(meta) if kneeland_full_set_b
            else select_set_b_eval_indices(meta)
        )
    else:
        row_idx = select_imagery_set_indices(meta, set_id, n=0, all_valid=True)
    betas_raw = betas_avg[row_idx]
    if beta_adapter_run_id:
        from train_imagery_beta_adapter import apply_imagery_beta_adapter, resolve_beta_adapter_path

        adapter_path = resolve_beta_adapter_path(
            root, beta_adapter_run_id, subj, ckpt_subj=adapter_train,
        )
        betas_i = apply_imagery_beta_adapter(
            betas_raw, root, beta_adapter_run_id, subj,
            adapter_train_subj=adapter_train, device=device,
        )
        adapter_ckpt = torch.load(adapter_path, map_location="cpu", weights_only=False)
    else:
        betas_i = betas_raw

    gt_i, labels_i = [], []
    for i in row_idx:
        row = meta["averaged"][i]
        cue = row.get("cue", "?")
        labels_i.append(f"{cue}\n{row.get('label', '')[:20]}")
        gt_i.append(load_imagery_gt_image(
            root, row.get("label", ""), cue=str(row.get("cue", "")), size=256,
        ))

    vis = "HF image-variation" if not text_source else f"HF dual ({text_source})"
    print(
        f"\n  Imagery Set-{set_id} ({len(row_idx)} stimuli, betas={tag}, "
        f"stage1={stage1_source}, stage2={stage2_source}, VD={vd_strength}, {vis})..."
    )
    vd_pipe = _load_vd_pipe(text_source, device, root, cache=vd_cache, coco_holder=coco_holder)
    s1_i, s2_i = reconstruct_stimuli(
        betas_batch=betas_i, gt_imgs=gt_i, labels=labels_i,
        ridge=ridge, ema_vae=ema_vae, ref_stats=ref_stats, layer_dims=layer_dims,
        dual_model=dual_model, variant=variant, vd_pipe=vd_pipe, device=device,
        vd_strength=vd_strength, vd_steps=vd_steps, guidance_scale=guidance_scale,
        hybrid_layers=hybrid_layers, root=root, subj=subj,
        stage2_source=stage2_source, prior_bundle=prior_bundle,
        prior_temperature=prior_temperature,
        dual_vm=dual_vm, dual_vs=dual_vs,
        joint_ridge_pack=joint_ridge_pack,
        text_source=text_source,
        vd_vision_mix=vd_vision_mix,
        coco_bank=coco_holder[0],
        prior_seed=prior_seed,
        vd_seed=vd_seed,
        text_to_image_strength=text_to_image_strength,
        brain_token_weight=brain_token_weight,
        prior_blend_weight=prior_blend_weight,
        stage1_source=stage1_source,
    )

    cues = [meta["averaged"][i].get("cue") for i in row_idx]
    set_lower = set_id.lower()
    grid_i = out_dir / f"imagery_set{set_id}_grid.png"
    temp_note = (
        f", prior_τ={prior_temperature}" if stage2_source == "prior" and prior_temperature != 1.0 else ""
    )
    src_note = f", {stage2_source}" + (f", τ={prior_temperature}" if stage2_source == "prior" and prior_temperature != 1.0 else "")
    text_note = (
        f", HF vision (VD={vd_strength}{src_note})" if not text_source
        else ", HF text-only (MLP)" if text_source == "mlp_text_only"
        else f", {text_source}"
    )
    save_grid(
        gt_i, s1_i, s2_i, labels_i, grid_i,
        title=f"VDVAE+VD imagery Set-{set_id} ({tag}, {subj})",
        stage2_label=f"Stage 2 (VD={vd_strength}, {stage2_source}{temp_note}{text_note})",
    )
    stage2_dir = out_dir / f"stage2_set{set_id}"
    stage2_dir.mkdir(parents=True, exist_ok=True)
    for cue, s2 in zip(cues, s2_i):
        if cue:
            s2.save(stage2_dir / f"{cue}_stage2.png")

    key = f"imagery_set_{set_lower}"
    results[f"{key}_grid"] = str(grid_i)
    results[f"{key}_stage2_dir"] = str(stage2_dir)
    results[f"{key}_cues"] = cues
    results[f"{key}_text_source"] = text_source or "vision_only"
    results[f"{key}_row_indices"] = [int(i) for i in row_idx]

    if eval_recon_kneeland:
        from eval_recon_kneeland import eval_recon_kneeland as _eval_recon_k2

        subset = f"set_{set_lower}"
        k2 = _eval_recon_k2(
            s2_i, row_idx, subj, root, subset=subset, n_pairs=kneeland_n_pairs, device=device,
        )
        results[f"{key}_recon_kneeland_2wc"] = k2
        pct = 100.0 * k2["kneeland_2wc"]
        print(f"\n  Set-{set_id} recon 2WC (subset={subset}, n_gallery={k2['n_gallery']}): "
              f"{k2['kneeland_2wc']:.4f} ({pct:.1f}%)")


def run_reconstruction(
    subj: str,
    root: Path,
    *,
    run_id: str,
    variant: str = "4H_CTR2",
    ckpt_prefer: str = "best_retrieval",
    n_perception: int = 3,
    vd_strength: float = 0.75,
    vd_steps: int = 50,
    guidance_scale: float = 7.5,
    hybrid_layers: bool = True,
    perception_only: bool = False,
    imagery_only: bool = False,
    beta_adapter_run_id: str | None = None,
    stage2_source: str = "projector",
    prior_run_id: str | None = None,
    prior_temperature: float = 1.0,
    out_subdir: str | None = None,
    eval_recon_kneeland: bool = False,
    kneeland_n_pairs: int = 1000,
    text_source: str = "",
    vd_vision_mix: float = VD_DEFAULT_VISION_MIX,
    text_to_image_strength: float | None = None,
    brain_token_weight: float = 0.85,
    prior_seed: int | None = None,
    vd_seed: int | None = None,
    imagery_sets: str = "set_b",
    prior_blend_weight: float = 0.85,
    kneeland_b_vd_strength: float | None = None,
    kneeland_a_text_source: str = "",
    kneeland_b_text_source: str = "",
    stage1_source: str = "vdvae_ridge",
    stage1_source_perception: str | None = None,
    stage1_source_imagery: str | None = None,
    beta_adapter_train_subj: str | None = None,
    prior_subj: str | None = None,
    perception_setb_only: bool = False,
) -> dict:
    root = get_root(root)
    if imagery_sets not in IMAGERY_SET_CHOICES:
        raise ValueError(f"imagery_sets must be one of {IMAGERY_SET_CHOICES} (Set C removed)")
    if stage2_source not in STAGE2_SOURCES:
        raise ValueError(f"stage2_source must be one of {STAGE2_SOURCES}")
    s1_perception, s1_imagery = resolve_stage1_sources(
        stage1_source,
        stage1_source_perception=stage1_source_perception,
        stage1_source_imagery=stage1_source_imagery,
    )
    if stage2_source == "joint_ridge" and (
        s1_perception == "mlp_vae" or s1_imagery == "mlp_vae"
    ):
        raise ValueError("stage1_source=mlp_vae requires dual MLP for stage 2 (not joint_ridge)")
    if text_source not in TEXT_SOURCES:
        raise ValueError(f"text_source must be one of {TEXT_SOURCES} (Brain Diffuser disabled)")
    if text_source == "mlp_text_projector":
        raise ValueError("mlp_text_projector (Brain Diffuser) is disabled; use HF modes only")
    if imagery_sets == "kneeland" and (kneeland_a_text_source or kneeland_b_text_source or text_source):
        print(
            f"  Kneeland text: A={kneeland_a_text_source or '(vision)'}  "
            f"B={kneeland_b_text_source or '(vision)'}"
        )
    if text_source == "caption_retrieval" and stage2_source != "prior":
        raise ValueError("caption_retrieval requires stage2_source=prior (brain prior → VD image_embeddings)")
    if eval_recon_kneeland and not perception_only:
        require_dual_mlp_imagery_path(
            variant=variant,
            stage2_source=stage2_source,
            text_source=text_source or kneeland_a_text_source or kneeland_b_text_source,
            eval_recon_kneeland=True,
        )
    use_joint_ridge = stage2_source == "joint_ridge"
    sub = out_subdir or (
        "prior_adapted" if stage2_source == "prior"
        else "joint_ridge_vdvae" if use_joint_ridge
        else "vdvae_dual"
    )
    out_dir = root / "reconstructions" / subj / sub
    out_dir.mkdir(parents=True, exist_ok=True)

    download_vdvae_weights(root)
    download_vd_pretrained(root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("VDVAE reconstruction requires CUDA.")

    ridge = load_vdvae_ridge(root, subj)
    if hybrid_layers and "latent_mean_late" not in ridge:
        from vdvae_regression import ensure_hybrid_latent_stats
        ridge = ensure_hybrid_latent_stats(dict(ridge), root, subj)
    stage1_tag = (
        "layers 0-5 pred + 6-30 train mean"
        if hybrid_layers
        else "all layers predicted"
    )
    print(f"  Ridge model: mode={ridge.get('mode', 'trials_global_raw')}  "
          f"data={str(ridge.get('data_source', 'trials'))}")
    print(
        f"  Stage 1: perception={s1_perception}  imagery={s1_imagery}  "
        f"(Stage 2={stage2_source}: dual MLP → prior/projector → VD)"
    )
    if s1_perception == "mlp_vae" or s1_imagery == "mlp_vae":
        print("  Stage 1 mlp_vae: dual MLP vae_head → SD 1.4 VAE decode")
    if s1_perception == "vdvae_ridge" or s1_imagery == "vdvae_ridge":
        print(f"  Stage 1 vdvae_ridge: {stage1_tag}")
    if "neutral" in (s1_perception, s1_imagery):
        print("  Stage 1 neutral: gray placeholder (ablation)")
    ema_vae, preprocess_fn, _H = load_ema_vae(root, device)
    ref_stats = capture_ref_stats(ema_vae, preprocess_fn, device)
    if "layer_flat_dims" in ridge:
        layer_dims = ridge["layer_flat_dims"].astype(np.int64)
    else:
        ref_npz = root / "vdvae" / "ref_stats.npz"
        saved_dims = np.load(ref_npz)["layer_flat_dims"] if ref_npz.exists() else None
        layer_dims = saved_dims if saved_dims is not None else layer_flat_dims(ref_stats)
    print(f"  VDVAE flat dim: {int(layer_dims.sum())}")

    joint_ridge_pack = None
    dual_model = None
    dual_vm = dual_vs = None
    ckpt_path = None
    if use_joint_ridge:
        from eval_imagery_crossdecode import _load_joint_ridge_pack

        jr_rid = resolve_joint_ridge_run_id(run_id)
        joint_ridge_pack = _load_joint_ridge_pack(root, subj, jr_rid)
        print(f"  Joint ridge CLIP: {joint_ridge_pack['ckpt_dir']}")
    else:
        variant_norm = normalize_variant(variant)
        ckpt_path = resolve_variant_checkpoint(root, variant_norm, subj, run_id, prefer=ckpt_prefer)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        dual_vm = np.asarray(ckpt["voxel_mean"]).reshape(1, -1).astype(np.float32)
        dual_vs = np.asarray(ckpt["voxel_std"]).reshape(1, -1).astype(np.float32)
        dual_model = build_recon_model(variant_norm, int(ckpt["n_voxels"]), device)
        dual_model.load_state_dict(ckpt["model_state"])
        dual_model.eval()
        print(f"  Dual checkpoint: {ckpt_path} (epoch {ckpt.get('epoch')})")
        variant = variant_norm

    adapter_train = beta_adapter_train_subj or os.environ.get("ADAPTER_TRAIN_SUBJ", subj)
    psubj = prior_subj or subj
    if os.environ.get("USE_POOLED_PRIOR", "0") == "1":
        from diffusion_prior_io import POOLED_PRIOR_SUBJ
        psubj = POOLED_PRIOR_SUBJ

    prior_bundle = None
    if stage2_source in ("prior", "prior_blend"):
        prior_bundle = load_prior_bundle(root, subj, device, prior_run_id, prior_subj=psubj)

    vd_cache: dict[str, object] = {}
    coco_holder: list[COCORetrievalBank | None] = [None]
    vd_pipe = None
    if imagery_sets == "set_b":
        vd_pipe = _load_vd_pipe(text_source, device, root, cache=vd_cache, coco_holder=coco_holder)

    results: dict = {
        "imagery_sets": imagery_sets,
        "subj": subj,
        "run_id": run_id,
        "variant": variant,
        "dual_ckpt": str(ckpt_path) if ckpt_path else None,
        "joint_ridge_run_id": resolve_joint_ridge_run_id(run_id) if use_joint_ridge else None,
        "vd_strength": vd_strength,
        "vd_steps": vd_steps,
        "guidance_scale": guidance_scale,
        "vdvae_ridge_alpha": float(ridge.get("alpha", 50000)),
        "stage2_source": stage2_source,
        "text_source": text_source or None,
        "vd_backend": (
            "hf_image_variation_legacy" if not text_source else "hf_dual_guided"
        ),
        "vd_vision_mix": None,
        "kneeland_hf_config": KNEELAND_HF_CONFIG if imagery_sets == "kneeland" else None,
        "text_to_image_strength": (
            float(text_to_image_strength)
            if text_to_image_strength is not None
            else DEFAULT_TEXT_TO_IMAGE_STRENGTH.get(text_source)
        ) if text_source in ("caption_retrieval", "mlp_text_hf_dual", "mlp_text_only") else None,
        "brain_token_weight": (
            float(brain_token_weight)
            if text_source in ("mlp_text_hf_dual", "mlp_text_only")
            else None
        ),
        "stage1_source": stage1_source,
        "stage1_source_perception": s1_perception,
        "stage1_source_imagery": s1_imagery,
        "stage1_hybrid_layers": hybrid_layers if "vdvae_ridge" in (s1_perception, s1_imagery) else None,
        "n_early_layers": int(ridge.get("n_early_layers", 6)) if "vdvae_ridge" in (s1_perception, s1_imagery) else None,
        "out_dir": str(out_dir),
        "prior_seed": prior_seed,
        "vd_seed": vd_seed,
        "prior_blend_weight": float(prior_blend_weight) if stage2_source == "prior_blend" else None,
    }
    if prior_bundle:
        results["prior_ckpt"] = prior_bundle["path"]
        results["prior_temperature"] = float(prior_temperature)

    if perception_setb_only:
        from imagery_perception_pairs import perception_beta_for_row
        from paths import averaged_meta_dir

        meta_path = root / "nsd_meta" / f"imagery_trial_meta_{subj}.json"
        meta = json.loads(meta_path.read_text())
        row_idx = select_set_b_eval_indices(meta)
        avg_dir = averaged_meta_dir(root)
        perc_avg = np.load(avg_dir / f"betas_avg_perception_{subj}.npy").astype(np.float32)
        td = np.load(avg_dir / f"targets_avg_{subj}.npz")
        i73 = td["image_id_73k"].astype(np.int64)
        slot_10k = td["slot_10k"].astype(np.int64) if "slot_10k" in td.files else np.array([], dtype=np.int64)

        betas_list: list[np.ndarray] = []
        gt_p, labels_p = [], []
        for i in row_idx:
            row = meta["averaged"][i]
            pb = perception_beta_for_row(
                row,
                perc_avg=perc_avg,
                image_id_73k=i73,
                slot_10k=slot_10k,
                vis_avg={},
            )
            if pb is None:
                raise ValueError(f"No perception beta for Set-B row {i} cue={row.get('cue')}")
            betas_list.append(pb)
            cue = row.get("cue", "?")
            labels_p.append(f"{cue}\n{row.get('label', '')[:24]}")
            gt_p.append(load_imagery_gt_image(
                root, row.get("label", ""), cue=str(row.get("cue", "")), size=256,
            ))
        betas_p = np.stack(betas_list, axis=0)

        print(
            f"\n  Perception Set-B Kneeland ({len(row_idx)} stimuli, same cues as imagery Set-B):"
            f"\n    betas=NSD perception avg (no imagery beta adapter)"
            f"\n    stage1={s1_perception}  stage2={stage2_source}  VD={vd_strength}  prior_τ={prior_temperature}"
        )
        vd_pipe = _load_vd_pipe("", device, root, cache=vd_cache, coco_holder=coco_holder)
        s1_p, s2_p = reconstruct_stimuli(
            betas_batch=betas_p, gt_imgs=gt_p, labels=labels_p,
            ridge=ridge, ema_vae=ema_vae, ref_stats=ref_stats, layer_dims=layer_dims,
            dual_model=dual_model, variant=variant, vd_pipe=vd_pipe, device=device,
            vd_strength=vd_strength, vd_steps=vd_steps, guidance_scale=guidance_scale,
            hybrid_layers=hybrid_layers, root=root, subj=subj,
            stage2_source=stage2_source, prior_bundle=prior_bundle,
            prior_temperature=prior_temperature,
            dual_vm=dual_vm, dual_vs=dual_vs,
            joint_ridge_pack=joint_ridge_pack,
            text_source="",
            prior_seed=prior_seed,
            vd_seed=vd_seed,
            stage1_source=s1_perception,
        )
        grid_p = out_dir / "perception_setB_kneeland_grid.png"
        save_grid(
            gt_p, s1_p, s2_p, labels_p, grid_p,
            title=f"Perception Set-B Kneeland ({subj}, prior τ={prior_temperature}, VD={vd_strength})",
            stage2_label=f"Stage 2 (prior, VD={vd_strength})",
        )
        results["perception_setb_grid"] = str(grid_p)
        results["perception_setb_cues"] = [meta["averaged"][i].get("cue") for i in row_idx]
        results["betas_source"] = "betas_avg_perception_matched_to_kneeland_setB"
        results["beta_adapter"] = None
        result_json = root / "results" / f"vdvae_recon_{subj}_{sub}_perception_setb.json"
        result_json.write_text(json.dumps(results, indent=2))
        print(f"  Saved {grid_p}")
        return results

    if not imagery_only and imagery_sets != "kneeland":
        # ── Perception val (individual trials, not image-averaged betas) ─────
        betas_all, _, img_ids_all = load_trial_vdvae_targets(subj, root)
        trial_idx = pick_val_trial_indices(
            len(betas_all), n_perception, image_ids=img_ids_all,
        )
        betas_p = betas_all[trial_idx]
        stim_ids = img_ids_all[trial_idx].astype(int)
        gt_p, labels_p = [], []
        with h5py.File(root / "nsd_meta" / "nsd_stimuli.hdf5", "r") as f:
            for i, sid in enumerate(stim_ids):
                img = Image.fromarray(f["imgBrick"][int(sid)]).convert("RGB")
                gt_p.append(img.resize((256, 256), Image.LANCZOS))
                labels_p.append(f"trial{trial_idx[i]}\nnsd{int(sid)}")

        print(f"\n  Perception val ({len(trial_idx)} unaveraged trials)...")
        s1_p, s2_p = reconstruct_stimuli(
            betas_batch=betas_p, gt_imgs=gt_p, labels=labels_p,
            ridge=ridge, ema_vae=ema_vae, ref_stats=ref_stats, layer_dims=layer_dims,
            dual_model=dual_model, variant=variant, vd_pipe=vd_pipe, device=device,
            vd_strength=vd_strength, vd_steps=vd_steps, guidance_scale=guidance_scale,
            hybrid_layers=hybrid_layers, root=root, subj=subj,
            stage2_source=stage2_source, prior_bundle=prior_bundle,
            prior_temperature=prior_temperature,
            dual_vm=dual_vm, dual_vs=dual_vs,
            joint_ridge_pack=joint_ridge_pack,
            text_source=text_source,
            vd_vision_mix=vd_vision_mix,
            coco_bank=coco_holder[0],
            text_to_image_strength=text_to_image_strength,
            brain_token_weight=brain_token_weight,
            prior_seed=prior_seed,
            vd_seed=vd_seed,
            prior_blend_weight=prior_blend_weight,
            stage1_source=s1_perception,
        )
        grid_p = out_dir / "vdvae_perception_val3_grid.png"
        ckpt_tag = ckpt_path.name if ckpt_path else "joint_ridge"
        save_grid(
            gt_p, s1_p, s2_p, labels_p, grid_p,
            title=f"VDVAE+VD perception val ({subj}, {ckpt_tag})",
            stage2_label=f"Stage 2 (VD strength={vd_strength})",
        )
        results["perception_grid"] = str(grid_p)
        s1_p_path = out_dir / "vdvae_perception_val3_stage1.png"
        save_stage1_grid(
            gt_p, s1_p, labels_p, s1_p_path,
            title=f"Stage 1 only — perception val ({subj}, hybrid ridge)",
        )
        results["perception_stage1_grid"] = str(s1_p_path)
        s2_p_path = out_dir / "vdvae_perception_val3_stage2.png"
        s2_title = (
            f"Stage 2 only — perception val ({subj}, prior τ={prior_temperature}, VD={vd_strength})"
            if stage2_source == "prior"
            else f"Stage 2 only — perception val ({subj}, {stage2_source}, VD={vd_strength})"
        )
        save_stage2_grid(
            gt_p, s2_p, labels_p, s2_p_path,
            title=s2_title,
        )
        results["perception_stage2_grid"] = str(s2_p_path)
        results["perception_trials"] = [int(i) for i in trial_idx]
        results["perception_nsd_ids"] = [int(s) for s in stim_ids]

    if perception_only:
        result_json = root / "results" / f"vdvae_dual_recon_{subj}_perception.json"
        result_json.parent.mkdir(parents=True, exist_ok=True)
        result_json.write_text(json.dumps(results, indent=2))
        print(f"  Metrics saved {result_json}")
        return results

    meta_path = root / "nsd_meta" / f"imagery_trial_meta_{subj}.json"
    meta = json.loads(meta_path.read_text())
    betas_avg = np.load(root / "nsd_meta" / f"betas_avg_imagery_{subj}.npy")

    if imagery_sets in ("set_a", "kneeland"):
        if beta_adapter_run_id:
            from train_imagery_beta_adapter import resolve_beta_adapter_path

            adapter_path = resolve_beta_adapter_path(root, beta_adapter_run_id, subj)
            print(f"\n  Beta adapter: {adapter_path}")
            ad_meta = torch.load(adapter_path, map_location="cpu", weights_only=False)
            hp = ad_meta.get("honest_protocol") or ""
            if hp and hp != "full_fit":
                print(f"  Adapter honest_protocol={hp} (final fit excluded Kneeland identities)")
                results["beta_adapter_honest_protocol"] = hp
            results["beta_adapter_run_id"] = beta_adapter_run_id
            results["beta_adapter_ckpt"] = str(adapter_path)
            tag = "adapted"
        else:
            tag = "raw"
        if imagery_sets == "kneeland":
            sets_to_run = ["A", "B"]
        else:
            sets_to_run = ["A"]
        for sid in sets_to_run:
            cfg = KNEELAND_HF_CONFIG[sid]
            ts = cfg["text_source"]
            if sid == "A" and kneeland_a_text_source:
                ts = kneeland_a_text_source
            elif sid == "B":
                if kneeland_b_text_source:
                    ts = kneeland_b_text_source
                elif text_source:
                    ts = text_source
                else:
                    ts = cfg["text_source"]
            s2_src = cfg["stage2_source"]
            vd_s = float(cfg["vd_strength"])
            if sid == "B" and kneeland_b_vd_strength is not None:
                vd_s = float(kneeland_b_vd_strength)
            _reconstruct_imagery_set(
                set_id=sid,
                text_source=ts,
                meta=meta,
                betas_avg=betas_avg,
                out_dir=out_dir,
                subj=subj,
                tag=tag,
                results=results,
                eval_recon_kneeland=eval_recon_kneeland and imagery_sets != "kneeland",
                kneeland_n_pairs=kneeland_n_pairs,
                device=device,
                root=root,
                vd_cache=vd_cache,
                coco_holder=coco_holder,
                ridge=ridge,
                ema_vae=ema_vae,
                ref_stats=ref_stats,
                layer_dims=layer_dims,
                dual_model=dual_model,
                variant=variant,
                stage2_source=s2_src,
                prior_bundle=prior_bundle,
                prior_temperature=prior_temperature,
                dual_vm=dual_vm,
                dual_vs=dual_vs,
                joint_ridge_pack=joint_ridge_pack,
                vd_strength=vd_s,
                vd_steps=vd_steps,
                guidance_scale=guidance_scale,
                hybrid_layers=hybrid_layers,
                vd_vision_mix=vd_vision_mix,
                text_to_image_strength=text_to_image_strength,
                brain_token_weight=brain_token_weight,
                prior_seed=prior_seed,
                vd_seed=vd_seed,
                prior_blend_weight=prior_blend_weight,
                beta_adapter_run_id=beta_adapter_run_id,
                kneeland_full_set_b=(imagery_sets == "kneeland"),
                stage1_source=s1_imagery,
                beta_adapter_train_subj=adapter_train,
            )
        if eval_recon_kneeland and imagery_sets == "kneeland":
            from eval_imagery_crossdecode import _eval_row_indices
            from eval_recon_kneeland import encode_images_clip, eval_recon_kneeland as _eval_recon_k2

            rows = meta["averaged"]
            row_a = select_imagery_set_indices(meta, "A", n=0, all_valid=True)
            row_b = select_set_b_eval_indices(meta)

            def _load_stage2_pngs(row_indices: list[int], set_letter: str) -> list[Image.Image]:
                imgs = []
                for ri in row_indices:
                    cue = rows[ri].get("cue", "?")
                    p = out_dir / f"stage2_set{set_letter}" / f"{cue}_stage2.png"
                    imgs.append(Image.open(p).convert("RGB"))
                return imgs

            def _gt_pngs(row_indices: list[int]) -> list[Image.Image]:
                out = []
                for ri in row_indices:
                    row = rows[ri]
                    out.append(load_imagery_gt_image(
                        root, row.get("label", ""), cue=str(row.get("cue", "")), size=256,
                    ))
                return out

            from eval_recon_kneeland import load_kneeland_stage2_recons

            gallery_idx_full = _eval_row_indices(rows, "kneeland")
            gallery_recons, gallery_idx = load_kneeland_stage2_recons(
                out_dir, rows, gallery_idx_full,
            )

            s2_a = _load_stage2_pngs(row_a, "A")
            s2_b = _load_stage2_pngs(row_b, "B")
            k2a = _eval_recon_k2(
                s2_a, row_a, subj, root, subset="kneeland", n_pairs=kneeland_n_pairs, device=device,
                gallery_recon_images=gallery_recons, gallery_row_indices=gallery_idx,
            )
            k2b = _eval_recon_k2(
                s2_b, row_b, subj, root, subset="kneeland", n_pairs=kneeland_n_pairs, device=device,
                gallery_recon_images=gallery_recons, gallery_row_indices=gallery_idx,
            )
            k2_avg_gt = (
                k2a["kneeland_2wc_gt_distractor"]["kneeland_2wc"]
                + k2b["kneeland_2wc_gt_distractor"]["kneeland_2wc"]
            ) / 2.0
            k2_avg_paper = (
                k2a["kneeland_2wc_paper"]["kneeland_2wc"]
                + k2b["kneeland_2wc_paper"]["kneeland_2wc"]
            ) / 2.0
            results["recon_kneeland_2wc_set_a"] = k2a
            results["recon_kneeland_2wc_set_b"] = k2b
            results["recon_kneeland_2wc_avg_ab"] = {
                "kneeland_2wc": k2_avg_gt,
                "n_queries": k2a["kneeland_2wc_gt_distractor"]["n_stimuli"]
                + k2b["kneeland_2wc_gt_distractor"]["n_stimuli"],
            }
            results["recon_kneeland_2wc_avg_ab_paper"] = {
                "kneeland_2wc": k2_avg_paper,
                "protocol": "paper_recon_distractor",
            }
            results["recon_kneeland_2wc"] = k2b["kneeland_2wc_gt_distractor"]
            results["recon_kneeland_2wc_paper"] = k2b["kneeland_2wc_paper"]
            print(
                f"\n  Set-A 2WC legacy (GT distractor): "
                f"{k2a['kneeland_2wc_gt_distractor']['kneeland_2wc']:.4f} "
                f"({100 * k2a['kneeland_2wc_gt_distractor']['kneeland_2wc']:.1f}%)"
            )
            print(
                f"  Set-A 2WC paper (recon distractor): "
                f"{k2a['kneeland_2wc_paper']['kneeland_2wc']:.4f} "
                f"({100 * k2a['kneeland_2wc_paper']['kneeland_2wc']:.1f}%)"
            )
            print(
                f"  Set-B 2WC legacy: "
                f"{k2b['kneeland_2wc_gt_distractor']['kneeland_2wc']:.4f} "
                f"({100 * k2b['kneeland_2wc_gt_distractor']['kneeland_2wc']:.1f}%)"
            )
            print(
                f"  Set-B 2WC paper: "
                f"{k2b['kneeland_2wc_paper']['kneeland_2wc']:.4f} "
                f"({100 * k2b['kneeland_2wc_paper']['kneeland_2wc']:.1f}%)"
            )
            print(
                f"  *** A+B avg legacy: {k2_avg_gt:.4f} ({100 * k2_avg_gt:.1f}%)  "
                f"paper: {k2_avg_paper:.4f} ({100 * k2_avg_paper:.1f}%) ***"
            )

            recon_clip = torch.tensor(encode_images_clip(s2_a + s2_b, device))
            gt_clip = torch.tensor(encode_images_clip(_gt_pngs(row_a) + _gt_pngs(row_b), device))
            cos_gt = F.cosine_similarity(recon_clip, gt_clip, dim=-1)
            clip_ab = {
                "clip_cosim_mean": float(cos_gt.mean()),
                "clip_top1": float(
                    ((recon_clip @ gt_clip.T).argmax(dim=1) == torch.arange(len(cos_gt))).sum()
                    / len(cos_gt)
                ),
                "per_cue": {
                    str(rows[row_a[i]].get("cue", i)): float(cos_gt[i])
                    for i in range(len(row_a))
                } | {
                    str(rows[row_b[j]].get("cue", j)): float(cos_gt[len(row_a) + j])
                    for j in range(len(row_b))
                },
            }
            results["recon_clip_to_gt_png"] = clip_ab
            print(
                f"  CLIP(recon) vs GT (A+B, n={len(cos_gt)}): mean={clip_ab['clip_cosim_mean']:.4f}  "
                f"top-1={clip_ab['clip_top1']:.2f}"
            )
            print(f"QUALITY_SCORE={k2_avg_gt:.6f},{clip_ab['clip_cosim_mean']:.6f}")
            print(f"KNEELAND_2WC_GT_DIST_AVG_AB={k2_avg_gt:.6f}")
            print(f"KNEELAND_2WC_PAPER_AVG_AB={k2_avg_paper:.6f}")
        result_json = root / "results" / f"vdvae_recon_{subj}_{sub}.json"
        result_json.parent.mkdir(parents=True, exist_ok=True)
        result_json.write_text(json.dumps(results, indent=2))
        print(f"  Metrics saved {result_json}")
        return results

    # ── Imagery Set B (W/K/B/C/D) ──────────────────────────────────────────────
    row_idx = select_set_b_eval_indices(meta)
    betas_i_raw = betas_avg[row_idx]
    gt_i, labels_i = [], []
    for i in row_idx:
        row = meta["averaged"][i]
        cue = row.get("cue", "?")
        labels_i.append(f"{cue}\n{row.get('label', '')[:20]}")
        gt_i.append(load_imagery_gt_image(
            root, row.get("label", ""), cue=str(row.get("cue", "")), size=256,
        ))

    adapter_ckpt = None
    if beta_adapter_run_id:
        from train_imagery_beta_adapter import apply_imagery_beta_adapter, resolve_beta_adapter_path

        adapter_path = resolve_beta_adapter_path(
            root, beta_adapter_run_id, subj, ckpt_subj=adapter_train,
        )
        adapter_ckpt = torch.load(adapter_path, map_location="cpu", weights_only=False)
        betas_i = apply_imagery_beta_adapter(
            betas_i_raw, root, beta_adapter_run_id, subj,
            adapter_train_subj=adapter_train, device=device,
        )
        print(
            f"\n  Beta adapter: {adapter_path} "
            f"(train={adapter_train} infer={subj} align={adapter_ckpt.get('align_space', 'beta')})"
        )
        results["beta_adapter_run_id"] = beta_adapter_run_id
        results["beta_adapter_ckpt"] = str(adapter_path)
        tag = "adapted"
    else:
        betas_i = betas_i_raw
        tag = "raw"

    print(f"\n  Imagery Set-B eval cues ({len(row_idx)} stimuli, betas={tag}, stage2={stage2_source})...")
    s1_i, s2_i = reconstruct_stimuli(
        betas_batch=betas_i, gt_imgs=gt_i, labels=labels_i,
        ridge=ridge, ema_vae=ema_vae, ref_stats=ref_stats, layer_dims=layer_dims,
        dual_model=dual_model, variant=variant, vd_pipe=vd_pipe, device=device,
        vd_strength=vd_strength, vd_steps=vd_steps, guidance_scale=guidance_scale,
        hybrid_layers=hybrid_layers, root=root, subj=subj,
        stage2_source=stage2_source, prior_bundle=prior_bundle,
        prior_temperature=prior_temperature,
        dual_vm=dual_vm, dual_vs=dual_vs,
        joint_ridge_pack=joint_ridge_pack,
        text_source=text_source,
        vd_vision_mix=vd_vision_mix,
        coco_bank=coco_holder[0],
        prior_seed=prior_seed,
        vd_seed=vd_seed,
        text_to_image_strength=text_to_image_strength,
        brain_token_weight=brain_token_weight,
        prior_blend_weight=prior_blend_weight,
        stage1_source=s1_imagery,
    )
    suffix = f"_{tag}" if beta_adapter_run_id else ""
    if text_source == "mlp_text_hf_dual":
        suffix = f"{suffix}_texthf" if suffix else "_texthf"
    elif text_source == "mlp_text_only":
        suffix = f"{suffix}_textonly" if suffix else "_textonly"
    if text_source == "caption_retrieval":
        suffix = f"{suffix}_captret" if suffix else "_captret"
    if stage2_source == "prior":
        suffix = f"{suffix}_prior" if suffix else "_prior"
    elif use_joint_ridge:
        suffix = f"{suffix}_jr" if suffix else "_jr"
    grid_i = out_dir / ("imagery_setB_grid.png" if stage2_source == "prior" else f"vdvae_imagery_setB5{suffix}_grid.png")
    temp_note = (
        f", prior_τ={prior_temperature}" if stage2_source == "prior" and prior_temperature != 1.0 else ""
    )
    text_note = ""
    if text_source == "caption_retrieval":
        text_note = ", HF dual + COCO caption retrieval"
    elif text_source == "mlp_text_hf_dual":
        ts = text_to_image_strength or DEFAULT_TEXT_TO_IMAGE_STRENGTH["mlp_text_hf_dual"]
        text_note = f", HF dual + MLP text (txt={ts})"
    s2_label = (
        f"Stage 2 (VD strength={vd_strength}, source={stage2_source}{temp_note}{text_note})"
    )
    save_grid(
        gt_i, s1_i, s2_i, labels_i, grid_i,
        title=f"VDVAE+VD imagery Set-B ({tag}, {subj}, stage2={stage2_source})",
        stage2_label=s2_label,
    )
    results["imagery_grid"] = str(grid_i)
    cues = [meta["averaged"][i].get("cue") for i in row_idx]
    stage2_dir = out_dir / "stage2_setB"
    stage2_dir.mkdir(parents=True, exist_ok=True)
    for cue, s2 in zip(cues, s2_i):
        s2.save(stage2_dir / f"{cue}_stage2.png")
    results["imagery_stage2_dir"] = str(stage2_dir)
    s1_i_path = out_dir / f"vdvae_imagery_setB5{suffix}_stage1.png"
    save_stage1_grid(
        gt_i, s1_i, labels_i, s1_i_path,
        title=f"Stage 1 only — imagery Set-B ({tag}, {subj}, hybrid ridge)",
    )
    results["imagery_stage1_grid"] = str(s1_i_path)
    results["imagery_betas_tag"] = tag
    results["imagery_cues"] = cues

    if beta_adapter_run_id:
        print("\n  Imagery Set-B raw betas (no adapter) — Stage 1 baseline...")
        pred_lat_raw = predict_vdvae_latents(
            betas_i_raw, ridge, hybrid=hybrid_layers, root=root, subj=subj,
        )
        s1_raw = decode_flat_latents(
            ema_vae, pred_lat_raw, ref_stats, layer_dims=layer_dims,
            out_size=512, batch_size=4, device=device,
        )
        s1_raw_path = out_dir / "vdvae_imagery_setB5_raw_stage1.png"
        save_stage1_grid(
            gt_i, s1_raw, labels_i, s1_raw_path,
            title=f"Stage 1 only — imagery Set-B (raw OOD, {subj})",
        )
        results["imagery_raw_stage1_grid"] = str(s1_raw_path)

    if eval_recon_kneeland:
        from eval_imagery_crossdecode import _eval_row_indices
        from eval_recon_kneeland import encode_images_clip, eval_recon_kneeland as _eval_recon_k2

        from eval_recon_kneeland import load_kneeland_stage2_recons

        gallery_idx_full = _eval_row_indices(meta["averaged"], "kneeland")
        gallery_recons, gallery_idx = load_kneeland_stage2_recons(
            out_dir, meta["averaged"], gallery_idx_full,
        )

        k2 = _eval_recon_k2(
            s2_i, row_idx, subj, root, subset="kneeland", n_pairs=kneeland_n_pairs, device=device,
            gallery_recon_images=gallery_recons,
            gallery_row_indices=gallery_idx,
        )
        results["recon_kneeland_2wc"] = k2.get("kneeland_2wc_gt_distractor", k2)
        results["recon_kneeland_2wc_paper"] = k2.get("kneeland_2wc_paper", {})
        pct_gt = 100.0 * k2["kneeland_2wc_gt_distractor"]["kneeland_2wc"]
        pct_paper = 100.0 * k2["kneeland_2wc_paper"]["kneeland_2wc"]
        print(f"\n  Set-B 2WC legacy (GT distractor): {k2['kneeland_2wc_gt_distractor']['kneeland_2wc']:.4f} ({pct_gt:.1f}%)")
        print(f"  Set-B 2WC paper (recon distractor): {k2['kneeland_2wc_paper']['kneeland_2wc']:.4f} ({pct_paper:.1f}%)")

        # Direct match: CLIP(recon PNG) vs CLIP(GT stimulus PNG) — tracks visual/semantic fidelity
        recon_clip = torch.tensor(encode_images_clip(s2_i, device))
        gt_clip = torch.tensor(encode_images_clip(gt_i, device))
        cos_gt = F.cosine_similarity(recon_clip, gt_clip, dim=-1)
        top1 = int(
            ((recon_clip @ gt_clip.T).argmax(dim=1) == torch.arange(len(s2_i))).sum()
        )
        clip_gt = {
            "clip_cosim_mean": float(cos_gt.mean()),
            "clip_top1": float(top1 / len(s2_i)),
            "per_cue": {str(c): float(v) for c, v in zip(cues, cos_gt.tolist())},
        }
        results["recon_clip_to_gt_png"] = clip_gt
        print(f"  CLIP(recon) vs CLIP(GT PNG): mean={clip_gt['clip_cosim_mean']:.4f}  "
              f"top-1={clip_gt['clip_top1']:.2f} ({top1}/{len(s2_i)})")
        for c, v in clip_gt["per_cue"].items():
            print(f"    {c}: {v:.4f}")
        print(f"QUALITY_SCORE={k2['kneeland_2wc_gt_distractor']['kneeland_2wc']:.6f},{clip_gt['clip_cosim_mean']:.6f}")
        print(f"KNEELAND_2WC_PAPER_SET_B={k2['kneeland_2wc_paper']['kneeland_2wc']:.6f}")
        print(f"KNEELAND_2WC_GT_DIST_SET_B={k2['kneeland_2wc_gt_distractor']['kneeland_2wc']:.6f}")

    result_json = root / "results" / f"vdvae_recon_{subj}_{sub}.json"
    result_json.parent.mkdir(parents=True, exist_ok=True)
    result_json.write_text(json.dumps(results, indent=2))
    print(f"  Metrics saved {result_json}")
    return results


def run_kneeland_multisample_sweep(
    subj: str,
    root: Path,
    *,
    seeds: list[int],
    run_id: str,
    beta_adapter_run_id: str | None = None,
    prior_run_id: str | None = None,
    kneeland_a_text_source: str = "mlp_text_hf_dual",
    text_to_image_strength: float = 0.05,
    prior_temperature: float = 1.5,
    kneeland_b_vd_strength: float = 0.3,
    out_subdir_base: str = "kn_atxt_ms",
    n_pairs: int = 1000,
    eval_seed: int = 42,
) -> dict:
    """Kneeland A=text / B=prior: one full recon per VD/prior seed, then paper multisample 2WC."""
    root = get_root(root)
    seed_dirs: list[Path] = []
    for s in seeds:
        sub = f"{out_subdir_base}_s{s}"
        print(f"\n{'=' * 60}\n  Multisample recon seed={s} → {sub}\n{'=' * 60}")
        run_reconstruction(
            subj,
            root,
            run_id=run_id,
            variant="4H_CTR2",
            ckpt_prefer="best_retrieval",
            beta_adapter_run_id=beta_adapter_run_id,
            stage2_source="prior",
            prior_run_id=prior_run_id,
            prior_temperature=prior_temperature,
            out_subdir=sub,
            eval_recon_kneeland=False,
            text_source="",
            text_to_image_strength=text_to_image_strength,
            prior_seed=s,
            vd_seed=s,
            imagery_sets="kneeland",
            kneeland_a_text_source=kneeland_a_text_source,
            kneeland_b_text_source="",
            kneeland_b_vd_strength=kneeland_b_vd_strength,
            stage1_source="vdvae_ridge",
        )
        seed_dirs.append(root / "reconstructions" / subj / sub)

    from eval_recon_kneeland import eval_kneeland_multisample_from_seed_dirs

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ms = eval_kneeland_multisample_from_seed_dirs(
        seed_dirs, subj, root, n_pairs=n_pairs, seed=eval_seed, device=device,
    )
    out_json = root / "results" / f"kneeland_multisample_{subj}_{out_subdir_base}.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {"subj": subj, "seeds": seeds, "config": out_subdir_base, **ms}
    out_json.write_text(json.dumps(payload, indent=2))
    print(f"\n  Saved {out_json}")
    print(f"KNEELAND_2WC_PAPER_MULTISAMPLE_AVG_AB={ms['kneeland_2wc_paper_multisample_avg_ab']:.6f}")
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description="VDVAE Stage 1 + dual projector VD Stage 2.")
    ap.add_argument("--subj", type=str, default="subj01")
    ap.add_argument("--root", type=str, default=None)
    ap.add_argument("--run-id", type=str, default=DEFAULT_DUAL_RUN)
    ap.add_argument("--variant", type=str, default="4H_CTR2")
    ap.add_argument("--ckpt-prefer", type=str, default="best_retrieval")
    ap.add_argument("--n-perception", type=int, default=3)
    ap.add_argument("--vd-strength", type=float, default=0.75)
    ap.add_argument("--vd-steps", type=int, default=50)
    ap.add_argument("--guidance-scale", type=float, default=7.5)
    ap.add_argument(
        "--full-layers",
        action="store_true",
        help="Predict all 31 layers (default: hybrid — pred layers 0-5, train mean 6-30)",
    )
    ap.add_argument(
        "--perception-setb-only",
        action="store_true",
        help="Perception betas on same 6 Kneeland Set-B cues as imagery (prior+ridge, no imagery adapter)",
    )
    ap.add_argument(
        "--perception-only",
        action="store_true",
        help="Skip imagery Set-B; run perception val only",
    )
    ap.add_argument(
        "--imagery-only",
        action="store_true",
        help="Skip perception val; run imagery Set-B only",
    )
    ap.add_argument(
        "--beta-adapter-run-id",
        default=None,
        help="Apply beta-space imagery→perception adapter before imagery VDVAE/dual decode",
    )
    ap.add_argument(
        "--stage1-source",
        choices=STAGE1_SOURCES,
        default="vdvae_ridge",
        help="Default Stage 1 for both modalities if split args omitted",
    )
    ap.add_argument(
        "--stage1-source-perception",
        choices=STAGE1_SOURCES,
        default=None,
        help="Perception val Stage 1 (default: --stage1-source)",
    )
    ap.add_argument(
        "--stage1-source-imagery",
        choices=STAGE1_SOURCES,
        default=None,
        help="Imagery Stage 1 (default: --stage1-source)",
    )
    ap.add_argument(
        "--stage2-source",
        choices=STAGE2_SOURCES,
        default="projector",
        help="Stage 2 CLIP: projector | prior (DDIM) | regression | joint_ridge",
    )
    ap.add_argument("--prior-run-id", default=None, help="Run id for checkpoints_prior (optional)")
    ap.add_argument(
        "--prior-temperature",
        type=float,
        default=1.0,
        help="DDIM sampling temperature (>1 spreads prior CLIP, reduces mean collapse)",
    )
    ap.add_argument(
        "--prior-blend-weight",
        type=float,
        default=0.85,
        help="For prior_blend: weight on DDIM prior vs (1-w) on projector→768",
    )
    ap.add_argument("--out-subdir", default=None)
    ap.add_argument(
        "--text-source",
        default="",
        choices=TEXT_SOURCES,
        help=(
            "mlp_text_hf_dual: HF dual + brain text | mlp_text_only: HF dual text-only | "
            "caption_retrieval: HF dual + COCO top-1 (HF only, no Brain Diffuser)"
        ),
    )
    ap.add_argument(
        "--text-to-image-strength",
        type=float,
        default=None,
        help="HF dual text branch weight (default 0.15 texthf, 0.4 captret)",
    )
    ap.add_argument(
        "--brain-token-weight",
        type=float,
        default=0.85,
        help="Blend weight for brain text into HF 77-token layout (mlp_text_hf_dual)",
    )
    ap.add_argument(
        "--vd-vision-mix",
        type=float,
        default=VD_DEFAULT_VISION_MIX,
        help="Vision share in decode_dc mixed_ratio (default 0.6 vision / 0.4 text)",
    )
    ap.add_argument("--eval-recon-kneeland", action="store_true")
    ap.add_argument("--kneeland-n-pairs", type=int, default=1000)
    ap.add_argument("--prior-seed", type=int, default=None, help="DDIM prior RNG seed (reproducibility)")
    ap.add_argument("--vd-seed", type=int, default=None, help="VD img2img RNG seed (+stim index per image)")
    ap.add_argument(
        "--imagery-sets",
        default="set_b",
        choices=IMAGERY_SET_CHOICES,
        help="set_b (W/K/B/C/D) | set_a | kneeland (A=proj+low VD, B=prior; no Set C)",
    )
    ap.add_argument(
        "--kneeland-b-vd-strength",
        type=float,
        default=None,
        help="Override Set-B VD strength when --imagery-sets kneeland (A stays 0.12)",
    )
    ap.add_argument(
        "--kneeland-a-text-source",
        default="",
        choices=("",) + TEXT_SOURCES[1:],
        help="Set A only: mlp_text_hf_dual etc. (B unchanged unless --kneeland-b-text-source)",
    )
    ap.add_argument(
        "--kneeland-b-text-source",
        default="",
        choices=("",) + TEXT_SOURCES[1:],
        help="Set B only text mode (overrides legacy --text-source on B)",
    )
    ap.add_argument(
        "--multisample-seeds",
        default="",
        help="Comma-separated prior/vd seeds for kneeland multisample sweep (with --imagery-sets kneeland)",
    )
    args = ap.parse_args()

    if args.multisample_seeds and args.imagery_sets == "kneeland":
        seeds = [int(x) for x in args.multisample_seeds.split(",")]
        run_kneeland_multisample_sweep(
            args.subj,
            get_root(args.root),
            seeds=seeds,
            run_id=args.run_id or DEFAULT_DUAL_RUN,
            beta_adapter_run_id=args.beta_adapter_run_id,
            prior_run_id=args.prior_run_id,
            kneeland_a_text_source=args.kneeland_a_text_source or "mlp_text_hf_dual",
            text_to_image_strength=float(
                args.text_to_image_strength
                if args.text_to_image_strength is not None
                else 0.05
            ),
            prior_temperature=args.prior_temperature,
            kneeland_b_vd_strength=float(
                args.kneeland_b_vd_strength if args.kneeland_b_vd_strength is not None else 0.3
            ),
            out_subdir_base=args.out_subdir or "kn_atxt_ms",
            n_pairs=args.kneeland_n_pairs,
        )
        return

    if args.stage2_source == "joint_ridge" or args.variant.lower() in ("joint_ridge", "jointridge"):
        variant = "joint_ridge"
        run_id = args.run_id or DEFAULT_JOINT_RIDGE_RUN
        stage2 = "joint_ridge"
    else:
        variant = normalize_variant(args.variant)
        run_id = args.run_id
        stage2 = args.stage2_source
    results = run_reconstruction(
        args.subj,
        get_root(args.root),
        run_id=run_id,
        variant=variant,
        ckpt_prefer=args.ckpt_prefer,
        n_perception=args.n_perception,
        vd_strength=args.vd_strength,
        vd_steps=args.vd_steps,
        guidance_scale=args.guidance_scale,
        hybrid_layers=not args.full_layers,
        perception_only=args.perception_only,
        imagery_only=args.imagery_only,
        beta_adapter_run_id=args.beta_adapter_run_id,
        stage2_source=stage2,
        prior_run_id=args.prior_run_id,
        prior_temperature=args.prior_temperature,
        out_subdir=args.out_subdir,
        eval_recon_kneeland=args.eval_recon_kneeland,
        kneeland_n_pairs=args.kneeland_n_pairs,
        text_source=args.text_source,
        vd_vision_mix=args.vd_vision_mix,
        text_to_image_strength=args.text_to_image_strength,
        brain_token_weight=args.brain_token_weight,
        prior_seed=args.prior_seed,
        vd_seed=args.vd_seed,
        imagery_sets=args.imagery_sets,
        prior_blend_weight=args.prior_blend_weight,
        kneeland_b_vd_strength=args.kneeland_b_vd_strength,
        kneeland_a_text_source=args.kneeland_a_text_source,
        kneeland_b_text_source=args.kneeland_b_text_source,
        stage1_source=args.stage1_source,
        stage1_source_perception=args.stage1_source_perception,
        stage1_source_imagery=args.stage1_source_imagery,
        perception_setb_only=args.perception_setb_only,
    )
    if args.eval_recon_kneeland:
        k2a = results.get("recon_kneeland_2wc_set_a", {})
        k2b = results.get("recon_kneeland_2wc_set_b", results.get("recon_kneeland_2wc", {}))
        k2avg = results.get("recon_kneeland_2wc_avg_ab", {})
        if k2a and "kneeland_2wc_paper" in k2a:
            print(f"KNEELAND_2WC_SET_A_PAPER={k2a['kneeland_2wc_paper']['kneeland_2wc']:.6f}")
            print(f"KNEELAND_2WC_SET_A_GT={k2a['kneeland_2wc_gt_distractor']['kneeland_2wc']:.6f}")
        if k2b and "kneeland_2wc_paper" in k2b:
            print(f"KNEELAND_2WC_SET_B_PAPER={k2b['kneeland_2wc_paper']['kneeland_2wc']:.6f}")
            print(f"KNEELAND_2WC_SET_B_GT={k2b['kneeland_2wc_gt_distractor']['kneeland_2wc']:.6f}")
        if k2avg:
            print(f"KNEELAND_2WC_AVG_AB={k2avg['kneeland_2wc']:.6f}")
            print(f"KNEELAND_2WC_GT_DIST_AVG_AB={k2avg['kneeland_2wc']:.6f}")
        k2paper = results.get("recon_kneeland_2wc_avg_ab_paper", {})
        if k2paper:
            print(f"KNEELAND_2WC_PAPER_AVG_AB={k2paper['kneeland_2wc']:.6f}")
            print(f"KNEELAND_2WC_RESULT={k2paper['kneeland_2wc']:.6f}")
        hp = results.get("beta_adapter_honest_protocol", "")
        if hp and k2b and "kneeland_2wc_paper" in k2b:
            print(
                f"HONEST_KNEELAND_SET_B_PAPER={k2b['kneeland_2wc_paper']['kneeland_2wc']:.6f} "
                f"(primary metric; adapter not fit on Set B)"
            )
            if k2a and "kneeland_2wc_paper" in k2a:
                print(
                    f"HONEST_KNEELAND_SET_A_PAPER={k2a['kneeland_2wc_paper']['kneeland_2wc']:.6f} "
                    f"(Set A may be in adapter training for train_a_holdout_b)"
                )
        elif results.get("recon_kneeland_2wc"):
            print(f"KNEELAND_2WC_RESULT={results['recon_kneeland_2wc']['kneeland_2wc']:.6f}")
        cg = results.get("recon_clip_to_gt_png", {})
        if cg:
            print(f"CLIP_GT_COSIM={cg.get('clip_cosim_mean', 0):.6f}")
    return results


if __name__ == "__main__":
    main()
