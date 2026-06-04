# -*- coding: utf-8 -*-
"""
Reconstruct 3 Set-B (naturalistic NSD) imagery stimuli from averaged betas.

Stage 1: predicted VAE latents -> SD VAE decoder
Stage 2: Stage-1 image + predicted CLIP -> Versatile Diffusion img2img (strength=0.6)

Usage:
    python reconstruct_imagery_small.py --subj subj01 --root /mnt/mindbridge
"""

import argparse
import json
import os
import sys

sys.path = [p for p in sys.path if "share/software" not in p and "jupyterlab" not in p]

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL
from pathlib import Path
from PIL import Image
from torchvision import transforms
from paths import (
    ensure_runtime_dirs,
    get_root,
    reconstruction_run_dir,
    resolve_run_id,
    resolve_variant_checkpoint,
    update_run_manifest,
)
from vd_brain_diffuser import (
    BDVDBundle,
    VD_DEFAULT_TEXT_MIXING,
    load_bd_vd_bundle,
    stage2_bd_vd_img2img,
)


from reconstruct_common import (
    CLIP_SOURCE_CHOICES,
    DEFAULT_RUN_IDS,
    VARIANT_CHOICES,
    VARIANT_LABELS,
    build_recon_model,
    forward_recon,
    normalize_variant,
)


# =============================================================================
# Helpers
# =============================================================================
def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    t = t.clamp(-1, 1)
    t = (t + 1) / 2
    t = (t * 255).byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(t)


def print_metadata_summary(meta: dict) -> None:
    print("\n=== imagery_trial_meta structure ===")
    print(f"  subj:        {meta.get('subj')}")
    print(f"  n_total:     {meta.get('n_total')}  (all GLMsingle betas)")
    print(f"  n_imagery:   {meta.get('n_imagery')}  (imagery-task trials)")
    print(f"  n_averaged:  {meta.get('n_averaged')}  (unique stimulus conditions)")

    print("\n=== averaged[] — one row per unique (set, stimulus_id) ===")
    for i, row in enumerate(meta.get("averaged", [])):
        print(
            f"  [{i:2d}] set={row['set']}  stimulus_id={row['stimulus_id']}  "
            f"cue={row['cue']!r}  nsd_id={row.get('nsd_id')}  "
            f"label={row.get('label')!r}  n_reps={row.get('n_reps')}"
        )


def select_set_b_indices(meta: dict, n: int = 3, all_valid: bool = False) -> list[int]:
    """Return row indices into betas_avg for Set-B naturalistic stimuli with NSD images."""
    return select_imagery_set_indices(meta, "B", n=n, all_valid=all_valid, require_nsd=True)


def select_imagery_set_indices(
    meta: dict,
    set_id: str,
    n: int = 0,
    *,
    all_valid: bool = False,
    require_nsd: bool = False,
) -> list[int]:
    """Row indices into betas_avg for imagery set A, B, or C."""
    set_id = set_id.upper()
    rows = [
        i for i, row in enumerate(meta["averaged"])
        if row.get("set") == set_id
        and (not require_nsd or row.get("nsd_id") not in (None, 0))
    ]
    if not rows:
        raise ValueError(f"No Set-{set_id} stimuli found in imagery metadata")
    if all_valid or n <= 0:
        return rows
    if len(rows) < n:
        raise ValueError(f"Need {n} Set-{set_id} stimuli, found {len(rows)}")
    return rows[:n]


def find_imagery_png(root: Path, label: str) -> Path | None:
    meta_dir = root / "nsd_meta" / "nsdimagery"
    if not meta_dir.exists() or not label:
        return None
    hits = list(meta_dir.rglob(label))
    return hits[0] if hits else None


def load_imagery_png(root: Path, label: str, size: int = 256) -> Image.Image:
    png = find_imagery_png(root, label)
    if png is None:
        raise FileNotFoundError(f"Imagery PNG not found for label={label!r} under nsdimagery/")
    return Image.open(png).convert("RGB").resize((size, size), Image.LANCZOS)


def load_imagery_gt_image(
    root: Path,
    label: str,
    cue: str = "",
    size: int = 256,
) -> Image.Image:
    """Load Set-A PNG if present; otherwise render a text placeholder (Set C concepts)."""
    for candidate in (label, f"{label}.png" if label and "." not in label else ""):
        if candidate and find_imagery_png(root, candidate):
            return load_imagery_png(root, candidate, size=size)
    text = (label or cue or "?").replace("_", " ")
    img = Image.new("RGB", (size, size), color=(240, 240, 240))
    try:
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(img)
        font = ImageFont.load_default()
        draw.multiline_text((size // 2, size // 2), text, fill=(20, 20, 20), anchor="mm", font=font, align="center")
    except Exception:
        pass
    return img


@torch.no_grad()
def encode_images_openclip(
    images: list[Image.Image],
    device: torch.device,
) -> torch.Tensor:
    """ViT-L/14 image embeddings (768-d), L2-normalized — matches training CLIP."""
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="openai",
    )
    model = model.to(device).eval()
    batch = torch.stack([preprocess(im.convert("RGB")) for im in images]).to(device)
    return F.normalize(model.encode_image(batch), dim=-1).cpu()


def eval_recon_clip_2wc(
    recon_imgs: list[Image.Image],
    gt_imgs: list[Image.Image],
    device: torch.device,
    *,
    n_pairs: int = 10000,
    labels: list[str] | None = None,
) -> dict:
    """
    CLIP 2-way classification on reconstructed images vs ground-truth gallery.

    For random pairs (i, j), success when cos(recon_i, gt_i) > cos(recon_i, gt_j).
    Also reports mean cos(recon_i, gt_i) and leave-one-out top-1 when n >= 2.
    """
    from training_checkpoints import retrieval_2way

    recon_emb = encode_images_openclip(recon_imgs, device)
    gt_emb = encode_images_openclip(gt_imgs, device)
    cos_diag = F.cosine_similarity(recon_emb, gt_emb, dim=-1)

    two_wc = retrieval_2way(recon_emb, gt_emb, n_pairs=n_pairs)

    # Leave-one-out top-1: recon_i closest to gt_i among all gt
    n = len(recon_imgs)
    top1 = 0
    if n >= 2:
        sim = recon_emb @ gt_emb.T
        top1 = int((sim.argmax(dim=1) == torch.arange(n)).sum().item())
        top1_acc = top1 / n
    else:
        top1_acc = float("nan")

    per_stim = {}
    if labels:
        for lab, c in zip(labels, cos_diag.tolist()):
            per_stim[lab] = float(c)

    return {
        "n_stimuli": n,
        "clip_2wc": float(two_wc),
        "clip_top1": float(top1_acc),
        "clip_top1_correct": top1,
        "clip_cosim_mean": float(cos_diag.mean()),
        "clip_cosim_per_stim": per_stim,
    }


def print_clip_2wc_metrics(metrics: dict, stage: str) -> None:
    print(f"\n  [{stage}] CLIP ViT-L/14 metrics (chance 2WC=50%, top-1={100/metrics['n_stimuli']:.1f}% for n={metrics['n_stimuli']}):")
    print(f"    2-way classification: {metrics['clip_2wc']:.4f} ({metrics['clip_2wc']*100:.1f}%)")
    if metrics["n_stimuli"] >= 2:
        print(f"    Top-1 retrieval:      {metrics['clip_top1']:.4f} ({metrics['clip_top1']*100:.1f}%)  ({metrics['clip_top1_correct']}/{metrics['n_stimuli']})")
    print(f"    Mean cos(recon, GT):  {metrics['clip_cosim_mean']:.4f}")
    for lab, c in metrics.get("clip_cosim_per_stim", {}).items():
        print(f"      {lab}: {c:.4f}")

def load_nsd_image(root: Path, nsd_id: int, size: int = 256) -> Image.Image:
    import h5py

    stim_path = root / "nsd_meta" / "nsd_stimuli.hdf5"
    if not stim_path.exists():
        raise FileNotFoundError(f"NSD stimuli HDF5 not found: {stim_path}")
    with h5py.File(stim_path, "r") as f:
        img = Image.fromarray(f["imgBrick"][int(nsd_id)])
    return img.resize((size, size), Image.LANCZOS)


VD_DEFAULT_VISION_MIX = 1.0 - VD_DEFAULT_TEXT_MIXING


def load_vd_image_variation_pipe(device: torch.device, root: Path | None = None) -> BDVDBundle:
    """Brain Diffuser native VD (dual image/text UNet, decode_dc)."""
    root = get_root(root)
    return load_bd_vd_bundle(device, root)


def load_vd_dual_guided_pipe(device: torch.device, root: Path | None = None) -> BDVDBundle:
    """Same bundle — dual conditioning uses decode_dc when text CLIP is passed."""
    return load_vd_image_variation_pipe(device, root)


def stage1_vae_decode(vae: AutoencoderKL, pred_vae: torch.Tensor, device: torch.device) -> list[Image.Image]:
    with torch.no_grad():
        scaled = pred_vae.to(device=device, dtype=vae.dtype) / 0.18215
        decoded = vae.decode(scaled).sample
    return [tensor_to_pil(decoded[i]).resize((256, 256), Image.LANCZOS) for i in range(decoded.size(0))]


def _pil_to_vae_tensor(img: Image.Image, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.Resize(512, interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.CenterCrop(512),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    return transform(img.convert("RGB")).unsqueeze(0).to(device=device, dtype=dtype)


@torch.no_grad()
def stage2_vd_img2img(
    bundle: BDVDBundle,
    init_image: Image.Image,
    pred_clip: torch.Tensor,
    *,
    pred_text_clip: torch.Tensor | None = None,
    vd_vision_mix: float = VD_DEFAULT_VISION_MIX,
    strength: float = 0.6,
    num_inference_steps: int = 20,
    guidance_scale: float = 7.5,
    generator: torch.Generator | None = None,
) -> Image.Image:
    """Img2img via Brain Diffuser VD (vision-only or dual text+vision decode_dc)."""
    del generator
    text_mixing = 1.0 - float(vd_vision_mix)
    return stage2_bd_vd_img2img(
        bundle,
        init_image,
        pred_clip,
        pred_text_clip=pred_text_clip,
        text_mixing=text_mixing,
        strength=strength,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
    )


KANDINSKY_PRIOR_REPO = "kandinsky-community/kandinsky-2-2-prior"
KANDINSKY_DECODER_REPO = "kandinsky-community/kandinsky-2-2-decoder"
KANDINSKY_PRIOR_DIM = 1280


def clip768_to_kandinsky_prior1280(clip768: torch.Tensor) -> torch.Tensor:
    """Map brain regression CLIP (ViT-L/14, 768-d) into Kandinsky prior space (1280-d)."""
    x = F.normalize(clip768.float(), dim=-1)
    if x.shape[-1] == KANDINSKY_PRIOR_DIM:
        return x
    if x.shape[-1] != 768:
        raise ValueError(f"Expected 768-d CLIP embedding, got {x.shape[-1]}")
    return F.normalize(F.pad(x, (0, KANDINSKY_PRIOR_DIM - 768)), dim=-1)


def load_kandinsky_pipelines(device: torch.device):
    """Load Kandinsky 2.2 prior (emb2emb) + decoder (img2img) in float16."""
    from diffusers import KandinskyV22Img2ImgPipeline, KandinskyV22PriorEmb2EmbPipeline

    dtype = torch.float16
    print(f"  Loading Kandinsky 2.2 prior from {KANDINSKY_PRIOR_REPO}...")
    prior_pipe = KandinskyV22PriorEmb2EmbPipeline.from_pretrained(
        KANDINSKY_PRIOR_REPO, torch_dtype=dtype,
    ).to(device)
    print(f"  Loading Kandinsky 2.2 decoder from {KANDINSKY_DECODER_REPO}...")
    decoder_pipe = KandinskyV22Img2ImgPipeline.from_pretrained(
        KANDINSKY_DECODER_REPO, torch_dtype=dtype,
    ).to(device)
    return prior_pipe, decoder_pipe


@torch.no_grad()
def stage2_kandinsky_img2img(
    prior_pipe,
    decoder_pipe,
    init_image: Image.Image,
    pred_clip_768: torch.Tensor,
    *,
    prior_strength: float = 0.25,
    prior_steps: int = 25,
    img2img_strength: float = 0.3,
    decoder_steps: int = 50,
    guidance_scale: float = 4.0,
    generator: torch.Generator | None = None,
) -> Image.Image:
    """
    Stage 2 via Kandinsky 2.2: brain 768-d CLIP -> prior emb2emb -> decoder img2img.

    The regression CLIP embedding is padded to 1280-d and passed directly to
    KandinskyV22PriorEmb2EmbPipeline (ndim==2 tensor path), then decoded with
    KandinskyV22Img2ImgPipeline using the Stage-1 VAE image as init.
    """
    device = prior_pipe._execution_device
    dtype = torch.float16

    prior_input = clip768_to_kandinsky_prior1280(pred_clip_768).to(device=device, dtype=dtype)
    if prior_input.ndim == 1:
        prior_input = prior_input.unsqueeze(0)
    prior_out = prior_pipe(
        prompt="",
        image=prior_input,
        strength=prior_strength,
        num_inference_steps=prior_steps,
        guidance_scale=1.0,
        generator=generator,
    )
    init_768 = init_image.resize((768, 768), Image.LANCZOS)
    out = decoder_pipe(
        image=init_768,
        image_embeds=prior_out.image_embeds,
        negative_image_embeds=prior_out.negative_image_embeds,
        strength=img2img_strength,
        num_inference_steps=decoder_steps,
        guidance_scale=guidance_scale,
        height=768,
        width=768,
        generator=generator,
    )
    return out.images[0].resize((256, 256), Image.LANCZOS)


def stimuli_grid_name(
    kind: str = "imagery",
    ckpt_prefer: str = "final",
    variant: str = "A",
    n_stim: int = 3,
) -> str:
    """Output PNG name; non-default variant/checkpoint get suffixes."""
    suffix = ""
    if ckpt_prefer != "final":
        suffix += f"_{ckpt_prefer}"
    v = normalize_variant(variant)
    if v != "A":
        suffix += f"_{v.lower()}"
    return f"{kind}_setb_{n_stim}stimuli{suffix}.png"


def save_grid(
    gt_imgs: list[Image.Image],
    stage1_imgs: list[Image.Image],
    stage2_imgs: list[Image.Image],
    labels: list[str],
    save_path: Path,
    *,
    title: str = "NSD-Imagery Set-B reconstruction (Variant A)",
    stage2_label: str = "Stage 2 (VD img2img)",
) -> None:
    n = len(gt_imgs)
    fig, axes = plt.subplots(3, n, figsize=(n * 2.8, 9))
    row_titles = ["Ground truth", "Stage 1 (VAE decode)", stage2_label]

    for col in range(n):
        axes[0, col].imshow(gt_imgs[col])
        axes[0, col].set_title(labels[col], fontsize=8)
        axes[1, col].imshow(stage1_imgs[col])
        axes[2, col].imshow(stage2_imgs[col])
        for row in range(3):
            axes[row, col].axis("off")

    for row, title in enumerate(row_titles):
        axes[row, 0].set_ylabel(title, fontsize=9, rotation=90, labelpad=10)

    plt.suptitle(title, fontsize=12, y=1.01)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved figure: {save_path}")


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subj", default=os.environ.get("NSD_SUBJ", "subj01"))
    parser.add_argument("--root", default=os.environ.get("MINDBRIDGE_ROOT", "/mnt/mindbridge"))
    parser.add_argument("--n", type=int, default=3, help="Number of Set-B stimuli (ignored if --all-set-b)")
    parser.add_argument("--all-set-b", action="store_true",
                        help="Reconstruct all Set-B stimuli with valid NSD images")
    parser.add_argument("--eval-clip-2wc", action="store_true",
                        help="Compute CLIP 2-way classification on recon vs GT images")
    parser.add_argument("--strength", type=float, default=0.6, help="VD img2img noise strength")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ckpt", default=None, help="Override checkpoint path (default: final.pt in run dir)")
    parser.add_argument("--run-id", default=None, help="Run id under runs/{run_id}/ (default: new unique id)")
    parser.add_argument("--variant", default="A", choices=VARIANT_CHOICES,
                        help="Training variant checkpoint layout (A, 4H, 4H_CTR)")
    parser.add_argument("--clip-source", default="regression", choices=CLIP_SOURCE_CHOICES,
                        help="768-d CLIP for VD: regression head or 4H_CTR projector→768")
    parser.add_argument("--ckpt-prefer", default="final",
                        choices=("final", "best_clip", "best_retrieval"),
                        help="Which checkpoint to load when --ckpt is not set")
    args = parser.parse_args()

    root = get_root(args.root)
    subj = args.subj
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    variant = normalize_variant(args.variant)

    if args.run_id:
        run_id = args.run_id
    elif variant in DEFAULT_RUN_IDS:
        run_id = DEFAULT_RUN_IDS[variant]
    else:
        run_id = resolve_run_id(f"recon_imagery_{subj}")
    os.environ["MINDBRIDGE_RUN_ID"] = run_id

    label = VARIANT_LABELS.get(variant, variant)
    print(f"Variant:  {variant} ({label})")
    print(f"CLIP src: {args.clip_source}")

    print(f"Subject:  {subj}")
    print(f"Device:   {device}")
    print(f"Root:     {root}")
    print(f"Run ID:   {run_id}")

    ensure_runtime_dirs(root, subj)
    os.environ["TRANSFORMERS_CACHE"] = str(root / "hf_cache")
    os.environ["HF_HOME"] = str(root / "hf_cache")
    os.environ["TORCH_HOME"] = str(root / "torch_cache")

    meta_path = root / "nsd_meta" / f"imagery_trial_meta_{subj}.json"
    avg_path = root / "nsd_meta" / f"betas_avg_imagery_{subj}.npy"
    ckpt_path = (
        Path(args.ckpt)
        if args.ckpt
        else resolve_variant_checkpoint(root, variant, subj, run_id, prefer=args.ckpt_prefer)
    )
    out_dir = reconstruction_run_dir(root, subj, run_id)

    assert meta_path.exists(), f"Missing metadata: {meta_path}"
    assert avg_path.exists(), f"Missing averaged betas: {avg_path}"

    with open(meta_path) as f:
        meta = json.load(f)
    betas_avg = np.load(avg_path)

    print_metadata_summary(meta)

    set_b_rows = select_set_b_indices(meta, n=args.n, all_valid=args.all_set_b)
    n_stim = len(set_b_rows)
    out_path = out_dir / stimuli_grid_name("imagery", args.ckpt_prefer, variant, n_stim)

    assert ckpt_path.exists(), f"Missing {variant} checkpoint: {ckpt_path}"

    print(f"\n=== Selected {n_stim} Set-B stimuli (indices into averaged[]) ===")
    selected_meta = []
    for idx in set_b_rows:
        row = meta["averaged"][idx]
        selected_meta.append(row)
        print(
            f"  avg_idx={idx}  cue={row['cue']!r}  nsd_id={row['nsd_id']}  "
            f"label={row['label']!r}  n_reps={row['n_reps']}"
        )

    sample_betas = betas_avg[set_b_rows]
    print(f"\nAveraged betas slice shape: {sample_betas.shape}")

    print(f"\nLoading {label} checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    n_voxels = ckpt["n_voxels"]
    voxel_mean = ckpt["voxel_mean"]
    voxel_std = ckpt["voxel_std"]
    ckpt_epoch = ckpt.get("epoch", "?")
    print(f"  epoch={ckpt_epoch}  n_voxels={n_voxels}  file={ckpt_path.name}")
    if ckpt.get("contrastive_target"):
        print(f"  contrastive_target={ckpt['contrastive_target']}")

    model = build_recon_model(variant, n_voxels, device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    normed = (sample_betas - voxel_mean) / voxel_std
    betas_tensor = torch.tensor(normed, dtype=torch.float32, device=device)

    print("\nRunning model inference on averaged Set-B betas...")
    pred_clip, pred_vae = forward_recon(
        model, variant, betas_tensor, clip_source=args.clip_source, device=device,
    )
    print(f"  pred_clip: {pred_clip.shape}  pred_vae: {pred_vae.shape}")

    print("\n[Stage 1] Loading SD VAE decoder (float16)...")
    vae = AutoencoderKL.from_pretrained(
        "CompVis/stable-diffusion-v1-4", subfolder="vae", torch_dtype=torch.float16,
    ).to(device).eval()
    stage1_imgs = stage1_vae_decode(vae, pred_vae, device)
    del vae
    torch.cuda.empty_cache()

    print(f"\n[Stage 2] Loading Versatile Diffusion img2img pipeline (float16, strength={args.strength})...")
    vd_pipe = load_vd_image_variation_pipe(device, root)
    vd_pipe.set_progress_bar_config(disable=True)

    gen = torch.Generator(device=device).manual_seed(args.seed)
    stage2_imgs = []
    gt_imgs = []
    col_labels = []

    for i, row in enumerate(selected_meta):
        nsd_id = row["nsd_id"]
        print(f"\n  Stimulus {i + 1}/{args.n}: cue={row['cue']!r}  nsd_id={nsd_id}  label={row['label']!r}")
        gt = load_nsd_image(root, nsd_id)
        gt_imgs.append(gt)
        col_labels.append(f"{row['cue']}\nnsd{nsd_id}")

        s2 = stage2_vd_img2img(
            vd_pipe,
            stage1_imgs[i],
            pred_clip[i],
            strength=args.strength,
            num_inference_steps=20,
            generator=gen,
        )
        stage2_imgs.append(s2)
        print(f"    Stage 1 + Stage 2 reconstruction done.")

    del vd_pipe
    torch.cuda.empty_cache()

    save_grid(
        gt_imgs, stage1_imgs, stage2_imgs, col_labels, out_path,
        title=f"NSD-Imagery Set-B ({label}, {ckpt_path.name}, clip={args.clip_source})",
    )

    clip_metrics = {}
    if args.eval_clip_2wc:
        print("\n=== CLIP 2-way classification on reconstructions ===")
        clip_metrics["stage1"] = eval_recon_clip_2wc(
            stage1_imgs, gt_imgs, device, labels=col_labels,
        )
        clip_metrics["stage2"] = eval_recon_clip_2wc(
            stage2_imgs, gt_imgs, device, labels=col_labels,
        )
        print_clip_2wc_metrics(clip_metrics["stage1"], "Stage 1 (VAE)")
        print_clip_2wc_metrics(clip_metrics["stage2"], "Stage 2 (VD)")

        metrics_path = out_dir / f"imagery_setb_clip2wc_{n_stim}stim_{args.ckpt_prefer}.json"
        metrics_path.write_text(json.dumps({
            "subj": subj,
            "variant": variant,
            "run_id": run_id,
            "checkpoint": str(ckpt_path),
            "ckpt_prefer": args.ckpt_prefer,
            "clip_source": args.clip_source,
            "n_stimuli": n_stim,
            "nsd_ids": [row["nsd_id"] for row in selected_meta],
            **clip_metrics,
        }, indent=2))
        print(f"\n  Saved CLIP metrics: {metrics_path}")

    from paths import run_root
    update_run_manifest(
        run_root(root, run_id),
        run_id=run_id,
        reconstructions={
            subj: {
                out_path.name: str(out_path),
                "checkpoint": str(ckpt_path),
                "ckpt_prefer": args.ckpt_prefer,
                "variant": variant,
                "clip_source": args.clip_source,
                "clip_2wc": clip_metrics if clip_metrics else None,
            }
        },
    )
    print("Done.")


if __name__ == "__main__":
    main()
