# -*- coding: utf-8 -*-
"""HuggingFace Versatile Diffusion Stage 2 (image-variation + dual-guided img2img)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

VD_REPO = "shi-labs/versatile-diffusion"
CLIP_REPO = "openai/clip-vit-large-patch14"


def _bin_kw() -> dict:
    return {"use_safetensors": False}


def _pil_to_vae_tensor(img: Image.Image, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.Resize(512, interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.CenterCrop(512),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    return transform(img.convert("RGB")).unsqueeze(0).to(device=device, dtype=dtype)


def _normalize_vd_image_tokens(raw: torch.Tensor, pipe) -> torch.Tensor:
    """Match VersatileDiffusionDualGuidedPipeline._encode_image_prompt normalization."""
    embeds = pipe.image_encoder.vision_model.post_layernorm(raw.last_hidden_state)
    embeds = pipe.image_encoder.visual_projection(embeds)
    embeds_pooled = embeds[:, 0:1]
    return embeds / torch.norm(embeds_pooled, dim=-1, keepdim=True)


@dataclass
class COCORetrievalBank:
    """Global COCO caption CLIP embeddings + parallel caption strings."""
    embeddings: torch.Tensor  # (N, 768) float32 on CPU
    captions: list[str]


def _load_coco_id_73k(root: Path) -> np.ndarray:
    from ingest_averaged import _load_coco_id_73k
    from vdvae_trial_data import load_expdesign

    meta_dir = root / "nsd_meta"
    return _load_coco_id_73k(meta_dir, load_expdesign(root))


def load_coco_retrieval_bank(root: Path) -> COCORetrievalBank:
    """
    Load targets_text_clip.npy (~70566×768) and build caption strings per row.

    Captions come from coco_captions.json via 73k image id → COCO id mapping.
    """
    from paths import averaged_meta_dir
    from ingest_averaged import ensure_coco_captions_json

    avg_dir = averaged_meta_dir(root)
    emb_path = avg_dir / "targets_text_clip.npy"
    id_path = avg_dir / "targets_text_image_id_73k.npy"
    cap_cache = avg_dir / "targets_text_captions.json"

    if not emb_path.exists() or not id_path.exists():
        raise FileNotFoundError(
            f"Missing {emb_path.name} or {id_path.name} under {avg_dir}. "
            "Run ingest_averaged.py first."
        )

    embeddings = np.load(emb_path).astype(np.float32)
    image_ids = np.load(id_path).astype(np.int64)
    n = len(image_ids)
    if embeddings.shape[0] != n:
        raise ValueError(f"targets_text_clip rows {embeddings.shape[0]} != ids {n}")

    if cap_cache.exists():
        captions = json.loads(cap_cache.read_text())
        if len(captions) != n:
            raise ValueError(f"{cap_cache.name} length {len(captions)} != {n}")
    else:
        meta_dir = root / "nsd_meta"
        coco_id_73k = _load_coco_id_73k(root)
        by_image = json.loads(ensure_coco_captions_json(meta_dir).read_text())
        captions = []
        missing = 0
        for i73 in image_ids:
            i73 = int(i73)
            cap = ""
            if 0 <= i73 < len(coco_id_73k):
                coco_id = int(coco_id_73k[i73])
                caps = by_image.get(str(coco_id), [])
                if caps:
                    cap = str(caps[0])
            if not cap:
                missing += 1
            captions.append(cap)
        cap_cache.write_text(json.dumps(captions))
        print(f"  Cached {cap_cache.name} ({n} rows, {missing} without caption)")

    emb_t = torch.from_numpy(embeddings)
    emb_t = F.normalize(emb_t, dim=-1)
    print(f"  COCO retrieval bank: {n} embeddings, {sum(bool(c) for c in captions)} captions")
    return COCORetrievalBank(embeddings=emb_t, captions=captions)


@torch.no_grad()
def retrieve_top1_captions(
    query_768: torch.Tensor,
    bank: COCORetrievalBank,
    *,
    device: torch.device,
) -> tuple[list[str], list[int], torch.Tensor]:
    """
    Cosine top-1 retrieval against the global caption embedding bank.

    query_768: (B, 768) brain text_projector outputs (L2-normalized).
    Returns captions, bank indices, and similarity scores.
    """
    q = F.normalize(query_768.float().to(device), dim=-1)
    bank_dev = bank.embeddings.to(device)
    sim = q @ bank_dev.T
    idx = sim.argmax(dim=1)
    scores = sim.gather(1, idx.unsqueeze(1)).squeeze(1)
    caps = [bank.captions[int(i)] if bank.captions[int(i)] else "a photograph" for i in idx.cpu().tolist()]
    return caps, idx.cpu().tolist(), scores.cpu()


def load_hf_vd_image_variation_pipe(device: torch.device):
    from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel, VersatileDiffusionImageVariationPipeline
    from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

    dtype = torch.float16
    bin_kw = _bin_kw()
    print(f"  Loading HF VD image-variation from {VD_REPO}...")
    pipe = VersatileDiffusionImageVariationPipeline(
        image_encoder=CLIPVisionModelWithProjection.from_pretrained(
            VD_REPO, subfolder="image_encoder", torch_dtype=dtype, **bin_kw,
        ),
        image_feature_extractor=CLIPImageProcessor.from_pretrained(
            VD_REPO, subfolder="image_feature_extractor",
        ),
        image_unet=UNet2DConditionModel.from_pretrained(
            VD_REPO, subfolder="image_unet", torch_dtype=dtype, **bin_kw,
        ),
        vae=AutoencoderKL.from_pretrained(VD_REPO, subfolder="vae", torch_dtype=dtype, **bin_kw),
        scheduler=DDIMScheduler.from_pretrained(VD_REPO, subfolder="scheduler"),
    )
    pipe = pipe.to(device)
    return pipe


def _patch_dual_transformer_module(module: torch.nn.Module) -> int:
    """Patch DualTransformer2DModel.forward to ignore encoder_attention_mask (new UNet API)."""
    patched_classes: set[type] = set()
    n = 0
    for m in module.modules():
        cls = m.__class__
        if cls.__name__ != "DualTransformer2DModel" or cls in patched_classes:
            continue
        _orig = cls.forward

        def _forward(self, hidden_states, encoder_hidden_states=None, **kwargs):
            return _orig(self, hidden_states, encoder_hidden_states)

        cls.forward = _forward
        patched_classes.add(cls)
        n += 1
    return n


def _import_vd_dual_guided_pipeline():
    try:
        from diffusers import VersatileDiffusionDualGuidedPipeline
        return VersatileDiffusionDualGuidedPipeline
    except ImportError:
        from diffusers.pipelines.deprecated.versatile_diffusion import (
            VersatileDiffusionDualGuidedPipeline,
        )
        return VersatileDiffusionDualGuidedPipeline


def load_hf_vd_dual_guided_pipe(device: torch.device):
    """Load dual-guided VD (manual components — Hub model_index references missing text_unet module)."""
    from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
    from diffusers.pipelines.deprecated.versatile_diffusion.modeling_text_unet import UNetFlatConditionModel
    from transformers import (
        CLIPImageProcessor,
        CLIPTextModelWithProjection,
        CLIPTokenizer,
        CLIPVisionModelWithProjection,
    )

    dtype = torch.float16
    bin_kw = _bin_kw()
    cls = _import_vd_dual_guided_pipeline()
    print(f"  Loading HF VD dual-guided components from {VD_REPO}...")
    pipe = cls(
        vae=AutoencoderKL.from_pretrained(VD_REPO, subfolder="vae", torch_dtype=dtype, **bin_kw),
        image_unet=UNet2DConditionModel.from_pretrained(
            VD_REPO, subfolder="image_unet", torch_dtype=dtype, **bin_kw,
        ),
        text_unet=UNetFlatConditionModel.from_pretrained(
            VD_REPO, subfolder="text_unet", torch_dtype=dtype, **bin_kw,
        ),
        text_encoder=CLIPTextModelWithProjection.from_pretrained(
            CLIP_REPO, torch_dtype=dtype, **bin_kw,
        ),
        tokenizer=CLIPTokenizer.from_pretrained(CLIP_REPO),
        image_encoder=CLIPVisionModelWithProjection.from_pretrained(
            VD_REPO, subfolder="image_encoder", torch_dtype=dtype, **bin_kw,
        ),
        image_feature_extractor=CLIPImageProcessor.from_pretrained(
            VD_REPO, subfolder="image_feature_extractor",
        ),
        scheduler=DDIMScheduler.from_pretrained(VD_REPO, subfolder="scheduler"),
    )
    pipe.remove_unused_weights()
    n = _patch_dual_transformer_module(pipe.image_unet)
    if n:
        print(f"  Patched {n} DualTransformer2DModel blocks (drop encoder_attention_mask)")
    pipe = pipe.to(device)
    return pipe


def _brain_prior_pooled_embedding(
    prior_768: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    *,
    do_cfg: bool,
) -> torch.Tensor:
    """(B, 1, 768) brain prior as VD image-variation conditioning."""
    cond = F.normalize(prior_768.float().view(1, 1, 768), dim=-1).to(device=device, dtype=dtype)
    if not do_cfg:
        return cond
    uncond = torch.zeros_like(cond)
    return torch.cat([uncond, cond], dim=0)


def _brain_prior_image_tokens(
    pipe,
    prior_768: torch.Tensor,
    init_image: Image.Image,
    device: torch.device,
    *,
    do_cfg: bool,
) -> torch.Tensor:
    """257 VD vision tokens with CLS replaced by brain prior (768-d)."""
    pixel = pipe.image_feature_extractor(images=init_image, return_tensors="pt")
    pixel_values = pixel.pixel_values.to(device).to(pipe.image_encoder.dtype)
    raw = pipe.image_encoder(pixel_values)
    cond = _normalize_vd_image_tokens(raw, pipe)
    prior = F.normalize(prior_768.float().view(1, 1, 768).to(device), dim=-1).to(cond.dtype)
    cond = cond.clone()
    cond[:, 0:1, :] = prior
    pooled = cond[:, 0:1]
    cond = cond / torch.norm(pooled, dim=-1, keepdim=True)

    if not do_cfg:
        return cond

    uncond_pix = pipe.image_feature_extractor(
        images=[np.zeros((512, 512, 3), dtype=np.float32) + 0.5],
        return_tensors="pt",
    )
    uvals = uncond_pix.pixel_values.to(device).to(pipe.image_encoder.dtype)
    uncond = _normalize_vd_image_tokens(pipe.image_encoder(uvals), pipe)
    return torch.cat([uncond, cond], dim=0)


def _encode_text_prompt_embeds(pipe, captions: list[str], device: torch.device, *, do_cfg: bool) -> torch.Tensor:
    """(B*cfg, 77, 768) token sequence via CLIP text encoder + VD normalization."""
    return pipe._encode_text_prompt(captions, device, 1, do_cfg)


def _renorm_hf_text_tokens(seq: torch.Tensor, pooler_vec: torch.Tensor) -> torch.Tensor:
    """VD CLIP-text norm: scale all tokens by inverse pooler-vector norm."""
    pn = torch.norm(pooler_vec.float().reshape(-1)).clamp(min=1e-6)
    return seq / pn


@torch.no_grad()
def brain_text_to_hf_prompt_embeds(
    pipe,
    brain_768: torch.Tensor,
    device: torch.device,
    *,
    do_cfg: bool,
    brain_token_weight: float = 0.85,
    layout_prompt: str = "a photograph",
) -> torch.Tensor:
    """
    (B*cfg, 77, 768) HF dual-guided text branch from brain text_projector (768-d).

    Mirrors Brain Diffuser brain_clip_to_vd_text_tokens: short layout prompt + brain blend.
    """
    layout = _encode_text_prompt_embeds(pipe, [layout_prompt], device, do_cfg=False)
    brain = F.normalize(brain_768.float().view(1, -1), dim=-1).to(device=device, dtype=layout.dtype)
    w = float(np.clip(brain_token_weight, 0.0, 1.0))
    brain_exp = brain.unsqueeze(1).expand(-1, layout.shape[1], -1)
    cond = (1.0 - w) * layout + w * brain_exp
    cond[:, 0] = brain[0]
    cond = _renorm_hf_text_tokens(cond, brain[0])

    if not do_cfg:
        return cond
    uncond = _encode_text_prompt_embeds(pipe, [""], device, do_cfg=False)
    return torch.cat([uncond, cond], dim=0)


@torch.no_grad()
def _img2img_latents(
    pipe,
    init_image: Image.Image,
    *,
    strength: float,
    num_inference_steps: int,
    device: torch.device,
    generator: torch.Generator | None,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Encode init image to noisy latents at img2img strength (DDIM).

    Matches original reconstruct_imagery_small.py indexing:
      n_keep = round(steps * strength); noise at timesteps[n_keep-1]; denoise from there.
    """
    x = _pil_to_vae_tensor(init_image, device, dtype)
    init_latents = pipe.vae.encode(x).latent_dist.sample()
    init_latents = init_latents * pipe.vae.config.scaling_factor

    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = pipe.scheduler.timesteps
    strength = float(np.clip(strength, 0.0, 1.0))
    n_keep = max(1, int(round(num_inference_steps * strength)))
    start_t = timesteps[n_keep - 1]

    noise = torch.randn(init_latents.shape, generator=generator, device=device, dtype=init_latents.dtype)
    latents = pipe.scheduler.add_noise(init_latents, noise, start_t)
    return latents, timesteps[n_keep - 1:]


@torch.no_grad()
def stage2_hf_vd_img2img(
    pipe,
    init_image: Image.Image,
    prior_768: torch.Tensor,
    *,
    strength: float = 0.3,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    generator: torch.Generator | None = None,
) -> Image.Image:
    """
    HF image-variation img2img — exact logic from reconstruct_imagery_small.py
    at the 51.8% prior_adapted_vd03_t15 run (pooled prior 1×768, n_keep indexing).
    """
    device = pipe._execution_device
    dtype = pipe.vae.dtype

    init_tensor = _pil_to_vae_tensor(init_image, device, dtype)
    init_latents = pipe.vae.encode(init_tensor).latent_dist.sample() * pipe.vae.config.scaling_factor

    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = pipe.scheduler.timesteps
    n_keep = max(1, int(round(num_inference_steps * strength)))
    t_start = timesteps[n_keep - 1]

    noise = torch.randn(init_latents.shape, generator=generator, device=device, dtype=dtype)
    latents = pipe.scheduler.add_noise(init_latents, noise, t_start)

    clip_emb = F.normalize(prior_768.float(), dim=-1).to(device=device, dtype=dtype)
    clip_emb = clip_emb.unsqueeze(0).unsqueeze(0)  # (1, 1, 768)
    do_cfg = guidance_scale > 1.0
    if do_cfg:
        uncond = torch.zeros_like(clip_emb)
        image_embeddings = torch.cat([uncond, clip_emb])
    else:
        image_embeddings = clip_emb

    extra_step_kwargs = pipe.prepare_extra_step_kwargs(generator, eta=0.0)
    denoise_timesteps = timesteps[n_keep - 1:]

    for t in denoise_timesteps:
        latent_in = torch.cat([latents] * 2) if do_cfg else latents
        latent_in = pipe.scheduler.scale_model_input(latent_in, t)
        noise_pred = pipe.image_unet(latent_in, t, encoder_hidden_states=image_embeddings).sample
        if do_cfg:
            noise_uncond, noise_cond = noise_pred.chunk(2)
            noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
        latents = pipe.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

    decoded = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False)[0]
    out = pipe.image_processor.postprocess(decoded, output_type="pil")[0]
    return out.resize((256, 256), Image.LANCZOS)


def stage2_hf_vd_img2img_train(
    pipe,
    init_image: Image.Image,
    prior_768: torch.Tensor,
    *,
    strength: float = 0.3,
    num_inference_steps: int = 20,
    guidance_scale: float = 7.5,
    output_size: int = 256,
) -> torch.Tensor:
    """
    Differentiable HF VD img2img (frozen UNet/VAE weights; grad via CLIP conditioning).

    Returns (1, 3, output_size, output_size) float in [0, 1].
    """
    device = pipe._execution_device
    dtype = pipe.vae.dtype

    init_tensor = _pil_to_vae_tensor(init_image, device, dtype)
    init_latents = pipe.vae.encode(init_tensor).latent_dist.sample() * pipe.vae.config.scaling_factor

    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = pipe.scheduler.timesteps
    n_keep = max(1, int(round(num_inference_steps * strength)))
    t_start = timesteps[n_keep - 1]

    noise = torch.randn(init_latents.shape, device=device, dtype=dtype)
    latents = pipe.scheduler.add_noise(init_latents, noise, t_start)

    clip_emb = F.normalize(prior_768.float(), dim=-1).to(device=device, dtype=dtype)
    clip_emb = clip_emb.unsqueeze(0).unsqueeze(0)
    do_cfg = guidance_scale > 1.0
    if do_cfg:
        uncond = torch.zeros_like(clip_emb)
        image_embeddings = torch.cat([uncond, clip_emb])
    else:
        image_embeddings = clip_emb

    extra_step_kwargs = pipe.prepare_extra_step_kwargs(None, eta=0.0)
    denoise_timesteps = timesteps[n_keep - 1:]

    for t in denoise_timesteps:
        latent_in = torch.cat([latents] * 2) if do_cfg else latents
        latent_in = pipe.scheduler.scale_model_input(latent_in, t)
        noise_pred = pipe.image_unet(latent_in, t, encoder_hidden_states=image_embeddings).sample
        if do_cfg:
            noise_uncond, noise_cond = noise_pred.chunk(2)
            noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
        latents = pipe.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

    decoded = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False)[0]
    out = (decoded.float().clamp(-1, 1) + 1) * 0.5
    if output_size != out.shape[-1]:
        out = F.interpolate(out, size=(output_size, output_size), mode="bilinear", align_corners=False)
    return out


def _hf_dual_uncond_image_tokens(
    pipe,
    device: torch.device,
    *,
    do_cfg: bool,
) -> torch.Tensor:
    """Unconditional VD vision tokens (gray image); used for text-only Set-C conditioning."""
    uncond_pix = pipe.image_feature_extractor(
        images=[np.zeros((512, 512, 3), dtype=np.float32) + 0.5],
        return_tensors="pt",
    )
    uvals = uncond_pix.pixel_values.to(device).to(pipe.image_encoder.dtype)
    uncond = _normalize_vd_image_tokens(pipe.image_encoder(uvals), pipe)
    if not do_cfg:
        return uncond
    return torch.cat([uncond, uncond], dim=0)


def _hf_dual_img2img_loop(
    pipe,
    init_image: Image.Image,
    prompt_embeds: torch.Tensor,
    image_embeddings: torch.Tensor,
    *,
    strength: float,
    num_inference_steps: int,
    guidance_scale: float,
    text_to_image_strength: float,
    generator: torch.Generator | None,
) -> Image.Image:
    """Shared HF dual-guided img2img denoise (text + image conditioning)."""
    device = pipe._execution_device
    dtype = pipe.image_unet.dtype
    do_cfg = guidance_scale > 1.0

    dual_prompt_embeddings = torch.cat([prompt_embeds, image_embeddings], dim=1)
    prompt_types = ("text", "image")

    latents, timesteps = _img2img_latents(
        pipe, init_image, strength=strength, num_inference_steps=num_inference_steps,
        device=device, generator=generator, dtype=dtype,
    )
    extra = pipe.prepare_extra_step_kwargs(generator, 0.0)
    pipe.set_transformer_params(float(text_to_image_strength), prompt_types)

    for t in timesteps:
        latent_in = torch.cat([latents] * 2) if do_cfg else latents
        latent_in = pipe.scheduler.scale_model_input(latent_in, t)
        noise_pred = pipe.image_unet(
            latent_in, t, encoder_hidden_states=dual_prompt_embeddings,
        ).sample
        if do_cfg:
            uncond, cond = noise_pred.chunk(2)
            noise_pred = uncond + guidance_scale * (cond - uncond)
        latents = pipe.scheduler.step(noise_pred, t, latents, **extra).prev_sample

    image = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False)[0]
    out = pipe.image_processor.postprocess(image, output_type="pil")[0]
    return out.resize((256, 256), Image.LANCZOS)


@torch.no_grad()
def stage2_hf_vd_dual_img2img(
    pipe,
    init_image: Image.Image,
    prior_768: torch.Tensor,
    caption: str,
    *,
    strength: float = 0.3,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    text_to_image_strength: float = 0.4,
    generator: torch.Generator | None = None,
) -> Image.Image:
    """
    HF VersatileDiffusionDualGuided img2img with custom conditioning.

    image branch: brain prior replaces CLS in tokens from Stage-1 image.
    text branch: retrieved caption → 77×768 CLIP tokens (openai/clip-vit-large-patch14).
    """
    device = pipe._execution_device
    do_cfg = guidance_scale > 1.0
    prompt_embeds = _encode_text_prompt_embeds(pipe, [caption], device, do_cfg=do_cfg)
    image_embeddings = _brain_prior_image_tokens(
        pipe, prior_768, init_image, device, do_cfg=do_cfg,
    )
    return _hf_dual_img2img_loop(
        pipe, init_image, prompt_embeds, image_embeddings,
        strength=strength,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        text_to_image_strength=text_to_image_strength,
        generator=generator,
    )


@torch.no_grad()
def stage2_hf_vd_dual_brain_text_img2img(
    pipe,
    init_image: Image.Image,
    prior_768: torch.Tensor,
    brain_text_768: torch.Tensor,
    *,
    strength: float = 0.3,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    text_to_image_strength: float = 0.15,
    brain_token_weight: float = 0.85,
    generator: torch.Generator | None = None,
) -> Image.Image:
    """
    HF dual-guided img2img: brain prior (vision) + brain text_projector (text).

    Uses low text_to_image_strength by default so vision stays close to the 51.8% image-only path.
    """
    device = pipe._execution_device
    do_cfg = guidance_scale > 1.0
    prompt_embeds = brain_text_to_hf_prompt_embeds(
        pipe, brain_text_768, device,
        do_cfg=do_cfg,
        brain_token_weight=brain_token_weight,
    )
    image_embeddings = _brain_prior_image_tokens(
        pipe, prior_768, init_image, device, do_cfg=do_cfg,
    )
    return _hf_dual_img2img_loop(
        pipe, init_image, prompt_embeds, image_embeddings,
        strength=strength,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        text_to_image_strength=text_to_image_strength,
        generator=generator,
    )


@torch.no_grad()
def stage2_hf_vd_dual_text_only_img2img(
    pipe,
    init_image: Image.Image,
    brain_text_768: torch.Tensor,
    *,
    strength: float = 0.3,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    brain_token_weight: float = 0.95,
    generator: torch.Generator | None = None,
) -> Image.Image:
    """
    HF dual-guided img2img: brain text_projector only (Set C).

    Image branch is unconditional (no brain prior); text branch fully active (strength=1.0).
    """
    device = pipe._execution_device
    do_cfg = guidance_scale > 1.0
    prompt_embeds = brain_text_to_hf_prompt_embeds(
        pipe, brain_text_768, device,
        do_cfg=do_cfg,
        brain_token_weight=brain_token_weight,
    )
    image_embeddings = _hf_dual_uncond_image_tokens(pipe, device, do_cfg=do_cfg)
    return _hf_dual_img2img_loop(
        pipe, init_image, prompt_embeds, image_embeddings,
        strength=strength,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        text_to_image_strength=1.0,
        generator=generator,
    )
