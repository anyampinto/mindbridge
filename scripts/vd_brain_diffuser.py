# -*- coding: utf-8 -*-
"""Native Versatile Diffusion (Brain Diffuser vendor) for Stage-2 img2img + dual conditioning."""

from __future__ import annotations

import os
import sys
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as tvtrans
from PIL import Image

from vdvae_brain_diffuser import _brain_diffuser_root

VD_VISION_TOKENS = 257
VD_TEXT_TOKENS = 77
VD_DEFAULT_TEXT_MIXING = 0.4
VD_CKPT_NAME = "vd-four-flow-v1-0-fp16-deprecated.pth"
VD_PRETRAINED_FILES = {
    VD_CKPT_NAME: (
        "https://huggingface.co/shi-labs/versatile-diffusion/"
        "resolve/main/pretrained_pth/vd-four-flow-v1-0-fp16-deprecated.pth"
    ),
    "kl-f8.pth": (
        "https://huggingface.co/shi-labs/versatile-diffusion/"
        "resolve/main/pretrained_pth/kl-f8.pth"
    ),
    "optimus-vae.pth": (
        "https://huggingface.co/shi-labs/versatile-diffusion/"
        "resolve/main/pretrained_pth/optimus-vae.pth"
    ),
}


@dataclass
class BDVDBundle:
    net: object
    sampler: object
    device: torch.device
    fp16: bool = True
    ckpt_path: str = ""


def _vd_dir() -> Path:
    return _brain_diffuser_root() / "versatile_diffusion"


def vd_pretrained_path(root: Path) -> Path:
    d = root / "vdvae" / "versatile_diffusion"
    d.mkdir(parents=True, exist_ok=True)
    return d / VD_CKPT_NAME


def download_vd_pretrained(root: Path, force: bool = False) -> Path:
    """Fetch VD four-flow + autokl + optimus weights to the MindBridge volume."""
    pre_dir = vd_pretrained_path(root).parent
    pre_dir.mkdir(parents=True, exist_ok=True)
    for name, url in VD_PRETRAINED_FILES.items():
        dest = pre_dir / name
        if dest.exists() and not force:
            continue
        print(f"  Downloading {name} from HuggingFace...")
        tmp = dest.with_suffix(dest.suffix + ".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(dest)
        print(f"  Saved {dest}")
    main = pre_dir / VD_CKPT_NAME
    print(f"  VD weights OK: {main}")
    return main


def _patch_vd_cfg_pretrained(cfgm, pre_dir: Path) -> None:
    """Point submodule checkpoints at volume paths (not vendor/pretrained)."""
    cfgm.args.autokl_cfg.pth = str(pre_dir / "kl-f8.pth")
    cfgm.args.optimus_cfg.pth = str(pre_dir / "optimus-vae.pth")


@contextmanager
def _bd_vd_runtime():
    """Brain Diffuser configs resolve paths relative to repo root."""
    bd_root = _brain_diffuser_root()
    vd_sub = str(_vd_dir())
    old_cwd = os.getcwd()
    old_path = list(sys.path)
    if vd_sub not in sys.path:
        sys.path.insert(0, vd_sub)
    os.chdir(bd_root)
    try:
        yield
    finally:
        os.chdir(old_cwd)
        sys.path[:] = old_path


def regularize_image_512(x) -> torch.Tensor:
    bicubic = Image.Resampling.BICUBIC
    if isinstance(x, Image.Image):
        x = x.resize([512, 512], resample=bicubic)
        x = tvtrans.ToTensor()(x)
    elif isinstance(x, torch.Tensor):
        if x.dim() == 3 and x.shape[-1] != 512:
            x = tvtrans.functional.resize(x.unsqueeze(0), [512, 512]).squeeze(0)
    else:
        raise TypeError(f"Expected PIL or Tensor, got {type(x)}")
    assert x.shape[1] == 512 and x.shape[2] == 512
    return x


def _as_batch_768(emb: torch.Tensor, device: torch.device) -> torch.Tensor:
    z = F.normalize(emb.float(), dim=-1).to(device)
    if z.dim() == 1:
        z = z.unsqueeze(0)
    if z.shape[-1] != 768:
        raise ValueError(f"Expected 768-d CLIP, got {z.shape[-1]}")
    return z


def _vd_renorm_vision_tokens(seq: torch.Tensor) -> torch.Tensor:
    """Re-apply VD CLIP-vision norm: divide all tokens by CLS (index 0) norm."""
    cls_norm = torch.norm(seq[:, 0:1], dim=-1, keepdim=True).clamp(min=1e-6)
    return seq / cls_norm


def _vd_renorm_text_tokens(seq: torch.Tensor, pooler_vec: torch.Tensor) -> torch.Tensor:
    """Re-apply VD CLIP-text norm: divide all tokens by pooler-vector norm (scalar)."""
    pn = torch.norm(pooler_vec.float().reshape(-1)).clamp(min=1e-6)
    return seq / pn


@torch.no_grad()
def brain_clip_to_vd_vision_tokens(
    net,
    brain_768: torch.Tensor,
    init_image: Image.Image | torch.Tensor,
    *,
    device: torch.device,
    fp16: bool,
    cls_brain_weight: float = 1.0,
) -> torch.Tensor:
    """
    (1, 257, 768) via frozen VD CLIP: patch tokens from Stage-1 image, CLS from brain prior.

    Matches Brain Diffuser ridge targets (per-token vision) better than repeating pooled 768-d.
    """
    zim = regularize_image_512(init_image)
    zin = (zim * 2 - 1).unsqueeze(0).to(device)
    if fp16:
        zin = zin.half()
    seq = net.clip_encode_vision(zin)
    brain = _as_batch_768(brain_768, device)
    w = float(cls_brain_weight)
    if w >= 1.0:
        seq[:, 0:1] = brain.unsqueeze(1)
    elif w > 0.0:
        seq[:, 0] = F.normalize((1.0 - w) * seq[:, 0] + w * brain[0], dim=-1)
    seq = _vd_renorm_vision_tokens(seq.float())
    return seq.half() if fp16 else seq


@torch.no_grad()
def brain_clip_to_vd_text_tokens(
    net,
    brain_768: torch.Tensor,
    *,
    device: torch.device,
    fp16: bool,
    layout_prompt: str = "a photograph",
    brain_token_weight: float = 0.85,
) -> torch.Tensor:
    """
    (1, 77, 768) via frozen VD CLIP: token layout from a short prompt, semantics from brain text CLIP.

    Injects brain embedding at BOS and blends all positions toward brain (not flat repeat).
    """
    brain = _as_batch_768(brain_768, device)
    layout = net.clip_encode_text(layout_prompt).float()
    w = float(brain_token_weight)
    brain_exp = brain.unsqueeze(1).expand(-1, VD_TEXT_TOKENS, -1)
    seq = (1.0 - w) * layout + w * brain_exp
    seq[:, 0] = brain[0]
    seq = _vd_renorm_text_tokens(seq, brain[0])
    return seq.half() if fp16 else seq


def clip768_to_vd_vision(emb: torch.Tensor, device: torch.device, *, fp16: bool) -> torch.Tensor:
    """Legacy fallback: repeat pooled 768-d (avoid when Stage-1 image is available)."""
    z = _as_batch_768(emb, device)
    z = z.unsqueeze(1).expand(-1, VD_VISION_TOKENS, -1)
    return z.half() if fp16 else z


def clip768_to_vd_text(emb: torch.Tensor, device: torch.device, *, fp16: bool) -> torch.Tensor:
    """Legacy fallback: repeat pooled 768-d."""
    z = _as_batch_768(emb, device)
    z = z.unsqueeze(1).expand(-1, VD_TEXT_TOKENS, -1)
    return z.half() if fp16 else z


def load_bd_vd_bundle(
    device: torch.device,
    root: Path | None = None,
    *,
    fp16: bool = True,
) -> BDVDBundle:
    """Load SHI-Labs VD (vd_noema) + DDIMSampler_VD with dual image/text UNet."""
    if device.type != "cuda":
        raise RuntimeError("Brain Diffuser VD requires CUDA")
    root = root or Path(os.environ.get("MINDBRIDGE_ROOT", "/mnt/mindbridge"))
    ckpt = download_vd_pretrained(root)

    with _bd_vd_runtime():
        from lib.cfg_helper import model_cfg_bank
        from lib.model_zoo import get_model
        from lib.model_zoo.ddim_vd import DDIMSampler_VD

        cfgm = model_cfg_bank()("vd_noema")
        pre_dir = ckpt.parent
        _patch_vd_cfg_pretrained(cfgm, pre_dir)
        device_str = str(device)
        cfgm.args.autokl_cfg.map_location = device_str
        cfgm.args.optimus_cfg.map_location = device_str
        net = get_model()(cfgm)
        sd = torch.load(ckpt, map_location=device, weights_only=False)
        net.load_state_dict(sd, strict=False)
        if fp16:
            net.clip.fp16 = True
            net = net.half()
        net.to(device)
        net.eval()
        if fp16:
            net.autokl.half()
        sampler = DDIMSampler_VD(net)
        dm = net.model.diffusion_model
        dm.device = device_str
        if fp16:
            dm.half()
        dm.to(device)

    print(
        f"  Loaded Brain Diffuser VD (vd_noema, dual UNet) from {ckpt}\n"
        "  Stage-2 cond: frozen CLIP token sequences (257 vision / 77 text), not pooled repeat"
    )
    return BDVDBundle(net=net, sampler=sampler, device=device, fp16=fp16, ckpt_path=str(ckpt))


@torch.no_grad()
def _uncond_vision(net, device: torch.device, *, fp16: bool) -> torch.Tensor:
    dummy = torch.zeros((1, 3, 224, 224), device=device)
    u = net.clip_encode_vision(dummy)
    return u.half() if fp16 else u


@torch.no_grad()
def _uncond_text(net, device: torch.device, *, fp16: bool) -> torch.Tensor:
    u = net.clip_encode_text("")
    return u.half() if fp16 else u


@torch.no_grad()
def stage2_bd_vd_img2img(
    bundle: BDVDBundle,
    init_image: Image.Image,
    pred_clip_768: torch.Tensor,
    *,
    pred_text_clip: torch.Tensor | None = None,
    text_mixing: float = VD_DEFAULT_TEXT_MIXING,
    strength: float = 0.3,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
) -> Image.Image:
    """
    Img2img via Brain Diffuser DDIMSampler_VD.decode / decode_dc.

    text_mixing=0.4 → mixed_ratio=0.6 on vision branch (Brain Diffuser default).
    """
    net, sampler = bundle.net, bundle.sampler
    device, fp16 = bundle.device, bundle.fp16

    zim = regularize_image_512(init_image)
    zin = (zim * 2 - 1).unsqueeze(0).to(device)
    if fp16:
        zin = zin.half()

    init_latent = net.autokl_encode(zin)
    sampler.make_schedule(ddim_num_steps=num_inference_steps, ddim_eta=0.0, verbose=False)
    assert 0.0 <= strength <= 1.0
    t_enc = int(strength * num_inference_steps)
    t_tensor = torch.tensor([t_enc], device=device)
    z_enc = sampler.stochastic_encode(init_latent, t_tensor)

    uim = _uncond_vision(net, device, fp16=fp16)
    cim = brain_clip_to_vd_vision_tokens(
        net, pred_clip_768, init_image, device=device, fp16=fp16,
    )

    use_dual = pred_text_clip is not None
    if use_dual:
        utx = _uncond_text(net, device, fp16=fp16)
        ctx = brain_clip_to_vd_text_tokens(
            net, pred_text_clip, device=device, fp16=fp16,
        )
        mixed_ratio = float(1.0 - text_mixing)
        z_dec = sampler.decode_dc(
            z_enc,
            first_conditioning=[uim, cim],
            second_conditioning=[utx, ctx],
            t_start=t_enc,
            unconditional_guidance_scale=guidance_scale,
            xtype="image",
            first_ctype="vision",
            second_ctype="prompt",
            mixed_ratio=mixed_ratio,
        )
    else:
        z_dec = sampler.decode(
            z_enc,
            cim,
            t_enc,
            unconditional_guidance_scale=guidance_scale,
            unconditional_conditioning=uim,
            xtype="image",
            ctype="vision",
        )

    if fp16:
        z_dec = z_dec.half()
    decoded = net.autokl_decode(z_dec)
    out = torch.clamp((decoded[0] + 1.0) / 2.0, 0.0, 1.0)
    pil = tvtrans.ToPILImage()(out.cpu().float())
    return pil.resize((256, 256), Image.LANCZOS)
