# -*- coding: utf-8 -*-
"""Shared checkpoint load + inference for Set-B reconstruction scripts."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Variant A (3-head) — kept local so recon works without train_variant_a.py ──

class BrainMLP(nn.Module):
    def __init__(self, n_voxels, d_model=256, M=16, dropout=0.5):
        super().__init__()
        self.M, self.d_model = M, d_model
        self.net = nn.Sequential(
            nn.Linear(n_voxels, 4096), nn.LayerNorm(4096), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(4096, 4096), nn.LayerNorm(4096), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(4096, M * d_model),
        )

    def forward(self, x):
        return self.net(x).view(x.size(0), self.M, self.d_model)


class LatentQueryDecoder(nn.Module):
    def __init__(self, d_model=256, M=16, n_heads=8, K=2, dropout=0.1):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(1, M, d_model) * 0.02)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "self_attn": nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True),
                "cross_attn": nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True),
                "ffn": nn.Sequential(
                    nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Linear(d_model * 4, d_model),
                ),
                "norm1": nn.LayerNorm(d_model),
                "norm2": nn.LayerNorm(d_model),
                "norm3": nn.LayerNorm(d_model),
            })
            for _ in range(K)
        ])

    def forward(self, brain_tokens):
        q = self.queries.expand(brain_tokens.size(0), -1, -1)
        for layer in self.layers:
            q2, _ = layer["self_attn"](q, q, q)
            q = layer["norm1"](q + q2)
            q2, _ = layer["cross_attn"](q, brain_tokens, brain_tokens)
            q = layer["norm2"](q + q2)
            q = layer["norm3"](q + layer["ffn"](q))
        return q


class MindBridgeMLP(nn.Module):
    """Variant A — CLIP + DINO + VAE (768-d CLIP)."""

    def __init__(self, n_voxels, d_model=256, M=16, d_clip=768, d_dino=1024,
                 vae_shape=(4, 64, 64), dropout=0.5):
        super().__init__()
        self.encoder = BrainMLP(n_voxels, d_model, M, dropout)
        self.decoder = LatentQueryDecoder(d_model, M)
        self.clip_head = nn.Linear(d_model, d_clip)
        self.dino_head = nn.Linear(d_model, d_dino)
        self.vae_head = nn.Linear(d_model * M, vae_shape[0] * vae_shape[1] * vae_shape[2])
        self.vae_shape = vae_shape
        self.M = M

    def forward(self, x):
        brain_tokens = self.encoder(x)
        visual_tokens = self.decoder(brain_tokens)
        pooled = visual_tokens.mean(dim=1)
        pred_clip = self.clip_head(pooled)
        pred_dino = self.dino_head(pooled)
        flat = visual_tokens.reshape(visual_tokens.size(0), -1)
        pred_vae = self.vae_head(flat).view(-1, *self.vae_shape)
        return pred_clip, pred_dino, pred_vae


VARIANT_CHOICES = ("A", "4H", "4H_CTR", "4H_CTR2", "4H_CTR2_1H", "dual_ctr")
CLIP_SOURCE_CHOICES = ("regression", "projector")
CKPT_PREFERS = ("final", "best_clip", "best_retrieval")

DEFAULT_RUN_IDS = {
    "A": "20260601_052537_varA_subj01_150ep",
    "4H": "20260601_4head_150ep_all8",
    "4H_CTR": "20260602_034428_train_contrastive_retrieval_subj01_150ep",
    "4H_CTR2": "20260602_053625_train_dual_contrastive_subj01_150ep",
    "dual_ctr": "20260602_053625_train_dual_contrastive_subj01_150ep",
}

VARIANT_LABELS = {
    "A": "Variant A",
    "4H": "Global 4-head",
    "4H_CTR": "4-head + contrastive retrieval",
    "4H_CTR2": "4-head + dual contrastive (image + text projectors)",
    "4H_CTR2_1H": "4-head + single collapsed contrastive projector (ablation)",
    "dual_ctr": "4-head + dual contrastive (image + text projectors)",
}

_CLIP_VISUAL_PROJ: nn.Linear | None = None


def normalize_variant(variant: str) -> str:
    v = variant.strip().upper()
    if v in ("4HEAD_CTR", "4H-CTR"):
        return "4H_CTR"
    if v in ("4H_CTR2", "DUAL_CTR", "DUAL-CTR"):
        return "4H_CTR2"
    if v in ("4H_CTR2_1H", "1HEAD", "1H"):
        return "4H_CTR2_1H"
    return v


def build_recon_model(variant: str, n_voxels: int, device: torch.device) -> nn.Module:
    """Instantiate the architecture matching a training variant."""
    v = normalize_variant(variant)
    if v == "A":
        model = MindBridgeMLP(n_voxels=n_voxels)
    elif v == "4H":
        from train_4head import MindBridge4Head
        model = MindBridge4Head(n_voxels=n_voxels)
    elif v == "4H_CTR":
        from train_contrastive_retrieval import MindBridgeContrastive
        model = MindBridgeContrastive(n_voxels=n_voxels)
    elif v == "4H_CTR2":
        from train_dual_contrastive import MindBridgeContrastive
        model = MindBridgeContrastive(n_voxels=n_voxels)
    elif v == "4H_CTR2_1H":
        from train_dual_contrastive_1head import MindBridgeContrastive1Head
        model = MindBridgeContrastive1Head(n_voxels=n_voxels)
    else:
        raise ValueError(f"Unknown variant {variant!r}; choose from {VARIANT_CHOICES}")
    return model.to(device)


def _load_clip_visual_projection(device: torch.device) -> nn.Linear:
    """Frozen CLIP ViT-L/14 visual_projection: 1024 → 768 (CLS path)."""
    global _CLIP_VISUAL_PROJ
    if _CLIP_VISUAL_PROJ is None:
        from transformers import CLIPModel
        clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        _CLIP_VISUAL_PROJ = clip.visual_projection.to(device).eval()
        for p in _CLIP_VISUAL_PROJ.parameters():
            p.requires_grad = False
    return _CLIP_VISUAL_PROJ


@torch.no_grad()
def projector_to_clip768(proj_1024: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Map 1024-d contrastive projector output → 768-d CLIP image embedding space
    for Versatile Diffusion (uses the same visual_projection as CLS token path).
    """
    proj = F.normalize(proj_1024.float(), dim=-1).to(device)
    out = _load_clip_visual_projection(device)(proj)
    return F.normalize(out, dim=-1)


@torch.no_grad()
def forward_recon(
    model: nn.Module,
    variant: str,
    betas: torch.Tensor,
    *,
    clip_source: str = "regression",
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (pred_clip_768, pred_vae) for the two-stage recon pipeline.

    clip_source:
      regression — clip_image_head (768-d); default for all variants
      projector  — 4H_CTR / 4H_CTR2: 1024-d image projector → visual_projection → 768-d
    """
    v = normalize_variant(variant)
    clip_source = clip_source.lower()
    if clip_source not in CLIP_SOURCE_CHOICES:
        raise ValueError(f"clip_source must be one of {CLIP_SOURCE_CHOICES}")

    if v == "A":
        pred_clip, _, pred_vae = model(betas)
    elif v == "4H":
        pred_clip, _, _, pred_vae = model(betas)
    elif v in ("4H_CTR", "4H_CTR2", "4H_CTR2_1H"):
        out = model(betas)
        pred_clip, _, _, pred_vae, proj = out[0], out[1], out[2], out[3], out[4]
        if clip_source == "projector":
            pred_clip = projector_to_clip768(proj, device or betas.device)
    else:
        raise ValueError(variant)

    if clip_source == "projector" and v not in ("4H_CTR", "4H_CTR2", "4H_CTR2_1H"):
        raise ValueError("--clip-source projector requires variant 4H_CTR or 4H_CTR2")

    return pred_clip, pred_vae
