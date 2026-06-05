# -*- coding: utf-8 -*-
"""
4-Head → 1-Head contrastive collapse ablation.

Replaces separate image + text contrastive projectors with a single global
ContrastiveProjector (1024-d). Text retrieval/loss uses the first 768 dims of
that same head output — forcing all contrastive signal through one bottleneck.
"""

from __future__ import annotations

import os
import sys

sys.path = [p for p in sys.path if "share/software" not in p and "jupyterlab" not in p]

import torch
import torch.nn as nn
import torch.nn.functional as F

import train_dual_contrastive as base

VARIANT = "4H_CTR2_1H"


class MindBridgeContrastive1Head(base.MindBridgeContrastive):
    """Single shared contrastive projector (no separate text_projector)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        del self.text_projector

    @property
    def text_projector(self):
        """Alias for training/eval code that references text_projector separately."""
        return self.projector

    def forward(self, x):
        brain_tokens = self.encoder(x)
        visual_tokens = self.decoder(brain_tokens)
        pooled = visual_tokens.mean(dim=1)
        flat = visual_tokens.reshape(x.size(0), -1)
        pred_ci = self.clip_image_head(pooled)
        pred_ct = self.clip_text_head(pooled)
        pred_dino = self.dino_head(pooled)
        pred_vae = self.vae_head(flat).view(-1, *self.vae_shape)
        proj = self.projector(pooled)
        text_proj = F.normalize(proj[:, :768], dim=-1)
        return pred_ci, pred_ct, pred_dino, pred_vae, proj, text_proj


def run_training(subj: str = "subj01", root=None, epochs: int = 80):
    """Shorter default epochs — fewer params, faster convergence."""
    base.VARIANT = VARIANT
    base.MindBridgeContrastive = MindBridgeContrastive1Head
    return base.run_training(subj=subj, root=root or "/mnt/mindbridge", epochs=epochs)


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    p = argparse.ArgumentParser(description="MindBridge 1-head contrastive collapse ablation")
    p.add_argument("--subj", default=os.environ.get("NSD_SUBJ", "subj01"))
    p.add_argument("--root", default=os.environ.get("MINDBRIDGE_ROOT", "/mnt/mindbridge"))
    p.add_argument("--epochs", type=int, default=int(os.environ.get("ONEHEAD_EPOCHS", "80")))
    p.add_argument("--temperature", type=float, default=base.TEMPERATURE)
    p.add_argument("--accum-steps", type=int, default=base.ACCUM_STEPS)
    p.add_argument("--w-contrastive", type=float, default=base.W_CONTRASTIVE)
    p.add_argument("--w-text-contrastive", type=float, default=base.W_TEXT_CONTRASTIVE)
    args = p.parse_args()

    base.TEMPERATURE = args.temperature
    base.ACCUM_STEPS = args.accum_steps
    base.W_CONTRASTIVE = args.w_contrastive
    base.W_TEXT_CONTRASTIVE = args.w_text_contrastive

    run_training(subj=args.subj, root=Path(args.root), epochs=args.epochs)
