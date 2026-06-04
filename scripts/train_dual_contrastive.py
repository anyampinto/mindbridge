# -*- coding: utf-8 -*-
"""
MindBridge with dual decoupled contrastive retrieval heads (image + text).

Key changes vs train_contrastive_retrieval.py (image-only):
  - Image projector: brain → 1024-d L2-norm, trained with BiMixCo → SoftCLIP
    against pooled 257-token ViT-L/14 hidden states. Falls back to 768-d CLS.
  - Text projector:  brain → 768-d L2-norm, trained with SoftCLIP against
    COCO caption CLIP text embeddings (already in targets_avg_{subj}.npz).
    Masked where captions are missing (zero rows). Targets semantic/conceptual
    content — expected to improve Set C (concept) imagery retrieval.
  - Both projectors are SEPARATE from each other and from the 4 regression heads.
  - Gradient accumulation for large effective batch (>=256 negatives).
  - Retrieval reported separately for image projector and text projector.
  - Early stopping on image projector retrieval (primary metric).

Loss schedule (MindEye):
  Epochs [0, mixup_pct * total):   BiMixCo  — bidirectional CLIP loss + Beta mixup
  Epochs [mixup_pct * total, end): SoftCLIP — soft labels from CLIP-image x CLIP-image
  Text projector always uses SoftCLIP (text embeddings have no mixup defined).
"""

import os
import sys

sys.path = [p for p in sys.path if "share/software" not in p and "jupyterlab" not in p]

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader, Dataset

from paths import (
    averaged_meta_dir,
    resolve_run_id,
    run_root,
    update_run_manifest,
    variant_ckpt_dir,
)
from training_checkpoints import (
    _base_payload,
    compute_val_clip_cossim,
    maybe_save_best_clip,
    retrieval_2way,
    save_final_checkpoint,
)

# ── variant tag ──────────────────────────────────────────────────────────────
VARIANT = "4H_CTR2"  # 4-head + image contrastive + text contrastive

# ── regression loss weights (unchanged from 4H baseline) ─────────────────────
W_CLIP_IMAGE = 1.0
W_CLIP_TEXT  = 0.3
W_DINO       = 0.1
W_VAE        = 0.001

# ── contrastive loss weights ──────────────────────────────────────────────────
W_CONTRASTIVE      = 1.0   # image contrastive (primary retrieval signal)
W_TEXT_CONTRASTIVE = 0.5   # text contrastive (semantic/concept signal); tune 0.25–1.0

# ── BiMixCo → SoftCLIP schedule ──────────────────────────────────────────────
MIXUP_PCT      = 0.33          # fraction of epochs using BiMixCo
MIXUP_ALPHA    = 0.15          # Beta(alpha, alpha) for MixCo
TEMPERATURE    = 0.04          # tune over {0.01, 0.03, 0.05, 0.07}

# ── effective batch size via gradient accumulation ────────────────────────────
ACCUM_STEPS    = 4             # effective_batch = batch_size * accum_steps
                               # 64 * 4 = 256 negatives; increase accum or batch if memory allows

# ── early stopping ────────────────────────────────────────────────────────────
PATIENCE       = 30            # epochs without improvement before stopping
MIN_DELTA      = 1e-3          # minimum improvement to reset patience

# ── projector head dims ───────────────────────────────────────────────────────
D_PROJ         = 1024          # must match hidden target dim (ViT-L/14 internal = 1024)

# ── 257-token hidden layer dim for ViT-L/14 ──────────────────────────────────
# ViT-L/14 last_hidden_state is (B, 257, 1024) — 1024 is the internal d_model.
# The 768-d CLS you've been using is AFTER visual_projection (a 1024→768 linear).
# We use the pre-projection hidden states for richer spatial token information.
D_HIDDEN_TOKEN = 1024          # ViT-L/14 internal dim (NOT 768)
N_TOKENS       = 257


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class AveragedNSDDataset(Dataset):
    """
    Returns (betas, clip_image_cls, clip_text, dino, vae, clip_hidden).
    clip_hidden is (257, 1024) if available, else zeros → signals fallback to CLS.
    """
    def __init__(self, betas, clip_i, clip_t, dino, vae, clip_hidden=None):
        self.betas      = torch.tensor(betas,   dtype=torch.float32)
        self.clip_i     = torch.tensor(clip_i,  dtype=torch.float32)
        self.clip_t     = torch.tensor(clip_t,  dtype=torch.float32)
        self.dino       = torch.tensor(dino,    dtype=torch.float32)
        self.vae        = torch.tensor(vae,     dtype=torch.float32)
        # clip_hidden: (N, 257, 1024) or None → store zeros as sentinel
        if clip_hidden is not None:
            self.clip_hidden = torch.tensor(clip_hidden, dtype=torch.float32)
            self.has_hidden  = True
        else:
            self.clip_hidden = torch.zeros(len(betas), N_TOKENS, D_HIDDEN_TOKEN)
            self.has_hidden  = False

    def __len__(self):
        return len(self.betas)

    def __getitem__(self, idx):
        return (
            self.betas[idx],
            self.clip_i[idx],
            self.clip_t[idx],
            self.dino[idx],
            self.vae[idx],
            self.clip_hidden[idx],
        )


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

class BrainMLP(nn.Module):
    def __init__(self, n_voxels, d_model=256, M=16, dropout=0.5):
        super().__init__()
        self.M, self.d_model = M, d_model
        self.net = nn.Sequential(
            nn.Linear(n_voxels, 4096), nn.LayerNorm(4096), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(4096, 4096),     nn.LayerNorm(4096), nn.GELU(), nn.Dropout(dropout),
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
                "self_attn":  nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True),
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


class ContrastiveProjector(nn.Module):
    """
    Separate MLP projector for retrieval. NEVER shares gradients with regression heads.
    Input:  pooled visual token  (B, d_model=256) — from BrainMLP, NOT from CLIP
    Output: L2-normalized vector (B, D_PROJ=1024) — matched to hidden target dim

    The CLIP 257-token hidden layer is the *target* for the loss, not the input.
    Input to this projector is always your brain encoder's d_model (256).
    """
    def __init__(self, d_model=256, d_out=D_PROJ, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_out),
        )

    def forward(self, pooled):
        return F.normalize(self.net(pooled), dim=-1)


class TextContrastiveProjector(nn.Module):
    """
    Separate projector for text-contrastive retrieval.
    Input:  pooled visual token (B, d_model=256)
    Output: L2-normalized (B, 768) — matches COCO caption CLIP text embedding dim.

    Trained with SoftCLIP loss against text embeddings from targets_avg_{subj}.npz.
    Completely separate from ContrastiveProjector — no shared weights.
    Masked during training for images without captions (zero text embedding rows).
    """
    def __init__(self, d_model=256, d_out=768, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_out),
        )

    def forward(self, pooled):
        return F.normalize(self.net(pooled), dim=-1)


class MindBridgeContrastive(nn.Module):
    def __init__(
        self,
        n_voxels,
        d_model=256,
        M=16,
        d_clip=768,
        d_dino=1024,
        vae_shape=(4, 64, 64),
        dropout=0.5,
    ):
        super().__init__()
        self.encoder   = BrainMLP(n_voxels, d_model, M, dropout)
        self.decoder   = LatentQueryDecoder(d_model, M)
        # regression heads
        self.clip_image_head = nn.Linear(d_model, d_clip)
        self.clip_text_head  = nn.Linear(d_model, d_clip)
        self.dino_head       = nn.Linear(d_model, d_dino)
        self.vae_head        = nn.Linear(d_model * M, vae_shape[0] * vae_shape[1] * vae_shape[2])
        self.vae_shape = vae_shape
        self.M = M
        # image contrastive projector — separate from everything else
        self.projector      = ContrastiveProjector(d_model, D_PROJ, dropout=0.1)
        # text contrastive projector — separate from image projector and regression
        self.text_projector = TextContrastiveProjector(d_model, d_out=768, dropout=0.1)

    def forward(self, x):
        brain_tokens  = self.encoder(x)
        visual_tokens = self.decoder(brain_tokens)
        pooled = visual_tokens.mean(dim=1)          # (B, d_model)
        flat   = visual_tokens.reshape(x.size(0), -1)
        # regression outputs
        pred_ci   = self.clip_image_head(pooled)
        pred_ct   = self.clip_text_head(pooled)
        pred_dino = self.dino_head(pooled)
        pred_vae  = self.vae_head(flat).view(-1, *self.vae_shape)
        # contrastive projector outputs (both L2-normalized)
        proj      = self.projector(pooled)       # (B, 1024) image contrastive
        text_proj = self.text_projector(pooled)  # (B, 768)  text contrastive
        return pred_ci, pred_ct, pred_dino, pred_vae, proj, text_proj


# ─────────────────────────────────────────────────────────────────────────────
# Loss functions
# ─────────────────────────────────────────────────────────────────────────────

def regression_loss(pred_ci, pred_ct, pred_dino, pred_vae,
                    tgt_ci, tgt_ct, tgt_dino, tgt_vae):
    """Unchanged from 4H baseline."""
    lc = (1 - F.cosine_similarity(pred_ci, tgt_ci, dim=-1)).mean()

    text_mask = tgt_ct.norm(dim=-1) > 0.01
    if text_mask.any():
        lt = (1 - F.cosine_similarity(pred_ct[text_mask], tgt_ct[text_mask], dim=-1)).mean()
    else:
        lt = torch.zeros((), device=pred_ct.device, dtype=pred_ct.dtype)

    ld = F.mse_loss(pred_dino, tgt_dino)
    lv = F.mse_loss(pred_vae,  tgt_vae)

    total = W_CLIP_IMAGE * lc + W_CLIP_TEXT * lt + W_DINO * ld + W_VAE * lv
    return total, lc.item(), lt.item(), ld.item(), lv.item()


def mixco_sample(betas_batch, clip_hidden_target, alpha=MIXUP_ALPHA):
    """
    BiMixCo: create mixed brain signals and soft contrastive targets.
    Returns (mixed_betas, perm, lam) where lam is per-sample mixing weight.
    Soft target for sample i = lam[i] * clip[i] + (1-lam[i]) * clip[perm[i]]
    then L2-normalized.
    """
    B = betas_batch.size(0)
    lam = torch.distributions.Beta(alpha, alpha).sample((B,)).to(betas_batch.device)
    lam = torch.max(lam, 1 - lam)  # keep dominant sample > 0.5
    perm = torch.randperm(B, device=betas_batch.device)
    mixed = lam.view(-1, 1) * betas_batch + (1 - lam).view(-1, 1) * betas_batch[perm]
    # soft clip target
    t1 = F.normalize(clip_hidden_target, dim=-1)
    t2 = F.normalize(clip_hidden_target[perm], dim=-1)
    soft_tgt = F.normalize(lam.view(-1, 1) * t1 + (1 - lam).view(-1, 1) * t2, dim=-1)
    return mixed, soft_tgt, lam, perm


def bimixco_loss(proj, soft_tgt, temperature=TEMPERATURE):
    """
    Bidirectional CLIP contrastive loss with mixed targets.
    proj:     (B, D) L2-normalized projector outputs
    soft_tgt: (B, D) L2-normalized soft CLIP targets
    """
    logits = proj @ soft_tgt.T / temperature  # (B, B)
    labels = torch.arange(logits.size(0), device=logits.device)
    loss_i = F.cross_entropy(logits, labels)
    loss_t = F.cross_entropy(logits.T, labels)
    return (loss_i + loss_t) / 2


def softclip_loss(proj, clip_tgt, temperature=TEMPERATURE):
    """
    SoftCLIP: soft labels are pairwise CLIP-image similarities (knowledge distillation).
    proj:     (B, D) L2-normalized projector outputs
    clip_tgt: (B, D) L2-normalized CLIP image targets (CLS or pooled hidden)
    """
    # soft labels: similarity between all pairs of CLIP targets
    with torch.no_grad():
        tgt_norm = F.normalize(clip_tgt, dim=-1)
        soft_labels = tgt_norm @ tgt_norm.T  # (B, B), in [-1,1]
        soft_labels = F.softmax(soft_labels / temperature, dim=-1)

    logits_i = proj @ tgt_norm.T / temperature   # (B, B)
    logits_t = tgt_norm @ proj.T / temperature   # (B, B)
    loss_i = -(soft_labels * F.log_softmax(logits_i, dim=-1)).sum(dim=-1).mean()
    loss_t = -(soft_labels * F.log_softmax(logits_t, dim=-1)).sum(dim=-1).mean()
    return (loss_i + loss_t) / 2


def text_contrastive_loss(text_proj, clip_text_tgt, temperature=TEMPERATURE):
    """
    SoftCLIP loss for the text projector head.
    text_proj:     (B, 768) L2-normalized text projector outputs
    clip_text_tgt: (B, 768) L2-normalized COCO caption CLIP text embeddings

    Only called on samples where the text embedding is valid (non-zero).
    Returns scalar loss, or zero if no valid samples in batch.
    """
    # mask out zero/missing caption rows
    valid = clip_text_tgt.norm(dim=-1) > 0.01
    if valid.sum() < 2:
        return torch.zeros((), device=text_proj.device, dtype=text_proj.dtype)

    tp  = text_proj[valid]
    tgt = F.normalize(clip_text_tgt[valid], dim=-1)
    return softclip_loss(tp, tgt, temperature=temperature)

def contrastive_loss(proj, clip_tgt_pooled, epoch, total_epochs,
                     betas_for_mixco=None):
    """
    Dispatches to BiMixCo (first MIXUP_PCT fraction) or SoftCLIP.
    clip_tgt_pooled: (B, 768) — pooled from 257-token hidden or CLS fallback, L2-normed.
    betas_for_mixco: raw brain betas for mixing (only needed for BiMixCo phase).
    Returns scalar loss.
    """
    use_mixco = epoch < int(MIXUP_PCT * total_epochs)

    if use_mixco and betas_for_mixco is not None:
        _, soft_tgt, _, _ = mixco_sample(betas_for_mixco, clip_tgt_pooled)
        return bimixco_loss(proj, soft_tgt, temperature=TEMPERATURE)
    else:
        return softclip_loss(proj, clip_tgt_pooled, temperature=TEMPERATURE)


def prepare_contrastive_target(clip_hidden, clip_cls, has_hidden):
    """
    If 257-token hidden layer is available: mean-pool across tokens → (B, 1024), L2-norm.
    Else fall back to CLS vector → (B, 768), L2-norm.

    Note: proj output is D_PROJ=1024 when using hidden path, 1024 for CLS path too
    (ContrastiveProjector always outputs D_PROJ). Dims must match for dot product in loss.
    We keep D_PROJ=1024 throughout so hidden and fallback both work.
    """
    if has_hidden:
        # clip_hidden: (B, 257, 1024) — check not zero sentinel via first token
        if clip_hidden[:, 0, :].norm(dim=-1).mean() > 0.01:
            pooled = clip_hidden.mean(dim=1)   # (B, 257, 1024) → (B, 1024)
            return F.normalize(pooled, dim=-1)
    # CLS fallback: (B, 768) — project to 1024 via a learned linear would be ideal,
    # but for the fallback case we just pad with zeros to match D_PROJ dim.
    # In practice you should always have the hidden cache after running ingest_clip_hidden.py.
    cls_norm = F.normalize(clip_cls, dim=-1)   # (B, 768)
    pad = torch.zeros(cls_norm.size(0), D_PROJ - cls_norm.size(1), device=cls_norm.device)
    return torch.cat([cls_norm, pad], dim=-1)  # (B, 1024) zero-padded


# ─────────────────────────────────────────────────────────────────────────────
# Retrieval on projector
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def retrieval_2way_on_text_projector(model, val_loader, device):
    """2-way retrieval measured on the text contrastive projector head."""
    model.eval()
    projs, tgts = [], []
    for b, ci, ct, d, v, ch in val_loader:
        b  = b.to(device)
        ct = ct.to(device)
        _, _, _, _, _, text_proj = model(b)
        valid = ct.norm(dim=-1) > 0.01
        if valid.any():
            projs.append(text_proj[valid].cpu())
            tgts.append(F.normalize(ct[valid], dim=-1).cpu())
    if not projs:
        return float("nan")
    return retrieval_2way(torch.cat(projs), torch.cat(tgts))


@torch.no_grad()
def retrieval_2way_on_projector(model, val_loader, device, has_hidden):
    """2-way retrieval measured on the image contrastive projector head."""
    model.eval()
    projs, tgts = [], []
    for b, ci, ct, d, v, ch in val_loader:
        b  = b.to(device)
        ci = ci.to(device)
        ch = ch.to(device)
        _, _, _, _, proj, _ = model(b)
        tgt = prepare_contrastive_target(ch, ci, has_hidden)
        projs.append(proj.cpu())
        tgts.append(tgt.cpu())
    projs = torch.cat(projs)
    tgts  = torch.cat(tgts)
    return retrieval_2way(projs, tgts)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_averaged_subject(root: Path, subj: str):
    avg_dir = averaged_meta_dir(root)
    betas  = np.load(avg_dir / f"betas_avg_perception_{subj}.npy")
    td     = np.load(avg_dir / f"targets_avg_{subj}.npz")
    clip_i = td["clip"]
    dino   = td["dino"]
    vae    = td["vae"]

    if "clip_text" in td.files:
        clip_t = td["clip_text"]
    else:
        global_ids  = np.load(avg_dir / "targets_text_image_id_73k.npy")
        global_text = np.load(avg_dir / "targets_text_clip.npy")
        order       = np.argsort(global_ids)
        sorted_ids  = global_ids[order]
        sorted_text = global_text[order]
        i73 = td["image_id_73k"]
        pos = np.searchsorted(sorted_ids, i73)
        matched = sorted_ids[pos] == i73
        clip_t = np.zeros((len(i73), 768), dtype=np.float32)
        clip_t[matched] = sorted_text[pos[matched]]

    # 257-token CLIP hidden layer — optional, falls back to CLS if absent
    hidden_path = avg_dir / f"targets_clip_hidden_{subj}.npy"
    if hidden_path.exists():
        clip_hidden = np.load(hidden_path)   # expected (N, 257, 1024)
        print(f"  [hidden] Loaded 257-token CLIP hidden layer: {clip_hidden.shape}")
        has_hidden = True
    else:
        print(
            f"  [hidden] {hidden_path} not found — "
            "falling back to 768-d CLS for contrastive target."
        )
        clip_hidden = None
        has_hidden  = False

    return betas, clip_i, clip_t, dino, vae, clip_hidden, has_hidden


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def run_training(subj: str = "subj01", root: str | Path = "/mnt/mindbridge", epochs: int = 150):
    root   = Path(root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_id   = resolve_run_id(f"4head_ctr_{epochs}ep_{subj}")
    run_base = run_root(root, run_id)
    ckpt_path = variant_ckpt_dir(root, VARIANT, subj, run_id)

    print(f"[4H+CTR] Subject: {subj} | Device: {device}")
    print(f"  run_id: {run_id} | epochs: {epochs}")
    print(f"  effective batch: {64 * ACCUM_STEPS} negatives (batch=64, accum={ACCUM_STEPS})")
    print(f"  schedule: BiMixCo for first {int(MIXUP_PCT*epochs)} ep → SoftCLIP")
    print(f"  temperature: {TEMPERATURE}  (sweep {{0.01,0.03,0.05,0.07}} if retrieval ≈50%)")

    betas, clip_i, clip_t, dino, vae, clip_hidden, has_hidden = load_averaged_subject(root, subj)
    n_images, n_voxels = betas.shape
    print(f"  Images: {n_images}  Voxels: {n_voxels}")
    print(f"  Contrastive target: {'257-token hidden (pooled)' if has_hidden else '768-d CLS fallback'}")

    np.random.seed(42)
    perm  = np.random.permutation(n_images)
    n_val = max(1, int(0.1 * n_images))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    tb, vb = betas[tr_idx], betas[val_idx]
    vm = tb.mean(0, keepdims=True)
    vs = tb.std(0, keepdims=True) + 1e-6
    tb = (tb - vm) / vs
    vb = (vb - vm) / vs

    ch_tr = clip_hidden[tr_idx] if has_hidden else None
    ch_val = clip_hidden[val_idx] if has_hidden else None

    nw = min(4, os.cpu_count() or 4)
    # Smaller batch; ACCUM_STEPS handles effective size
    BATCH = 64
    train_loader = DataLoader(
        AveragedNSDDataset(tb, clip_i[tr_idx], clip_t[tr_idx], dino[tr_idx], vae[tr_idx], ch_tr),
        batch_size=BATCH, shuffle=True, num_workers=nw, pin_memory=True, persistent_workers=nw > 0,
    )
    val_loader = DataLoader(
        AveragedNSDDataset(vb, clip_i[val_idx], clip_t[val_idx], dino[val_idx], vae[val_idx], ch_val),
        batch_size=BATCH, shuffle=False, num_workers=nw, pin_memory=True, persistent_workers=nw > 0,
    )

    model = MindBridgeContrastive(n_voxels=n_voxels).to(device)
    n_params   = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_proj     = sum(p.numel() for p in model.projector.parameters() if p.requires_grad)
    n_txt_proj = sum(p.numel() for p in model.text_projector.parameters() if p.requires_grad)
    print(f"  Total params: {n_params:,}  (img projector: {n_proj:,}  txt projector: {n_txt_proj:,})")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # ── early stopping state ──────────────────────────────────────────────────
    best_val_ret  = -1.0
    best_val_cosim = -1.0
    patience_ctr  = 0

    print(
        f"\n{'Ep':>5}|{'reg_loss':>9}|{'img_ctr':>8}|{'txt_ctr':>8}|{'cosim':>7}|"
        f"{'ret_img':>8}|{'ret_txt':>8}|{'phase':>8}|{'pat':>5}"
    )
    print("-" * 88)

    last_epoch = 0
    for epoch in range(1, epochs + 1):
        last_epoch = epoch
        phase = "BiMixCo" if epoch < int(MIXUP_PCT * epochs) else "SoftCLIP"
        model.train()

        total_reg  = 0.0
        total_ctr  = 0.0
        total_tctr = 0.0
        n_batches  = 0

        optimizer.zero_grad()
        for step, (b, ci, ct, d, v, ch) in enumerate(train_loader):
            b  = b.to(device,  non_blocking=True)
            ci = ci.to(device, non_blocking=True)
            ct = ct.to(device, non_blocking=True)
            d  = d.to(device,  non_blocking=True)
            v  = v.to(device,  non_blocking=True)
            ch = ch.to(device, non_blocking=True)

            pred_ci, pred_ct, pred_dino, pred_vae, proj, text_proj = model(b)

            # ── regression loss ───────────────────────────────────────────────
            reg_loss, _, _, _, _ = regression_loss(
                pred_ci, pred_ct, pred_dino, pred_vae, ci, ct, d, v
            )

            # ── image contrastive loss (primary retrieval) ────────────────────
            ctr_tgt  = prepare_contrastive_target(ch, ci, has_hidden)
            ctr_loss = contrastive_loss(proj, ctr_tgt, epoch, epochs, betas_for_mixco=b)

            # ── text contrastive loss (semantic/concept signal) ───────────────
            txt_ctr_loss = text_contrastive_loss(text_proj, ct)

            loss = (
                reg_loss
                + W_CONTRASTIVE      * ctr_loss
                + W_TEXT_CONTRASTIVE * txt_ctr_loss
            ) / ACCUM_STEPS
            loss.backward()

            total_reg  += reg_loss.item()
            total_ctr  += ctr_loss.item()
            total_tctr += txt_ctr_loss.item()
            n_batches  += 1

            if (step + 1) % ACCUM_STEPS == 0 or (step + 1) == len(train_loader):
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

        avg_reg  = total_reg  / n_batches
        avg_ctr  = total_ctr  / n_batches
        avg_tctr = total_tctr / n_batches

        # ── validation ────────────────────────────────────────────────────────
        model.eval()
        predict_clip_image = lambda m, b_: m(b_)[0]
        val_cosim   = compute_val_clip_cossim(model, val_loader, device, predict_clip_image)
        val_ret_img = retrieval_2way_on_projector(model, val_loader, device, has_hidden)
        val_ret_txt = retrieval_2way_on_text_projector(model, val_loader, device)

        payload = _base_payload(
            epoch, model, optimizer, vm, vs, n_voxels, subj, VARIANT, run_id, epochs,
            extra={
                "data": "averaged_perception",
                "val_retrieval_img_projector": val_ret_img,
                "val_retrieval_txt_projector": val_ret_txt,
                "contrastive_target": "hidden_pooled" if has_hidden else "cls_fallback",
            },
        )
        best_val_cosim = maybe_save_best_clip(ckpt_path, val_cosim, best_val_cosim, payload)

        # early stopping driven by image projector retrieval (primary metric)
        if val_ret_img > best_val_ret + MIN_DELTA:
            best_val_ret = val_ret_img
            patience_ctr = 0
            torch.save(payload, ckpt_path / "best_retrieval.pt")
        else:
            patience_ctr += 1

        scheduler.step()

        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            print(
                f"{epoch:5d}|{avg_reg:9.4f}|{avg_ctr:8.4f}|{avg_tctr:8.4f}|{val_cosim:7.4f}|"
                f"{val_ret_img:8.4f}|{val_ret_txt:8.4f}|{phase:>8}|{patience_ctr:>5}/{PATIENCE}"
            )

        if patience_ctr >= PATIENCE:
            print(
                f"\n  Early stop at epoch {epoch}: val img retrieval flat for "
                f"{PATIENCE} epochs (best={best_val_ret:.4f})"
            )
            if best_val_ret < 0.55:
                print(
                    "  !! val retrieval ≈ 50% despite contrastive head — likely causes:\n"
                    "     1. temperature too high/low — sweep {0.01, 0.03, 0.05, 0.07}\n"
                    "     2. L2-norm not applied before similarity — check projector\n"
                    "     3. effective batch too small — increase ACCUM_STEPS\n"
                    "     4. hidden-layer cache missing — check has_hidden flag\n"
                    "     5. data/label alignment issue in your beta files"
                )
            break

    # ── final checkpoint ──────────────────────────────────────────────────────
    final_path = save_final_checkpoint(
        ckpt_path,
        _base_payload(last_epoch, model, optimizer, vm, vs, n_voxels, subj, VARIANT, run_id, epochs),
        last_epoch,
    )

    # final eval on both heads
    model.eval()
    final_cosim   = compute_val_clip_cossim(model, val_loader, device, lambda m, b_: m(b_)[0])
    final_ret_img = retrieval_2way_on_projector(model, val_loader, device, has_hidden)
    final_ret_txt = retrieval_2way_on_text_projector(model, val_loader, device)

    print(f"\n[4H+CTR2] {subj} final results:")
    print(f"  Regression head     — val CLIP cosim:       {final_cosim:.4f}  (best={best_val_cosim:.4f})")
    print(f"  Image projector     — val 2-way retrieval:  {final_ret_img:.4f}  (best={best_val_ret:.4f})")
    print(f"  Text projector      — val 2-way retrieval:  {final_ret_txt:.4f}")
    print(f"  Checkpoints: {ckpt_path}")

    update_run_manifest(
        run_base,
        run_id=run_id,
        variant=VARIANT,
        subjects={
            subj: {
                "val_clip_cosim_final":          final_cosim,
                "val_clip_cosim_best":           best_val_cosim,
                "val_retrieval_img_proj_final":  final_ret_img,
                "val_retrieval_img_proj_best":   best_val_ret,
                "val_retrieval_txt_proj_final":  final_ret_txt,
                "contrastive_target":            "hidden_pooled" if has_hidden else "cls_fallback",
                "temperature":                   TEMPERATURE,
                "effective_batch":               64 * ACCUM_STEPS,
                "w_text_contrastive":            W_TEXT_CONTRASTIVE,
            }
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="MindBridge 4H + image contrastive + text contrastive heads")
    p.add_argument("--subj",               default=os.environ.get("NSD_SUBJ",        "subj01"))
    p.add_argument("--root",               default=os.environ.get("MINDBRIDGE_ROOT", "/mnt/mindbridge"))
    p.add_argument("--epochs",             type=int,   default=150)
    p.add_argument("--temperature",        type=float, default=TEMPERATURE,
                   help="InfoNCE/SoftCLIP temperature — sweep 0.01,0.03,0.05,0.07")
    p.add_argument("--accum-steps",        type=int,   default=ACCUM_STEPS,
                   help="gradient accumulation steps (effective_batch = 64 * accum_steps)")
    p.add_argument("--w-contrastive",      type=float, default=W_CONTRASTIVE,
                   help="weight of image contrastive loss")
    p.add_argument("--w-text-contrastive", type=float, default=W_TEXT_CONTRASTIVE,
                   help="weight of text contrastive loss (0 to disable)")
    args = p.parse_args()

    TEMPERATURE         = args.temperature
    ACCUM_STEPS         = args.accum_steps
    W_CONTRASTIVE       = args.w_contrastive
    W_TEXT_CONTRASTIVE  = args.w_text_contrastive

    run_training(subj=args.subj, root=args.root, epochs=args.epochs)