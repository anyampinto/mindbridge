# -*- coding: utf-8 -*-
"""MindBridge Stage 1 Training — Modal-compatible"""

import os
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# =============================================================================
# DATASET & MODEL  (module-level classes — no side effects, safe to import)
# =============================================================================
class NSDDataset(Dataset):
    def __init__(self, betas, clip_t, dino_t, vae_t):
        self.betas = torch.tensor(betas,  dtype=torch.float32)
        self.clip  = torch.tensor(clip_t, dtype=torch.float32)
        self.dino  = torch.tensor(dino_t, dtype=torch.float32)
        self.vae   = torch.tensor(vae_t,  dtype=torch.float32)

    def __len__(self):
        return len(self.betas)

    def __getitem__(self, idx):
        return self.betas[idx], self.clip[idx], self.dino[idx], self.vae[idx]


class BrainMLP(nn.Module):
    def __init__(self, n_voxels, d_model=512, M=16, dropout=0.5):
        super().__init__()
        self.M = M
        self.d_model = d_model
        self.net = nn.Sequential(
            nn.Linear(n_voxels, 4096), nn.LayerNorm(4096), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(4096, 4096),     nn.LayerNorm(4096), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(4096, M * d_model),
        )

    def forward(self, x):
        return self.net(x).view(x.size(0), self.M, self.d_model)


class LatentQueryDecoder(nn.Module):
    def __init__(self, d_model=512, M=16, n_heads=8, K=2, dropout=0.1):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(1, M, d_model) * 0.02)
        self.layers  = nn.ModuleList([
            nn.ModuleDict({
                "self_attn":  nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True),
                "cross_attn": nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True),
                "ffn":  nn.Sequential(nn.Linear(d_model, d_model * 4), nn.GELU(),
                                      nn.Linear(d_model * 4, d_model)),
                "norm1": nn.LayerNorm(d_model),
                "norm2": nn.LayerNorm(d_model),
                "norm3": nn.LayerNorm(d_model),
            })
            for _ in range(K)
        ])

    def forward(self, brain_tokens):
        B = brain_tokens.size(0)
        q = self.queries.expand(B, -1, -1)
        for layer in self.layers:
            q2, _ = layer["self_attn"](q, q, q);              q = layer["norm1"](q + q2)
            q2, _ = layer["cross_attn"](q, brain_tokens, brain_tokens); q = layer["norm2"](q + q2)
            q     = layer["norm3"](q + layer["ffn"](q))
        return q


class MindBridgeMLP(nn.Module):
    def __init__(self, n_voxels, d_model=512, M=16,
                 d_clip=768, d_dino=1024, vae_shape=(4, 64, 64), dropout=0.5):
        super().__init__()
        self.encoder   = BrainMLP(n_voxels, d_model, M, dropout)
        self.decoder   = LatentQueryDecoder(d_model, M, n_heads=8, K=2)
        self.clip_head = nn.Linear(d_model, d_clip)
        self.dino_head = nn.Linear(d_model, d_dino)
        self.vae_head  = nn.Sequential(nn.Linear(d_model * M, vae_shape[0] * vae_shape[1] * vae_shape[2]))
        self.vae_shape = vae_shape
        self.M         = M

    def forward(self, x):
        brain_tokens  = self.encoder(x)
        visual_tokens = self.decoder(brain_tokens)
        pooled        = visual_tokens.mean(dim=1)
        pred_clip     = self.clip_head(pooled)
        pred_dino     = self.dino_head(pooled)
        flat          = visual_tokens.reshape(visual_tokens.size(0), -1)
        pred_vae      = self.vae_head(flat).view(-1, *self.vae_shape)
        return pred_clip, pred_dino, pred_vae


def mindbridge_loss(pred_clip, pred_dino, pred_vae, tgt_clip, tgt_dino, tgt_vae,
                    w_clip=1.0, w_dino=0.5, w_vae=0.5):
    loss_clip = (1 - F.cosine_similarity(pred_clip, tgt_clip, dim=-1)).mean()
    loss_dino = F.mse_loss(pred_dino, tgt_dino)
    loss_vae  = F.mse_loss(pred_vae,  tgt_vae)
    total     = w_clip * loss_clip + w_dino * loss_dino + w_vae * loss_vae
    return total, loss_clip.item(), loss_dino.item(), loss_vae.item()


# =============================================================================
# TRAINING ENTRY POINT
# =============================================================================
def run_training(subj: str = "subj01", root: Path = Path("/mnt/mindbridge")):
    """
    Full Stage 1 training run for one subject.
    Resume-safe: saves a resume checkpoint every 10 epochs so a disconnect
    never loses more than 10 epochs of progress.
    """

    # ── Paths ─────────────────────────────────────────────────────────────────
    meta_dir  = root / "nsd_meta"
    ckpt_path = root / "checkpoints" / subj
    log_path  = root / "logs" / subj
    for p in [ckpt_path, log_path]:
        p.mkdir(parents=True, exist_ok=True)

    betas_file   = meta_dir / f"betas_flat_{subj}.npy"
    targets_file = meta_dir / f"targets_full_{subj}.npz"
    resume_ckpt  = ckpt_path / "resume_stage1.pt"   # overwritten every 10 epochs
    best_ckpt    = ckpt_path / "best_stage1.pt"

    assert betas_file.exists(),   f"Betas not found: {betas_file}. Run ingestion first."
    assert targets_file.exists(), f"Targets not found: {targets_file}. Run ingestion first."

    # ── Device ────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training subject: {subj}")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Load data ─────────────────────────────────────────────────────────────
    print(f"\nLoading betas from {betas_file.name}...")
    betas_flat = np.load(betas_file)
    n_trials, n_voxels = betas_flat.shape
    print(f"  Betas: {betas_flat.shape}")

    print(f"Loading targets from {targets_file.name}...")
    td           = np.load(targets_file)
    clip_targets = td["clip"]   # (30000, 768)
    dino_targets = td["dino"]   # (30000, 1024)
    vae_targets  = td["vae"]    # (30000, 4, 64, 64)
    print(f"  CLIP {clip_targets.shape}, DINO {dino_targets.shape}, VAE {vae_targets.shape}")

    D_CLIP, D_DINO, D_VAE = 768, 1024, (4, 64, 64)
    assert dino_targets.shape[1] == D_DINO, \
        f"D_DINO mismatch: got {dino_targets.shape[1]}, expected {D_DINO}"

    # ── Normalize betas ───────────────────────────────────────────────────────
    N_TRAIN     = 27000
    train_betas = betas_flat[:N_TRAIN]
    val_betas   = betas_flat[N_TRAIN:]

    voxel_mean  = train_betas.mean(axis=0, keepdims=True)
    voxel_std   = train_betas.std(axis=0,  keepdims=True) + 1e-6
    train_betas = (train_betas - voxel_mean) / voxel_std
    val_betas   = (val_betas   - voxel_mean) / voxel_std

    train_clip = clip_targets[:N_TRAIN];  val_clip = clip_targets[N_TRAIN:]
    train_dino = dino_targets[:N_TRAIN];  val_dino = dino_targets[N_TRAIN:]
    train_vae  = vae_targets[:N_TRAIN];   val_vae  = vae_targets[N_TRAIN:]
    print(f"\nTrain: {train_betas.shape}, Val: {val_betas.shape}")

    # ── DataLoaders ───────────────────────────────────────────────────────────
    num_workers  = min(4, os.cpu_count() or 4)
    train_loader = DataLoader(NSDDataset(train_betas, train_clip, train_dino, train_vae),
                              batch_size=32, shuffle=True,
                              num_workers=num_workers, pin_memory=True, persistent_workers=True)
    val_loader   = DataLoader(NSDDataset(val_betas, val_clip, val_dino, val_vae),
                              batch_size=32, shuffle=False,
                              num_workers=num_workers, pin_memory=True, persistent_workers=True)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # ── Model ─────────────────────────────────────────────────────────────────
    EPOCHS = 150
    LR     = 1e-4

    model = MindBridgeMLP(n_voxels=n_voxels, d_model=256, M=16,
                          d_clip=D_CLIP, d_dino=D_DINO, vae_shape=D_VAE,
                          dropout=0.5).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_loss = float("inf")
    history       = {"train": [], "val": []}
    start_epoch   = 1

    # ── Resume if a checkpoint exists ─────────────────────────────────────────
    if resume_ckpt.exists():
        print(f"\nResume checkpoint found — loading {resume_ckpt.name}...")
        ckpt = torch.load(resume_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        best_val_loss = ckpt["best_val_loss"]
        history       = ckpt["history"]
        start_epoch   = ckpt["epoch"] + 1
        print(f"  Resuming from epoch {start_epoch}/{EPOCHS} "
              f"(best val loss so far: {best_val_loss:.4f})")
    else:
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\nModel parameters: {n_params:,}")
        print("No resume checkpoint found — starting from scratch.")

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f"\n=== Stage 1: Perception Pretraining ({subj}) ===")
    for epoch in range(start_epoch, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for betas, clip_t, dino_t, vae_t in train_loader:
            betas, clip_t, dino_t, vae_t = (
                betas.to(device, non_blocking=True), clip_t.to(device, non_blocking=True),
                dino_t.to(device, non_blocking=True), vae_t.to(device, non_blocking=True))
            optimizer.zero_grad()
            loss, *_ = mindbridge_loss(*model(betas), clip_t, dino_t, vae_t)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for betas, clip_t, dino_t, vae_t in val_loader:
                betas, clip_t, dino_t, vae_t = (
                    betas.to(device, non_blocking=True), clip_t.to(device, non_blocking=True),
                    dino_t.to(device, non_blocking=True), vae_t.to(device, non_blocking=True))
                loss, *_ = mindbridge_loss(*model(betas), clip_t, dino_t, vae_t)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        scheduler.step()
        history["train"].append(train_loss)
        history["val"].append(val_loss)

        # Best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch, "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "voxel_mean": voxel_mean, "voxel_std": voxel_std,
                "n_voxels": n_voxels, "subj": subj,
            }, best_ckpt)

        # Resume checkpoint — saved every 10 epochs, always reflects latest state
        if epoch % 10 == 0:
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_val_loss": best_val_loss,
                "history": history,
                "voxel_mean": voxel_mean,
                "voxel_std": voxel_std,
                "n_voxels": n_voxels,
                "subj": subj,
            }, resume_ckpt)
            print(f"Epoch {epoch:3d}/{EPOCHS} | "
                  f"train={train_loss:.4f} | val={val_loss:.4f} | best={best_val_loss:.4f} "
                  f"[resume checkpoint saved]")
        elif epoch % 10 == 0 or epoch == EPOCHS:
            print(f"Epoch {epoch:3d}/{EPOCHS} | "
                  f"train={train_loss:.4f} | val={val_loss:.4f} | best={best_val_loss:.4f}")

    print("Stage 1 complete.")

    # ── Loss curve ────────────────────────────────────────────────────────────
    plt.figure(figsize=(8, 4))
    plt.plot(history["train"], label="train")
    plt.plot(history["val"],   label="val")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title(f"Stage 1 Training Curve — {subj}")
    plt.legend(); plt.tight_layout()
    plt.savefig(log_path / "stage1_loss_curve.png", dpi=150)
    plt.close()
    print(f"Loss curve saved to {log_path / 'stage1_loss_curve.png'}")

    # ── Quick eval ────────────────────────────────────────────────────────────
    model.eval()
    clip_sims = []
    with torch.no_grad():
        for betas, clip_t, *_ in val_loader:
            betas  = betas.to(device, non_blocking=True)
            clip_t = clip_t.to(device, non_blocking=True)
            pred_clip, *_ = model(betas)
            sim = F.cosine_similarity(
                F.normalize(pred_clip, dim=-1),
                F.normalize(clip_t,    dim=-1), dim=-1)
            clip_sims.extend(sim.cpu().tolist())

    print(f"\nVal CLIP cosine similarity: {np.mean(clip_sims):.4f}")
    print("(Random baseline ≈ 0.0, good model ≈ 0.3–0.6 on NSD perception)")
    print(f"Checkpoints saved to: {ckpt_path}")
    print("Done.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--subj", default="subj01")
    parser.add_argument("--root", default="/mnt/mindbridge")
    args = parser.parse_args()
    run_training(subj=args.subj, root=Path(args.root))
