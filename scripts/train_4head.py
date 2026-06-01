# -*- coding: utf-8 -*-
"""
4-head MindBridge on trial-averaged perception data (nsd_meta/averaged/).

Heads: CLIP image, CLIP text, DINO, VAE.
Loss weights: CLIP image=1.0, CLIP text=0.3, DINO=0.1, VAE=0.001 (all cosine or MSE on heads).
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

from paths import averaged_meta_dir, resolve_run_id, run_root, update_run_manifest, variant_ckpt_dir
from training_checkpoints import (
    _base_payload,
    compute_val_clip_cossim,
    maybe_save_best_clip,
    retrieval_2way,
    save_final_checkpoint,
)

VARIANT = "4H"

W_CLIP_IMAGE = 1.0
W_CLIP_TEXT = 0.3
W_DINO = 0.1
W_VAE = 0.001

# Stop 150-epoch runs if val CLIP cosim flatlines (still saves best_clip.pt).
STAGNATION_PATIENCE = 40
STAGNATION_MIN_DELTA = 1e-4


class AveragedNSDDataset(Dataset):
    def __init__(self, betas, clip_i, clip_t, dino, vae):
        self.betas = torch.tensor(betas, dtype=torch.float32)
        self.clip_i = torch.tensor(clip_i, dtype=torch.float32)
        self.clip_t = torch.tensor(clip_t, dtype=torch.float32)
        self.dino = torch.tensor(dino, dtype=torch.float32)
        self.vae = torch.tensor(vae, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.betas)

    def __getitem__(self, idx):
        return self.betas[idx], self.clip_i[idx], self.clip_t[idx], self.dino[idx], self.vae[idx]


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


class MindBridge4Head(nn.Module):
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
        self.encoder = BrainMLP(n_voxels, d_model, M, dropout)
        self.decoder = LatentQueryDecoder(d_model, M)
        self.clip_image_head = nn.Linear(d_model, d_clip)
        self.clip_text_head = nn.Linear(d_model, d_clip)
        self.dino_head = nn.Linear(d_model, d_dino)
        self.vae_head = nn.Linear(d_model * M, vae_shape[0] * vae_shape[1] * vae_shape[2])
        self.vae_shape = vae_shape
        self.M = M

    def forward(self, x):
        brain_tokens = self.encoder(x)
        visual_tokens = self.decoder(brain_tokens)
        pooled = visual_tokens.mean(dim=1)
        flat = visual_tokens.reshape(x.size(0), -1)
        return (
            self.clip_image_head(pooled),
            self.clip_text_head(pooled),
            self.dino_head(pooled),
            self.vae_head(flat).view(-1, *self.vae_shape),
        )


def loss_fn(pred_ci, pred_ct, pred_dino, pred_vae, tgt_ci, tgt_ct, tgt_dino, tgt_vae):
    """CLIP text loss is masked where caption embedding is zero (no COCO captions)."""
    lc = (1 - F.cosine_similarity(pred_ci, tgt_ci, dim=-1)).mean()

    text_mask = tgt_ct.norm(dim=-1) > 0.01
    if text_mask.any():
        lt = (1 - F.cosine_similarity(pred_ct[text_mask], tgt_ct[text_mask], dim=-1)).mean()
    else:
        lt = torch.zeros((), device=pred_ct.device, dtype=pred_ct.dtype)

    ld = F.mse_loss(pred_dino, tgt_dino)
    lv = F.mse_loss(pred_vae, tgt_vae)
    w_lc = W_CLIP_IMAGE * lc
    w_lt = W_CLIP_TEXT * lt
    w_ld = W_DINO * ld
    w_lv = W_VAE * lv
    total = w_lc + w_lt + w_ld + w_lv
    return (
        total,
        lc.item(),
        lt.item(),
        ld.item(),
        lv.item(),
        w_lc.item(),
        w_lt.item(),
        w_ld.item(),
        w_lv.item(),
        int(text_mask.sum().item()),
    )


def load_averaged_subject(root: Path, subj: str):
    avg_dir = averaged_meta_dir(root)
    betas = np.load(avg_dir / f"betas_avg_perception_{subj}.npy")
    td = np.load(avg_dir / f"targets_avg_{subj}.npz")
    clip_i = td["clip"]
    dino = td["dino"]
    vae = td["vae"]
    if "clip_text" in td.files:
        clip_t = td["clip_text"]
    else:
        global_ids = np.load(avg_dir / "targets_text_image_id_73k.npy")
        global_text = np.load(avg_dir / "targets_text_clip.npy")
        order = np.argsort(global_ids)
        sorted_ids = global_ids[order]
        sorted_text = global_text[order]
        i73 = td["image_id_73k"]
        pos = np.searchsorted(sorted_ids, i73)
        matched = sorted_ids[pos] == i73
        clip_t = np.zeros((len(i73), 768), dtype=np.float32)
        clip_t[matched] = sorted_text[pos[matched]]
    return betas, clip_i, clip_t, dino, vae


def run_training(subj: str = "subj01", root: str | Path = "/mnt/mindbridge", epochs: int = 150):
    root = Path(root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_id = resolve_run_id(f"4head_{epochs}ep_{subj}")
    run_base = run_root(root, run_id)
    ckpt_path = variant_ckpt_dir(root, VARIANT, subj, run_id)

    print(f"[4-head] Subject: {subj} | Device: {device}")
    print(f"  run_id: {run_id}")
    print(f"  data:   {averaged_meta_dir(root)}")

    betas, clip_i, clip_t, dino, vae = load_averaged_subject(root, subj)
    n_images, n_voxels = betas.shape
    n_text_valid = int((np.linalg.norm(clip_t, axis=1) > 0.01).sum())
    print(f"  Images: {n_images}  Voxels: {n_voxels}")
    print(
        f"  clip_text valid: {n_text_valid}/{n_images} "
        f"({100.0 * n_text_valid / n_images:.1f}%) — zero rows excluded from text loss"
    )

    # Each row is one unique image; 90/10 holdout by image index (not trial-level).
    np.random.seed(42)
    perm = np.random.permutation(n_images)
    n_val = max(1, int(0.1 * n_images))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    print(f"  Val split: {len(tr_idx)} train / {len(val_idx)} val images (seed=42)")

    tb, vb = betas[tr_idx], betas[val_idx]
    vm = tb.mean(0, keepdims=True)
    vs = tb.std(0, keepdims=True) + 1e-6
    tb = (tb - vm) / vs
    vb = (vb - vm) / vs

    nw = min(4, os.cpu_count() or 4)
    train_loader = DataLoader(
        AveragedNSDDataset(tb, clip_i[tr_idx], clip_t[tr_idx], dino[tr_idx], vae[tr_idx]),
        batch_size=128,
        shuffle=True,
        num_workers=nw,
        pin_memory=True,
        persistent_workers=nw > 0,
    )
    val_loader = DataLoader(
        AveragedNSDDataset(vb, clip_i[val_idx], clip_t[val_idx], dino[val_idx], vae[val_idx]),
        batch_size=128,
        shuffle=False,
        num_workers=nw,
        pin_memory=True,
        persistent_workers=nw > 0,
    )

    model = MindBridge4Head(n_voxels=n_voxels).to(device)
    print(f"  Params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_clip_cosim = -1.0
    stagnation_epochs = 0
    predict_clip_image = lambda m, b: m(b)[0]

    log_every_epoch = epochs <= 10
    print(
        f"{'Ep':>5}|{'train':>8}|{'w_img':>8}|{'w_txt':>8}|{'w_dino':>8}|{'w_vae':>10}|"
        f"{'cossim':>8}|{'ret2w':>8}"
    )
    print("-" * 88)

    last_epoch = 0
    for epoch in range(1, epochs + 1):
        last_epoch = epoch
        model.train()
        tl = wci = wct = wd = wv = 0.0
        n_batches = 0
        for b, ci, ct, d, v in train_loader:
            b = b.to(device, non_blocking=True)
            ci = ci.to(device, non_blocking=True)
            ct = ct.to(device, non_blocking=True)
            d = d.to(device, non_blocking=True)
            v = v.to(device, non_blocking=True)
            optimizer.zero_grad()
            pci, pct, pd, pv = model(b)
            loss, _, _, _, _, bw_ci, bw_ct, bw_d, bw_v, _ = loss_fn(
                pci, pct, pd, pv, ci, ct, d, v,
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite train loss at epoch {epoch}")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tl += loss.item()
            wci += bw_ci
            wct += bw_ct
            wd += bw_d
            wv += bw_v
            n_batches += 1
        tl /= n_batches
        wci /= n_batches
        wct /= n_batches
        wd /= n_batches
        wv /= n_batches

        model.eval()
        vl = wci_v = wct_v = wd_v = wv_v = 0.0
        val_preds, val_tgts = [], []
        n_val_batches = 0
        with torch.no_grad():
            for b, ci, ct, d, v in val_loader:
                b = b.to(device, non_blocking=True)
                ci = ci.to(device, non_blocking=True)
                ct = ct.to(device, non_blocking=True)
                d = d.to(device, non_blocking=True)
                v = v.to(device, non_blocking=True)
                pci, pct, pd, pv = model(b)
                tot, _, _, _, _, bw_ci, bw_ct, bw_d, bw_v, _ = loss_fn(
                    pci, pct, pd, pv, ci, ct, d, v,
                )
                if not torch.isfinite(tot):
                    raise RuntimeError(f"Non-finite val loss at epoch {epoch}")
                vl += tot.item()
                wci_v += bw_ci
                wct_v += bw_ct
                wd_v += bw_d
                wv_v += bw_v
                val_preds.append(pci.cpu())
                val_tgts.append(ci.cpu())
                n_val_batches += 1
        vl /= n_val_batches
        wci_v /= n_val_batches
        wct_v /= n_val_batches
        wd_v /= n_val_batches
        wv_v /= n_val_batches

        val_clip_cosim = compute_val_clip_cossim(model, val_loader, device, predict_clip_image)
        val_ret = retrieval_2way(torch.cat(val_preds), torch.cat(val_tgts))

        prev_best = best_clip_cosim
        payload = _base_payload(
            epoch, model, optimizer, vm, vs, n_voxels, subj, VARIANT, run_id, epochs,
            extra={"data": "averaged_perception"},
        )
        best_clip_cosim = maybe_save_best_clip(ckpt_path, val_clip_cosim, best_clip_cosim, payload)
        if val_clip_cosim > prev_best + STAGNATION_MIN_DELTA:
            stagnation_epochs = 0
        else:
            stagnation_epochs += 1

        scheduler.step()

        if log_every_epoch or epoch % 10 == 0 or epoch == epochs:
            print(
                f"{epoch:5d}|{tl:8.4f}|{wci_v:8.4f}|{wct_v:8.4f}|{wd_v:8.4f}|{wv_v:10.4f}|"
                f"{val_clip_cosim:8.4f}|{val_ret:8.4f}"
            )
            if log_every_epoch:
                print(
                    f"       train weighted: img={wci:.4f} txt={wct:.4f} "
                    f"dino={wd:.4f} vae={wv:.4f}  (dominant: "
                    f"{max(('img', wci), ('txt', wct), ('dino', wd), ('vae', wv), key=lambda x: x[1])[0]})"
                )

        if epochs >= 50 and stagnation_epochs >= STAGNATION_PATIENCE:
            print(
                f"\n  Early stop epoch {epoch}: val CLIP cosim flat for "
                f"{STAGNATION_PATIENCE} epochs (best={best_clip_cosim:.4f})"
            )
            break

    final_path = save_final_checkpoint(
        ckpt_path,
        _base_payload(
            last_epoch, model, optimizer, vm, vs, n_voxels, subj, VARIANT, run_id, epochs,
        ),
        last_epoch,
    )
    final_clip = compute_val_clip_cossim(model, val_loader, device, predict_clip_image)
    print(f"\n[4-head] {subj} final val CLIP image CosSim: {final_clip:.4f}")
    print(f"  best_clip.pt: {ckpt_path / 'best_clip.pt'} (best={best_clip_cosim:.4f})")
    print(f"  final.pt:     {final_path}")
    update_run_manifest(
        run_base,
        run_id=run_id,
        variant=VARIANT,
        subjects={
            subj: {
                "val_clip_cosim_final": final_clip,
                "val_clip_cosim_best": best_clip_cosim,
            }
        },
    )


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--subj", default=os.environ.get("NSD_SUBJ", "subj01"))
    p.add_argument("--root", default=os.environ.get("MINDBRIDGE_ROOT", "/mnt/mindbridge"))
    p.add_argument("--epochs", type=int, default=150)
    args = p.parse_args()
    run_training(subj=args.subj, root=args.root, epochs=args.epochs)
