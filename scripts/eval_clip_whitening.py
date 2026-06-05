# eval_clip_whitening.py

import os
from pathlib import Path
import argparse

import numpy as np
import torch
import torch.nn.functional as F


# -----------------------------
# ARGS
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--subj", default=os.environ.get("NSD_SUBJ", "subj01"))
    p.add_argument("--root", default=os.environ.get("MINDBRIDGE_ROOT", "/mnt/mindbridge"))
    p.add_argument("--variant", default="A")
    p.add_argument("--run-id", default=None)
    p.add_argument("--n-pairs", type=int, default=10000)
    return p.parse_args()


# -----------------------------
# NORMALIZATION / ZCA
# -----------------------------
def l2norm(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


def fit_zca(x, eps=1e-5):
    mean = x.mean(axis=0, keepdims=True)
    xc = x - mean
    cov = (xc.T @ xc) / max(len(xc) - 1, 1)

    u, s, _ = np.linalg.svd(cov)
    s = np.clip(s, eps, None)

    w = u @ np.diag(1.0 / np.sqrt(s)) @ u.T
    return mean.astype(np.float32), w.astype(np.float32)


def apply_zca(x, mean, w):
    return (x - mean) @ w


# -----------------------------
# METRICS
# -----------------------------
def mean_cos(pred, tgt):
    pred = F.normalize(pred.float(), dim=-1)
    tgt = F.normalize(tgt.float(), dim=-1)
    return float((pred * tgt).sum(dim=-1).mean().item())


def offdiag(tgt, max_pairs=5000, seed=0):
    tgt = F.normalize(tgt.float(), dim=-1)
    n = tgt.size(0)

    rng = np.random.default_rng(seed)
    i = rng.integers(0, n, max_pairs)
    j = rng.integers(0, n, max_pairs)
    mask = i != j

    i, j = i[mask], j[mask]
    return float((tgt[i] * tgt[j]).sum(dim=-1).mean().item())


# -----------------------------
# LOAD DEBUG-SAFE
# -----------------------------
def unpack_averaged(out):
    """
    Your current format:
    (7,)
    0 betas
    1 clip_img
    2 clip_text
    3 dino
    4 vae
    5 clip_hidden
    6 flag
    """

    print("\n=== DEBUG: load_averaged_subject ===")
    print("type:", type(out))

    if not isinstance(out, tuple):
        raise ValueError("Expected tuple from load_averaged_subject")

    print("num items:", len(out))

    for i, x in enumerate(out):
        if hasattr(x, "shape"):
            print(f"[{i}] shape={x.shape}")
        else:
            print(f"[{i}] type={type(x)} value={x}")

    if len(out) < 6:
        raise ValueError("Unexpected tuple size")

    betas = out[0]
    clip_img = out[1]
    clip_text = out[2]
    dino = out[3]
    vae = out[4]
    clip_hidden = out[5]
    flag = out[6] if len(out) > 6 else None

    return betas, clip_img, clip_text, dino, vae, clip_hidden, flag


# -----------------------------
# MAIN
# -----------------------------
def main():
    args = parse_args()
    root = Path(args.root)

    print(f"\n[whitening] subj={args.subj} variant={args.variant}")

    # IMPORT INSIDE (avoids modal weirdness)
    from train_dual_contrastive import load_averaged_subject

    out = load_averaged_subject(root, args.subj)

    betas, clip_img, clip_text, dino, vae, clip_hidden, flag = unpack_averaged(out)

    print("\n=== FINAL SHAPES ===")
    print("betas:", betas.shape)
    print("clip_img:", clip_img.shape)
    print("clip_text:", clip_text.shape)
    print("dino:", dino.shape)
    print("vae:", vae.shape)
    print("clip_hidden:", clip_hidden.shape)
    print("flag:", flag)

    # -----------------------------
    # BASIC WHITENING TEST
    # -----------------------------
    print("\n[whitening] running sanity ZCA on CLIP image...")

    x = l2norm(clip_img)
    mean, w = fit_zca(x)
    xw = apply_zca(x, mean, w)

    xw = torch.tensor(xw)
    x = torch.tensor(x)

    print("cos(raw):", mean_cos(x, x))
    print("offdiag(raw):", offdiag(x))


if __name__ == "__main__":
    main()