# -*- coding: utf-8 -*-
"""
Encode NSD stimuli with brain-diffuser VDVAE (imagenet64 EMA).

1. Encode all 73k images in nsd_stimuli.hdf5 → nsd_meta/averaged/vdvae_latents_73k.npy
2. Index by targets_avg image_id_73k → vdvae_latents_subj{01..08}.npy

Upstream VDVAE code is unmodified under vendor/brain-diffuser/vdvae/.

Usage:
    python vdvae_encoder_inference.py --root /mnt/mindbridge
    python vdvae_encoder_inference.py --subj subj01 --skip-global
"""

from __future__ import annotations

import argparse
import sys

sys.path = [p for p in sys.path if "share/software" not in p and "jupyterlab" not in p]

import h5py
import numpy as np
import torch
from pathlib import Path

from paths import get_root
from vdvae_brain_diffuser import (
    capture_ref_stats,
    download_vdvae_weights,
    encode_flat_latents,
    layer_flat_dims,
    load_ema_vae,
)

ALL_SUBJ = [f"subj{i:02d}" for i in range(1, 9)]


def encode_global_73k(root: Path, *, batch_size: int = 64) -> Path:
    out_path = root / "nsd_meta" / "averaged" / "vdvae_latents_73k.npy"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        arr = np.load(out_path, mmap_mode="r")
        print(f"  Global latents exist: {out_path} shape={arr.shape} dtype={arr.dtype}")
        return out_path

    h5_path = root / "nsd_meta" / "nsd_stimuli.hdf5"
    if not h5_path.exists():
        raise FileNotFoundError(f"Missing {h5_path}")

    download_vdvae_weights(root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("VDVAE encoding requires CUDA (brain-diffuser preprocess uses .cuda()).")

    ema_vae, preprocess_fn, _H = load_ema_vae(root, device)
    ref_stats = capture_ref_stats(ema_vae, preprocess_fn, device)
    flat_dims = layer_flat_dims(ref_stats)
    ref_path = root / "vdvae" / "ref_stats.npz"
    ref_path.parent.mkdir(parents=True, exist_ok=True)
    shapes = np.array([s["z"].shape for s in ref_stats], dtype=object)
    np.savez(ref_path, layer_shapes=shapes, layer_flat_dims=flat_dims)
    print(f"  VDVAE latent dim: {int(flat_dims.sum())} ({NUM_LATENT_LAYERS} layers)")

    print(f"  Encoding 73k NSD images from {h5_path} ...")
    with h5py.File(h5_path, "r") as f:
        img_brick = f["imgBrick"]
        n = img_brick.shape[0]
        chunks: list[np.ndarray] = []
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch = img_brick[start:end].astype(np.uint8)
            lat = encode_flat_latents(
                ema_vae, preprocess_fn, batch, batch_size=batch_size, device=device,
            )
            chunks.append(lat.astype(np.float16))
            print(f"    {end}/{n}")

    latents = np.concatenate(chunks, axis=0)
    np.save(out_path, latents)
    print(f"  Saved {out_path} shape={latents.shape} dtype={latents.dtype}")
    return out_path


def slice_subject_latents(root: Path, subj: str, global_path: Path | None = None) -> Path:
    global_path = global_path or root / "nsd_meta" / "averaged" / "vdvae_latents_73k.npy"
    if not global_path.exists():
        raise FileNotFoundError(f"Run global encode first: {global_path}")

    td_path = root / "nsd_meta" / "averaged" / f"targets_avg_{subj}.npz"
    if not td_path.exists():
        raise FileNotFoundError(f"Missing {td_path}")

    latents_73k = np.load(global_path, mmap_mode="r")
    td = np.load(td_path)
    ids = td["image_id_73k"].astype(np.int64)
    subj_lat = np.array(latents_73k[ids], dtype=np.float16)

    out_path = root / "nsd_meta" / "averaged" / f"vdvae_latents_{subj}.npy"
    np.save(out_path, subj_lat)
    print(f"  [{subj}] {out_path} shape={subj_lat.shape} (n_unique={len(ids)})")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="VDVAE latent extraction for NSD (brain-diffuser).")
    ap.add_argument("--root", type=str, default=None)
    ap.add_argument("--subj", type=str, default="subj01")
    ap.add_argument("--all", action="store_true", help="Slice latents for all 8 subjects.")
    ap.add_argument("--skip-global", action="store_true", help="Only slice per-subject from cached 73k.")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    root = get_root(args.root)
    global_path = None
    if not args.skip_global:
        global_path = encode_global_73k(root, batch_size=args.batch_size)

    subjects = ALL_SUBJ if args.all else [args.subj]
    for subj in subjects:
        slice_subject_latents(root, subj, global_path)


if __name__ == "__main__":
    main()
