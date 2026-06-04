# -*- coding: utf-8 -*-
"""Write layer_flat_dims into vdvae/ref_stats.npz (quick GPU forward)."""

from __future__ import annotations

import argparse
import sys

sys.path = [p for p in sys.path if "share/software" not in p and "jupyterlab" not in p]

import numpy as np
import torch
from pathlib import Path

from paths import get_root
from vdvae_brain_diffuser import (
    NUM_LATENT_LAYERS,
    capture_ref_stats,
    download_vdvae_weights,
    layer_flat_dims,
    load_ema_vae,
)


def write_ref_dims(root: Path) -> Path:
    download_vdvae_weights(root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("VDVAE ref dims requires CUDA")
    ema_vae, preprocess_fn, _H = load_ema_vae(root, device)
    ref_stats = capture_ref_stats(ema_vae, preprocess_fn, device)
    flat_dims = layer_flat_dims(ref_stats)
    ref_path = root / "vdvae" / "ref_stats.npz"
    ref_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(ref_path, layer_flat_dims=flat_dims.astype(np.int64))
    print(f"  Saved {ref_path}  layers={NUM_LATENT_LAYERS}  dim={int(flat_dims.sum())}")
    return ref_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=None)
    args = ap.parse_args()
    write_ref_dims(get_root(args.root))


if __name__ == "__main__":
    main()
