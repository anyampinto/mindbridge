# -*- coding: utf-8 -*-
"""
Compute ViT-L/14 patch+CLS hidden states (257×1024) for averaged perception images.

Saves the RAW last_hidden_state before visual_projection — this is the correct
257-token representation used in MindEye for contrastive retrieval.

  ViT-L/14 internal dim = 1024  (not 768 — that's the projected CLS output)
  Shape written: (N, 257, 1024) float16

The training script pools these to (N, 1024) then projects to D_PROJ=768
via the ContrastiveProjector head. Update D_HIDDEN_TOKEN=1024 and
prepare_contrastive_target to match (see notes below).

Writes: averaged_meta_dir / targets_clip_hidden_{subj}.npy
  (filename matches what train_contrastive_retrieval.py expects)

Usage:
  python ingest_clip_hidden.py --subj subj01 --root /mnt/mindbridge
  python ingest_clip_hidden.py --subj all    --root /mnt/mindbridge
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path = [p for p in sys.path if "share/software" not in p and "jupyterlab" not in p]

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image

from paths import averaged_meta_dir, get_root

ALL_SUBJ = [f"subj{i:02d}" for i in range(1, 9)]

# ViT-L/14 internal hidden dim — NOT 768 (that's post-projection CLS only)
N_TOKENS    = 257
D_HIDDEN    = 1024   # raw last_hidden_state dim for ViT-L/14


def clip_hidden_path(out_dir: Path, subj: str) -> Path:
    # matches the filename train_contrastive_retrieval.py looks for
    return out_dir / f"targets_clip_hidden_{subj}.npy"


@torch.no_grad()
def encode_hidden_batch(
    vision_model,
    processor,
    images: list[Image.Image],
    device: torch.device,
) -> np.ndarray:
    """
    Returns raw last_hidden_state (B, 257, 1024), L2-normalised per token, float16.

    We do NOT apply visual_projection here. That projection is a single linear
    layer trained to map the CLS token to CLIP's 768-d embedding space — applying
    it token-wise to all 257 tokens is non-standard and loses the spatial structure
    that makes hidden-layer targets superior to CLS (per MindEye ablations).
    """
    inputs = processor(images=images, return_tensors="pt")
    pixel  = inputs["pixel_values"].to(device)

    out    = vision_model(pixel_values=pixel, output_hidden_states=False)
    hidden = out.last_hidden_state           # (B, 257, 1024)
    hidden = F.normalize(hidden, dim=-1)     # L2-norm per token
    return hidden.cpu().numpy().astype(np.float16)


def ingest_subj(subj: str, root: Path, batch_size: int = 32, device: str = "cuda") -> str:
    from transformers import CLIPModel, CLIPProcessor

    avg_dir  = averaged_meta_dir(root)
    td_path  = avg_dir / f"targets_avg_{subj}.npz"
    out_path = clip_hidden_path(avg_dir, subj)

    if out_path.exists():
        arr = np.load(out_path, mmap_mode="r")
        print(f"  [{subj}] already exists — shape={arr.shape}  path={out_path}")
        if arr.shape[1:] != (N_TOKENS, D_HIDDEN):
            print(
                f"  !! Shape mismatch: expected (N,{N_TOKENS},{D_HIDDEN}), "
                f"got {arr.shape} — delete and re-run to regenerate."
            )
        return str(out_path)

    if not td_path.exists():
        raise FileNotFoundError(
            f"Missing {td_path} — run ingest-averaged first for {subj}"
        )

    td  = np.load(td_path)
    i73 = td["image_id_73k"].astype(np.int64)
    n   = len(i73)

    stim_path = root / "nsd_meta" / "nsd_stimuli.hdf5"
    if not stim_path.exists():
        raise FileNotFoundError(
            f"NSD stimuli HDF5 not found at {stim_path}"
        )

    dev  = torch.device(device if torch.cuda.is_available() else "cpu")
    repo = "openai/clip-vit-large-patch14"
    print(f"  [{subj}] loading {repo} …")
    clip      = CLIPModel.from_pretrained(repo).to(dev).eval()
    processor = CLIPProcessor.from_pretrained(repo)

    # write to tmp first so a crash doesn't leave a partial file
    tmp_path = out_path.with_suffix(".npy.tmp")
    hidden_mm = np.lib.format.open_memmap(
        str(tmp_path), mode="w+", dtype=np.float16, shape=(n, N_TOKENS, D_HIDDEN),
    )
    print(
        f"  [{subj}] encoding {n} images → (N, {N_TOKENS}, {D_HIDDEN}) float16\n"
        f"           batch_size={batch_size}  device={dev}"
    )

    with h5py.File(stim_path, "r") as f:
        brick = f["imgBrick"]
        for start in range(0, n, batch_size):
            end  = min(start + batch_size, n)
            imgs = [
                Image.fromarray(brick[int(i73[k])]).convert("RGB")
                for k in range(start, end)
            ]
            hidden_mm[start:end] = encode_hidden_batch(
                clip.vision_model, processor, imgs, dev
            )
            if start % (batch_size * 20) == 0 or end == n:
                print(f"    {end}/{n}")

    hidden_mm.flush()
    del hidden_mm
    tmp_path.rename(out_path)

    final = np.load(out_path, mmap_mode="r")
    print(
        f"  [{subj}] saved {out_path}\n"
        f"           shape={final.shape}  dtype={final.dtype}  "
        f"size={final.nbytes / 1e9:.2f} GB"
    )
    return str(out_path)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Cache ViT-L/14 raw hidden states (257×1024) for NSD averaged images"
    )
    p.add_argument("--subj",       default=os.environ.get("NSD_SUBJ",        "subj01"))
    p.add_argument("--root",       default=os.environ.get("MINDBRIDGE_ROOT", "/mnt/mindbridge"))
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device",     default="cuda")
    args = p.parse_args()

    root = get_root(args.root)
    subs = ALL_SUBJ if args.subj == "all" else [args.subj]
    for s in subs:
        ingest_subj(s, root, args.batch_size, args.device)


if __name__ == "__main__":
    main()