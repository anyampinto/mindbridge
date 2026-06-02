# -*- coding: utf-8 -*-
"""
Compute ViT-L/14 patch+CLS hidden states (257×768) for averaged perception images.

Writes clip_hidden_avg_{subj}.npy (float16) — separate file to avoid corrupting targets_avg npz.

Usage:
  python ingest_clip_hidden.py --subj subj01 --root /mnt/mindbridge
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
N_TOKENS = 257
D_CLIP = 768


def clip_hidden_path(out_dir: Path, subj: str) -> Path:
    return out_dir / f"clip_hidden_avg_{subj}.npy"


@torch.no_grad()
def encode_clip_hidden_batch(
    vision_model,
    visual_projection,
    processor,
    images: list[Image.Image],
    device: torch.device,
) -> np.ndarray:
    inputs = processor(images=images, return_tensors="pt")
    pixel = inputs["pixel_values"].to(device)
    out = vision_model(pixel_values=pixel)
    hidden = visual_projection(out.last_hidden_state)
    hidden = F.normalize(hidden, dim=-1)
    return hidden.cpu().numpy().astype(np.float16)


def ingest_subj(subj: str, root: Path, batch_size: int = 16, device: str = "cuda") -> str:
    from transformers import CLIPModel, CLIPProcessor

    avg_dir = averaged_meta_dir(root)
    td_path = avg_dir / f"targets_avg_{subj}.npz"
    out_path = clip_hidden_path(avg_dir, subj)
    if out_path.exists():
        print(f"  [{subj}] already exists {out_path} shape={np.load(out_path, mmap_mode='r').shape}")
        return str(out_path)

    if not td_path.exists():
        raise FileNotFoundError(f"Missing {td_path} — run ingest-averaged first")

    td = np.load(td_path)
    i73 = td["image_id_73k"].astype(np.int64)
    n = len(i73)
    stim_path = root / "nsd_meta" / "nsd_stimuli.hdf5"
    if not stim_path.exists():
        raise FileNotFoundError(stim_path)

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    repo = "openai/clip-vit-large-patch14"
    clip = CLIPModel.from_pretrained(repo).to(dev).eval()
    processor = CLIPProcessor.from_pretrained(repo)

    tmp_path = out_path.with_suffix(".npy.tmp")
    hidden = np.lib.format.open_memmap(
        str(tmp_path), mode="w+", dtype=np.float16, shape=(n, N_TOKENS, D_CLIP),
    )
    print(f"  [{subj}] encoding {n} images → {out_path.name}")

    with h5py.File(stim_path, "r") as f:
        brick = f["imgBrick"]
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            imgs = [Image.fromarray(brick[int(i73[k])]).convert("RGB") for k in range(start, end)]
            hidden[start:end] = encode_clip_hidden_batch(
                clip.vision_model, clip.visual_projection, processor, imgs, dev,
            )
            if start % 320 == 0 or end == n:
                print(f"    {end}/{n}")

    hidden.flush()
    del hidden
    tmp_path.rename(out_path)
    print(f"  [{subj}] saved {out_path}")
    return str(out_path)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--subj", default=os.environ.get("NSD_SUBJ", "subj01"))
    p.add_argument("--root", default=os.environ.get("MINDBRIDGE_ROOT", "/mnt/mindbridge"))
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    root = get_root(args.root)
    subs = ALL_SUBJ if args.subj == "all" else [args.subj]
    for s in subs:
        ingest_subj(s, root, args.batch_size, args.device)


if __name__ == "__main__":
    main()
