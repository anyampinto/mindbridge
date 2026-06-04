# -*- coding: utf-8 -*-
"""Re-run reconstruction Kneeland 2WC on Stage-2 row extracted from a saved grid PNG."""

from __future__ import annotations

import argparse
import json
import sys

sys.path = [p for p in sys.path if "share/software" not in p and "jupyterlab" not in p]

from pathlib import Path

import numpy as np
from PIL import Image

from eval_recon_kneeland import eval_recon_kneeland
from paths import get_root
from reconstruct_vdvae_dual import SET_B_EVAL_CUES, select_set_b_eval_indices


def load_stage2_images(stage2_dir: Path, cues: tuple[str, ...]) -> list[Image.Image]:
    imgs = []
    for cue in cues:
        p = stage2_dir / f"{cue}_stage2.png"
        if not p.exists():
            raise FileNotFoundError(p)
        imgs.append(Image.open(p).convert("RGB").resize((256, 256), Image.LANCZOS))
    return imgs


def extract_stage2_from_grid(grid_path: Path, n_cols: int = 5, n_rows: int = 3) -> list[Image.Image]:
    """Split a 3×N matplotlib grid into Stage-2 (bottom row) PIL images."""
    img = Image.open(grid_path).convert("RGB")
    w, h = img.size
    row_h = h // n_rows
    imgs: list[Image.Image] = []
    for col in range(n_cols):
        left = col * (w // n_cols)
        right = (col + 1) * (w // n_cols) if col < n_cols - 1 else w
        top = 2 * row_h
        bottom = h if col == n_cols - 1 else (2 * row_h + row_h)
        crop = img.crop((left, top, right, bottom))
        imgs.append(crop.resize((256, 256), Image.LANCZOS))
    return imgs


def main() -> None:
    ap = argparse.ArgumentParser(description="Re-eval Kneeland 2WC from saved imagery grid PNG.")
    ap.add_argument("--subj", default="subj01")
    ap.add_argument("--root", default=None)
    ap.add_argument("--grid", default=None, help="Path to imagery_setB_grid.png (3 rows)")
    ap.add_argument("--stage2-dir", default=None, help="Dir with W_stage2.png … (preferred over grid crop)")
    ap.add_argument("--n-cols", type=int, default=5)
    ap.add_argument("--n-pairs", type=int, default=1000)
    ap.add_argument("--label", default="", help="Run label for logging")
    ap.add_argument("--expected-k2wc", type=float, default=None)
    args = ap.parse_args()

    root = get_root(args.root)
    meta_path = root / "nsd_meta" / f"imagery_trial_meta_{args.subj}.json"
    meta = json.loads(meta_path.read_text())
    row_idx = select_set_b_eval_indices(meta)

    if args.stage2_dir:
        stage2 = load_stage2_images(Path(args.stage2_dir), SET_B_EVAL_CUES)
        src = args.stage2_dir
    elif args.grid:
        grid_path = Path(args.grid)
        if not grid_path.exists():
            raise FileNotFoundError(grid_path)
        stage2 = extract_stage2_from_grid(grid_path, n_cols=args.n_cols)
        src = str(grid_path)
    else:
        raise ValueError("Provide --stage2-dir or --grid")
    if len(stage2) != len(row_idx):
        raise ValueError(f"Grid has {len(stage2)} cols but Set-B eval has {len(row_idx)} rows")

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    k2 = eval_recon_kneeland(
        stage2, row_idx, args.subj, root, subset="kneeland", n_pairs=args.n_pairs, device=device,
    )

    tag = args.label or (Path(src).parent.name if args.grid else Path(src).name)
    print(f"\n=== Re-eval {tag} ===")
    print(f"  Source: {src}")
    print(f"  Cues: {SET_B_EVAL_CUES}")
    print(f"  Kneeland 2WC: {k2['kneeland_2wc']:.4f} ({100 * k2['kneeland_2wc']:.1f}%)")
    if args.expected_k2wc is not None:
        delta = k2["kneeland_2wc"] - args.expected_k2wc
        print(f"  Expected:     {args.expected_k2wc:.4f} ({100 * args.expected_k2wc:.1f}%)  Δ={delta:+.4f}")
    print(json.dumps(k2, indent=2))


if __name__ == "__main__":
    main()
