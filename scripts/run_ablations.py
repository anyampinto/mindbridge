# -*- coding: utf-8 -*-
"""
Run MindBridge gatekeeper ablations against the production Kneeland A+B recipe.

Ablation 1 — Prior-only (no brain input): zeros | mean | random
Ablation 2 — ROI voxel shuffle (spatial topography destroyed)
Ablation 3 — Prior-free (brain-only): regression CLIP, no DDIM prior, CFG off, no text
Ablation 4 — 1-head contrastive (train separately, then --ablation 1head)

Usage:
  python run_ablations.py --subj subj01 --ablation zeros
  python run_ablations.py --subj subj01 --ablation all-inference
  modal run modal_app.py --step ablation-suite --subj subj01
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path = [p for p in sys.path if "share/software" not in p and "jupyterlab" not in p]

from pathlib import Path

from ablation_utils import BRAIN_ABLATION_CHOICES, ablation_out_subdir
from paths import get_root
from recon_pipeline_subj01 import (
    PRODUCTION_ADAPTER_RUN,
    PRODUCTION_DUAL_RUN,
    PRODUCTION_OUT,
    PRODUCTION_SEED,
    PRODUCTION_TAU,
    PRODUCTION_TXT_A,
    PRODUCTION_VD_B,
)

INFERENCE_ABLATIONS = ("zeros", "mean", "random", "roi_shuffle", "prior_free")


def run_single_ablation(
    name: str,
    subj: str,
    root: Path,
    *,
    dual_run_id: str,
    beta_adapter_run_id: str,
    base_out: str,
    seed: int,
) -> dict:
    from reconstruct_vdvae_dual import run_reconstruction

    brain_ablation = None
    prior_free = False
    out_subdir = ablation_out_subdir(base_out, name)
    guidance_scale = 7.5
    stage2_source = "prior"
    text_source = ""
    kneeland_a_text = "mlp_text_hf_dual"
    text_strength = PRODUCTION_TXT_A
    eval_kneeland = True

    if name in BRAIN_ABLATION_CHOICES:
        brain_ablation = name
        print(f"\n{'=' * 72}")
        print(f"  ABLATION: Prior-only gatekeeper — brain input = {name}")
        print(f"{'=' * 72}\n")
    elif name == "prior_free":
        prior_free = True
        out_subdir = ablation_out_subdir(base_out, "prior-free")
        guidance_scale = 1.0
        stage2_source = "regression"
        kneeland_a_text = ""
        text_strength = None
        print(f"\n{'=' * 72}")
        print("  ABLATION: Prior-free (brain-only regression CLIP, CFG=1.0, no text/prior)")
        print(f"{'=' * 72}\n")
    elif name == "1head":
        dual_run_id = os.environ.get(
            "ONEHEAD_RUN_ID",
            "20260605_train_dual_contrastive_1head_subj01",
        )
        out_subdir = ablation_out_subdir(base_out, "1head")
        print(f"\n{'=' * 72}")
        print(f"  ABLATION: 1-head contrastive collapse — run_id={dual_run_id}")
        print(f"{'=' * 72}\n")
    else:
        raise ValueError(f"Unknown ablation {name!r}")

    recon = run_reconstruction(
        subj,
        root,
        run_id=dual_run_id,
        variant="4H_CTR2" if name != "1head" else "4H_CTR2_1H",
        ckpt_prefer="best_retrieval",
        vd_strength=PRODUCTION_VD_B,
        vd_steps=50,
        guidance_scale=guidance_scale,
        hybrid_layers=True,
        beta_adapter_run_id=beta_adapter_run_id,
        stage2_source=stage2_source,
        prior_run_id="",
        prior_temperature=PRODUCTION_TAU,
        out_subdir=out_subdir,
        eval_recon_kneeland=eval_kneeland,
        text_source=text_source,
        text_to_image_strength=text_strength,
        prior_seed=seed,
        vd_seed=seed,
        imagery_sets="kneeland",
        kneeland_b_vd_strength=PRODUCTION_VD_B,
        kneeland_a_text_source=kneeland_a_text,
        kneeland_b_text_source="",
        stage1_source="vdvae_ridge",
        brain_ablation=brain_ablation,
        prior_free=prior_free,
    )

    paper = recon.get("recon_kneeland_2wc_avg_ab_paper") or recon.get("recon_kneeland_2wc_paper", {})
    score = float(paper.get("kneeland_2wc", float("nan")))
    summary = {
        "ablation": name,
        "subj": subj,
        "out_subdir": out_subdir,
        "dual_run_id": dual_run_id,
        "beta_adapter_run_id": beta_adapter_run_id,
        "brain_ablation": brain_ablation,
        "prior_free": prior_free,
        "kneeland_2wc_paper_ab": score,
        "kneeland_2wc_pct": 100.0 * score if score == score else None,
        "reconstruction": recon,
    }
    return summary


def run_all_inference(subj: str, root: Path, **kwargs) -> list[dict]:
    results = []
    for name in INFERENCE_ABLATIONS:
        try:
            results.append(run_single_ablation(name, subj, root, **kwargs))
        except Exception as exc:
            print(f"  FAILED ablation={name}: {exc}")
            results.append({"ablation": name, "error": str(exc)})
    return results


def run_eval_only(subj: str, root: Path, base_out: str = PRODUCTION_OUT) -> list[dict]:
    """Re-score saved ablation recon dirs (skip re-reconstruction)."""
    import torch
    from eval_recon_kneeland import eval_kneeland_ab_from_run_dir

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for name in INFERENCE_ABLATIONS:
        if name == "prior_free":
            sub = ablation_out_subdir(base_out, "prior-free")
        else:
            sub = ablation_out_subdir(base_out, name)
        out_dir = root / "reconstructions" / subj / sub
        if not out_dir.exists():
            rows.append({"ablation": name, "error": f"missing {out_dir}"})
            continue
        res = eval_kneeland_ab_from_run_dir(out_dir, subj, root, device=device)
        score = float(res["avg_ab_paper"])
        rows.append({
            "ablation": name,
            "out_subdir": sub,
            "kneeland_2wc_paper_ab": score,
            "kneeland_2wc_pct": 100.0 * score,
            "eval": res,
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="MindBridge ablation runner")
    ap.add_argument("--subj", default="subj01")
    ap.add_argument("--root", default=os.environ.get("MINDBRIDGE_ROOT", "/mnt/mindbridge"))
    ap.add_argument(
        "--ablation",
        default="all-inference",
        help=f"One of {INFERENCE_ABLATIONS + ('1head', 'all-inference', 'eval-only')}",
    )
    ap.add_argument("--dual-run-id", default=os.environ.get("DUAL_RUN_ID", PRODUCTION_DUAL_RUN))
    ap.add_argument(
        "--beta-adapter-run-id",
        default=os.environ.get("BETA_ADAPTER_RUN", PRODUCTION_ADAPTER_RUN),
    )
    ap.add_argument("--base-out", default=os.environ.get("KNEELAND_OUT", PRODUCTION_OUT))
    ap.add_argument("--seed", type=int, default=int(os.environ.get("PRIOR_SEED", str(PRODUCTION_SEED))))
    args = ap.parse_args()
    root = get_root(args.root)

    common = dict(
        dual_run_id=args.dual_run_id,
        beta_adapter_run_id=args.beta_adapter_run_id,
        base_out=args.base_out,
        seed=args.seed,
    )

    if args.ablation == "all-inference":
        rows = run_all_inference(args.subj, root, **common)
    elif args.ablation == "eval-only":
        rows = run_eval_only(args.subj, root, base_out=args.base_out)
    else:
        rows = [run_single_ablation(args.ablation, args.subj, root, **common)]

    out_json = root / "results" / f"ablations_{args.subj}_{args.ablation.replace('-', '_')}.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(rows, indent=2, default=str))

    print(f"\n{'=' * 72}")
    print("  ABLATION SUMMARY (Kneeland paper 2WC A+B)")
    print(f"{'=' * 72}")
    for row in rows:
        if "error" in row:
            print(f"  {row['ablation']:14s}  FAILED: {row['error']}")
        else:
            pct = row.get("kneeland_2wc_pct")
            pct_s = f"{pct:.1f}%" if pct is not None else "n/a"
            print(f"  {row['ablation']:14s}  2WC={pct_s}  out={row['out_subdir']}")
    print(f"\n  Saved {out_json}\n")


if __name__ == "__main__":
    main()
