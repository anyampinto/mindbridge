# -*- coding: utf-8 -*-
"""
subj01 production Kneeland A+B recon (~65% paper 2WC).

Canonical recipe (kn_atxt_a_txt005_b_vd03_s29):
  - Dual: 20260602_053625_train_dual_contrastive_subj01_150ep, best_retrieval
  - Adapter: 20260603_230807_imagery_beta_adapter_subj01
  - Set A: mlp_text_hf_dual @ 0.05, projector S2, VD=0.12
  - Set B: prior DDIM τ=1.5, VD=0.3, seed 29
  - Stage 1: vdvae_ridge (both sets)

No MLP-recon / neutral-S1 / E2E training — use existing checkpoints on the volume.

Usage:
  python recon_pipeline_subj01.py
  modal run modal_app.py --step recon-pipeline-subj01 --subj subj01
  modal run modal_app.py --step kneeland-calibrated-recon --subj subj01
"""

from __future__ import annotations

import json
import os
import sys

sys.path = [p for p in sys.path if "share/software" not in p and "jupyterlab" not in p]

from pathlib import Path

from paths import get_root

# Production ~65.6% Kneeland paper A+B (subj01).
PRODUCTION_DUAL_RUN = "20260602_053625_train_dual_contrastive_subj01_150ep"
PRODUCTION_ADAPTER_RUN = "20260603_230807_imagery_beta_adapter_subj01"
PRODUCTION_OUT = "kn_atxt_a_txt005_b_vd03_s29"
PRODUCTION_CKPT = "best_retrieval"
PRODUCTION_STAGE1 = "vdvae_ridge"
PRODUCTION_SEED = 29
PRODUCTION_TAU = 1.5
PRODUCTION_VD_B = 0.3
PRODUCTION_TXT_A = 0.05
PRODUCTION_PERCEPTION_OUT = "kn_perception_setb_prior_vd03_s29"


def run_production_perception(
    subj: str = "subj01",
    root: Path | str | None = None,
    *,
    dual_run_id: str | None = None,
    ckpt_prefer: str = PRODUCTION_CKPT,
    out_subdir: str = PRODUCTION_PERCEPTION_OUT,
    prior_temperature: float = PRODUCTION_TAU,
    vd_strength: float = PRODUCTION_VD_B,
    seed: int = PRODUCTION_SEED,
    stage1_source: str = PRODUCTION_STAGE1,
    n_perception: int = 6,
) -> dict:
    """Production stack on Kneeland Set-B cues using perception betas (same S1/S2 as imagery Set B)."""
    root = get_root(root or os.environ.get("MINDBRIDGE_ROOT", "/mnt/mindbridge"))
    dual_rid = dual_run_id or os.environ.get("DUAL_RUN_ID", PRODUCTION_DUAL_RUN)
    osd = os.environ.get("PERCEPTION_OUT", out_subdir)

    print(f"\n{'=' * 72}")
    print("  PRODUCTION perception on Kneeland Set-B cues (same stack as imagery Set B)")
    print(f"  dual={dual_rid}  ckpt={ckpt_prefer}")
    print("  betas=perception avg (no imagery beta adapter)")
    print(f"  S1={stage1_source}  prior_τ={prior_temperature}  VD={vd_strength}  seed={seed}")
    print(f"  out={osd}")
    print(f"{'=' * 72}\n")

    from reconstruct_vdvae_dual import run_reconstruction

    recon_res = run_reconstruction(
        subj,
        root,
        run_id=dual_rid,
        variant="4H_CTR2",
        ckpt_prefer=ckpt_prefer,
        vd_strength=vd_strength,
        vd_steps=50,
        guidance_scale=7.5,
        hybrid_layers=True,
        perception_setb_only=True,
        stage2_source="prior",
        prior_run_id="",
        prior_temperature=prior_temperature,
        out_subdir=osd,
        eval_recon_kneeland=False,
        prior_seed=seed,
        vd_seed=seed,
        imagery_sets="set_b",
        stage1_source=stage1_source,
        stage1_source_perception=stage1_source,
    )

    summary = {
        "recipe": "production_perception_trials",
        "subj": subj,
        "dual_run_id": dual_rid,
        "n_perception": n_perception,
        "out_subdir": osd,
        "reconstruction": recon_res,
    }
    out_json = root / "results" / f"production_perception_{subj}_{osd}.json"
    out_json.write_text(json.dumps(summary, indent=2, default=str))
    if recon_res.get("perception_setb_grid"):
        print(f"  Grid: {recon_res['perception_setb_grid']}")
    elif recon_res.get("perception_grid"):
        print(f"  Grid: {recon_res['perception_grid']}")
    print(f"  Summary: {out_json}\n")
    return summary


def run_pipeline(
    subj: str = "subj01",
    root: Path | str | None = None,
    *,
    dual_run_id: str | None = None,
    beta_adapter_run_id: str | None = None,
    ckpt_prefer: str = PRODUCTION_CKPT,
    out_subdir: str = PRODUCTION_OUT,
    prior_temperature: float = PRODUCTION_TAU,
    vd_strength: float = PRODUCTION_VD_B,
    hf_text_strength: float = PRODUCTION_TXT_A,
    seed: int = PRODUCTION_SEED,
    stage1_source: str = PRODUCTION_STAGE1,
    n_perception: int = 3,
    **_ignored,
) -> dict:
    """Run production Kneeland A+B grids only (no retraining)."""
    root = get_root(root or os.environ.get("MINDBRIDGE_ROOT", "/mnt/mindbridge"))
    dual_rid = dual_run_id or os.environ.get("DUAL_RUN_ID", PRODUCTION_DUAL_RUN)
    beta_rid = beta_adapter_run_id or os.environ.get(
        "BETA_ADAPTER_RUN", PRODUCTION_ADAPTER_RUN,
    )
    osd = os.environ.get("KNEELAND_OUT", os.environ.get("PIPELINE_OUT", out_subdir))

    print(f"\n{'=' * 72}")
    print("  PRODUCTION Kneeland A+B subj01 (~65% recipe)")
    print(f"  dual={dual_rid}")
    print(f"  adapter={beta_rid}  ckpt={ckpt_prefer}")
    print(f"  S1={stage1_source}  prior_τ={prior_temperature}  VD_B={vd_strength}")
    print(f"  Set A HF text={hf_text_strength}  seed={seed}  out={osd}")
    print(f"{'=' * 72}\n")

    from reconstruct_vdvae_dual import run_reconstruction

    recon_res = run_reconstruction(
        subj,
        root,
        run_id=dual_rid,
        variant="4H_CTR2",
        ckpt_prefer=ckpt_prefer,
        n_perception=n_perception,
        vd_strength=vd_strength,
        vd_steps=50,
        guidance_scale=7.5,
        hybrid_layers=True,
        beta_adapter_run_id=beta_rid,
        stage2_source="prior",
        prior_run_id="",
        prior_temperature=prior_temperature,
        out_subdir=osd,
        eval_recon_kneeland=True,
        text_source="",
        text_to_image_strength=hf_text_strength,
        prior_seed=seed,
        vd_seed=seed,
        imagery_sets="kneeland",
        kneeland_b_vd_strength=vd_strength,
        kneeland_a_text_source="mlp_text_hf_dual",
        kneeland_b_text_source="",
        stage1_source=stage1_source,
        stage1_source_perception=stage1_source,
        stage1_source_imagery=stage1_source,
    )

    summary = {
        "recipe": "production_kneeland_ab_65pct",
        "subj": subj,
        "dual_run_id": dual_rid,
        "beta_adapter_run_id": beta_rid,
        "ckpt_prefer": ckpt_prefer,
        "out_subdir": osd,
        "reconstruction": recon_res,
        "out_dir": recon_res.get("out_dir"),
    }
    out_json = root / "results" / f"production_kneeland_{subj}_{osd}.json"
    out_json.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\n{'=' * 72}")
    print(f"  DONE — {out_json}")
    if recon_res.get("imagery_setA_grid"):
        print(f"  Set A: {recon_res['imagery_setA_grid']}")
    if recon_res.get("imagery_setB_grid"):
        print(f"  Set B: {recon_res['imagery_setB_grid']}")
    k2 = recon_res.get("recon_kneeland_2wc_avg_ab_paper") or recon_res.get("recon_kneeland_2wc_paper", {})
    if k2:
        print(f"  Kneeland paper A+B: {k2}")
    print(f"{'=' * 72}\n")
    return summary


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Production Kneeland A+B recon (subj01)")
    ap.add_argument("--subj", default="subj01")
    ap.add_argument("--root", default=os.environ.get("MINDBRIDGE_ROOT", "/mnt/mindbridge"))
    ap.add_argument("--dual-run-id", default=None)
    ap.add_argument("--beta-adapter-run-id", default=None)
    ap.add_argument("--out-subdir", default=PRODUCTION_OUT)
    ap.add_argument(
        "--perception-only",
        action="store_true",
        help="Run production stack on perception val trials only",
    )
    ap.add_argument("--n-perception", type=int, default=int(os.environ.get("N_PERCEPTION", "6")))
    args = ap.parse_args()

    if args.perception_only:
        run_production_perception(
            args.subj,
            args.root,
            dual_run_id=args.dual_run_id,
            out_subdir=args.out_subdir or PRODUCTION_PERCEPTION_OUT,
            n_perception=args.n_perception,
        )
        return

    run_pipeline(
        args.subj,
        args.root,
        dual_run_id=args.dual_run_id,
        beta_adapter_run_id=args.beta_adapter_run_id,
        out_subdir=args.out_subdir,
    )


if __name__ == "__main__":
    main()
