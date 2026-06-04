# -*- coding: utf-8 -*-
"""
subj01 production Kneeland A+B grids — same recipe as ~65% run.

  kn_atxt_a_txt005_b_vd03_s29

Usage:
  python paper_grids_subj01.py
  modal run modal_app.py --step paper-grids-subj01 --subj subj01
  modal run modal_app.py --step kneeland-calibrated-recon --subj subj01
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path = [p for p in sys.path if "share/software" not in p and "jupyterlab" not in p]

from pathlib import Path

from paths import get_root
from recon_pipeline_subj01 import (
    PRODUCTION_ADAPTER_RUN,
    PRODUCTION_DUAL_RUN,
    PRODUCTION_OUT,
    PRODUCTION_SEED,
    PRODUCTION_STAGE1,
    PRODUCTION_TAU,
    PRODUCTION_TXT_A,
    PRODUCTION_VD_B,
    run_pipeline,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="Production Kneeland A+B grids (subj01)")
    ap.add_argument("--subj", default="subj01")
    ap.add_argument("--root", default=os.environ.get("MINDBRIDGE_ROOT", "/mnt/mindbridge"))
    ap.add_argument("--run-id", default=os.environ.get("DUAL_RUN_ID", PRODUCTION_DUAL_RUN))
    ap.add_argument("--beta-adapter-run-id", default=os.environ.get("BETA_ADAPTER_RUN", PRODUCTION_ADAPTER_RUN))
    ap.add_argument("--out-subdir", default=os.environ.get("PAPER_OUT", os.environ.get("KNEELAND_OUT", PRODUCTION_OUT)))
    ap.add_argument("--ckpt-prefer", default=os.environ.get("CKPT_PREFER", "best_retrieval"))
    ap.add_argument("--stage1-source", default=os.environ.get("STAGE1_SOURCE", PRODUCTION_STAGE1))
    ap.add_argument("--n-perception", type=int, default=int(os.environ.get("N_PERCEPTION", "3")))
    ap.add_argument("--prior-temperature", type=float, default=float(os.environ.get("PRIOR_TEMPERATURE", str(PRODUCTION_TAU))))
    ap.add_argument("--vd-strength", type=float, default=float(os.environ.get("VD_STRENGTH", str(PRODUCTION_VD_B))))
    ap.add_argument("--hf-text-strength", type=float, default=float(os.environ.get("HF_TEXT_STRENGTH", str(PRODUCTION_TXT_A))))
    ap.add_argument("--seed", type=int, default=int(os.environ.get("PRIOR_SEED", str(PRODUCTION_SEED))))
    args = ap.parse_args()

    root = get_root(args.root)
    results = run_pipeline(
        args.subj,
        root,
        dual_run_id=args.run_id,
        beta_adapter_run_id=args.beta_adapter_run_id,
        ckpt_prefer=args.ckpt_prefer,
        out_subdir=args.out_subdir,
        prior_temperature=args.prior_temperature,
        vd_strength=args.vd_strength,
        hf_text_strength=args.hf_text_strength,
        seed=args.seed,
        stage1_source=args.stage1_source,
        n_perception=args.n_perception,
    )
    recon = results.get("reconstruction", results)
    out_json = root / "results" / f"paper_grids_{args.subj}_{args.out_subdir}.json"
    out_json.write_text(json.dumps(recon, indent=2, default=str))
    print(f"\n  Saved summary {out_json}")


if __name__ == "__main__":
    main()
