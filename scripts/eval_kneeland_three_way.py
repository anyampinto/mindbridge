# -*- coding: utf-8 -*-
"""Report Kneeland 2WC for three imagery-decoding conditions."""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path = [p for p in sys.path if "share/software" not in p and "jupyterlab" not in p]

from pathlib import Path

from eval_imagery_crossdecode import eval_subj
from paths import get_root

DEFAULT_DUAL_RUN = "20260602_053625_train_dual_contrastive_subj01_150ep"
DEFAULT_JOINT_RIDGE_RUN = "20260601_053733_joint_ridge_all8"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subj", default=os.environ.get("NSD_SUBJ", "subj01"))
    ap.add_argument("--root", default=os.environ.get("MINDBRIDGE_ROOT", "/mnt/mindbridge"))
    ap.add_argument("--joint-ridge-run-id", default=DEFAULT_JOINT_RIDGE_RUN)
    ap.add_argument("--dual-run-id", default=DEFAULT_DUAL_RUN)
    ap.add_argument("--beta-adapter-run-id", default="")
    ap.add_argument("--mlp-adapter-run-id", default="")
    ap.add_argument("--n-pairs", type=int, default=1000)
    args = ap.parse_args()

    root = get_root(args.root)
    rows: list[dict] = []

    def kneeland_2wc(res: dict) -> float:
        if "joint_ridge_clip" in res:
            return float(res["joint_ridge_clip"]["kneeland_2wc"]["kneeland_2wc"])
        key = "projector_clip768" if "projector_clip768" in res else "regression_clip"
        return float(res[key]["kneeland_2wc"]["kneeland_2wc"])

    print("\n=== Kneeland 2WC comparison (Set A+B, 12 stimuli) ===\n")

    r1 = eval_subj(
        args.subj, root, "joint_ridge", args.joint_ridge_run_id, "best_retrieval",
        subset="kneeland", head="clip", n_pairs=args.n_pairs, metric="kneeland",
    )
    rows.append({"condition": "No adapter, joint ridge", "kneeland_2wc": kneeland_2wc(r1)})

    if args.beta_adapter_run_id:
        r2 = eval_subj(
            args.subj, root, "joint_ridge", args.joint_ridge_run_id, "best_retrieval",
            subset="kneeland", head="clip", n_pairs=args.n_pairs, metric="kneeland",
            beta_adapter_run_id=args.beta_adapter_run_id,
        )
        rows.append({"condition": "Beta adapter → joint ridge", "kneeland_2wc": kneeland_2wc(r2)})

    r3 = eval_subj(
        args.subj, root, "4H_CTR2", args.dual_run_id, "best_retrieval",
        subset="kneeland", head="projector", n_pairs=args.n_pairs, metric="kneeland",
        beta_adapter_run_id=None,
    )
    rows.append({"condition": "No adapter, dual projector", "kneeland_2wc": kneeland_2wc(r3)})

    if args.mlp_adapter_run_id:
        r4 = eval_subj(
            args.subj, root, "4H_CTR2", args.dual_run_id, "best_retrieval",
            subset="kneeland", head="projector", n_pairs=args.n_pairs, metric="kneeland",
            beta_adapter_run_id=args.mlp_adapter_run_id,
        )
        rows.append({
            "condition": "MLP adapter → dual projector",
            "kneeland_2wc": kneeland_2wc(r4),
        })

    for row in rows:
        pct = 100.0 * row["kneeland_2wc"]
        print(f"  {row['condition']:<35} {row['kneeland_2wc']:.4f}  ({pct:.1f}%)")

    out = {"subj": args.subj, "conditions": rows}
    out_path = root / "results" / f"kneeland_three_way_{args.subj}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n  Saved {out_path}")


if __name__ == "__main__":
    main()
