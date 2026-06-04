# -*- coding: utf-8 -*-
"""
Perception–Imagery beta adapter — align imagery betas before a frozen downstream model.

Default (--align-space mlp): contrastive alignment in frozen dual-contrastive BrainMLP
encoder space (B×16×256 → mean-pool → B×256) with BiMixCo loss.

Legacy (--align-space beta): raw beta-space InfoNCE (joint-ridge path).

LOO: train on N-1 paired stimuli, evaluate representation match on held-out 1.

Usage:
  python train_imagery_beta_adapter.py --subj subj01 --align-space mlp
  python train_imagery_beta_adapter.py --subj subj01 --align-space beta
  python train_imagery_beta_adapter.py --subj subj01 --align-space hybrid --beta-weight 0.35
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path = [p for p in sys.path if "share/software" not in p and "jupyterlab" not in p]

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

from contrastive_losses import bidirectional_clip_loss, bimixco_loss
from imagery_perception_pairs import load_paired_betas, perception_voxel_stats
from paths import get_root, resolve_run_id, resolve_variant_checkpoint, run_root, update_run_manifest
from reconstruct_common import build_recon_model, normalize_variant

VARIANT = "ImgBetaAdapter"
DEFAULT_DUAL_RUN = "20260602_053625_train_dual_contrastive_subj01_150ep"
KNEELAND_SETS = frozenset({"A", "B"})


def resolve_honest_fit_indices(
    paired_rows: list[dict],
    *,
    exclude_sets_from_fit: frozenset[str] | None = None,
    fit_only_sets: frozenset[str] | None = None,
) -> tuple[list[int], list[int], str]:
    """
    Choose final-fit rows and Kneeland holdout rows for honest evaluation.

    Kneeland-honest default: fit only Set C (concept trials, no nsdimagery PNGs).
    Sets A/B are image-based Kneeland eval cues — never used for final adapter fit.
    """
    n = len(paired_rows)
    if not exclude_sets_from_fit and not fit_only_sets:
        return list(range(n)), [], "full_fit"

    if fit_only_sets:
        fit_idx = [i for i, r in enumerate(paired_rows) if str(r.get("set", "")) in fit_only_sets]
        protocol = "fit_" + "".join(sorted(fit_only_sets)) + "_holdout_ab"
    elif exclude_sets_from_fit == KNEELAND_SETS:
        fit_idx = [i for i, r in enumerate(paired_rows) if str(r.get("set", "")) == "C"]
        protocol = "fit_set_c_holdout_ab"
        if not fit_idx:
            raise ValueError(
                "Honest adapter: no Set-C imagery↔perception pairs. "
                "Ingest Set C (ingest-imagery-setc) so adapter can train on concepts only.",
            )
        print(
            "  Honest fit: Set C only (no image stimuli) — Sets A/B held out for Kneeland eval.",
        )
    else:
        fit_idx = [
            i for i, r in enumerate(paired_rows)
            if str(r.get("set", "")) not in exclude_sets_from_fit
        ]
        protocol = "exclude_" + "".join(sorted(exclude_sets_from_fit))
        if not fit_idx:
            raise ValueError(
                f"Honest adapter fit: no paired rows outside {sorted(exclude_sets_from_fit)}.",
            )

    holdout_idx = [
        i for i, r in enumerate(paired_rows)
        if str(r.get("set", "")) in KNEELAND_SETS and i not in fit_idx
    ]
    fit_sets = sorted({str(paired_rows[i].get("set")) for i in fit_idx})
    hold_sets = sorted({str(paired_rows[i].get("set")) for i in holdout_idx})
    print(f"  Final fit: n={len(fit_idx)} sets={fit_sets}  holdout: n={len(holdout_idx)} sets={hold_sets}")
    print(f"  Honest protocol: {protocol}")
    return fit_idx, holdout_idx, protocol


class BetaAdapter(nn.Module):
    """Map normalized imagery betas → perception-aligned normalized betas (residual MLP)."""

    def __init__(self, n_voxels: int, hidden: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_voxels, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_voxels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


def batch_retrieval_acc(pred: torch.Tensor, target: torch.Tensor) -> float:
    p = F.normalize(pred.float(), dim=-1)
    t = F.normalize(target.float(), dim=-1)
    sim = p @ t.T
    labels = torch.arange(len(p), device=p.device)
    i2t = (sim.argmax(dim=1) == labels).float().mean()
    t2i = (sim.argmax(dim=0) == labels).float().mean()
    return float(((i2t + t2i) * 0.5).item())


def encoder_repr(encoder: nn.Module, x_norm: torch.Tensor) -> torch.Tensor:
    """BrainMLP tokens (B,16,256) → mean-pool → L2-normalized (B,256)."""
    tokens = encoder(x_norm)
    return F.normalize(tokens.mean(dim=1), dim=-1)


def load_frozen_dual_mlp(
    root: Path,
    subj: str,
    dual_run_id: str,
    device: torch.device,
) -> tuple[nn.Module, np.ndarray, np.ndarray]:
    ckpt_path = resolve_variant_checkpoint(
        root, "4H_CTR2", subj, dual_run_id, prefer="best_retrieval",
    )
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    vm = np.asarray(ckpt["voxel_mean"]).reshape(1, -1).astype(np.float32)
    vs = np.asarray(ckpt["voxel_std"]).reshape(1, -1).astype(np.float32)
    model = build_recon_model("4H_CTR2", int(ckpt["n_voxels"]), device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, vm, vs


def normalize_raw(betas: np.ndarray, vm: np.ndarray, vs: np.ndarray) -> np.ndarray:
    return ((betas.astype(np.float32) - vm) / vs).astype(np.float32)


def beta_align_loss(adapted: torch.Tensor, x_perc: torch.Tensor, temperature: float) -> torch.Tensor:
    """Direct normalized-beta alignment (paired perception target)."""
    return bidirectional_clip_loss(adapted, x_perc, temperature=temperature)


def train_adapter_fold(
    *,
    img_n: np.ndarray,
    perc_n: np.ndarray,
    train_idx: np.ndarray,
    encoder: nn.Module | None,
    device: torch.device,
    align_space: str,
    hidden: int,
    dropout: float,
    lr: float,
    epochs: int,
    batch_size: int,
    temperature: float,
    beta_weight: float = 0.35,
    init_state: dict | None = None,
) -> BetaAdapter:
    n_voxels = img_n.shape[1]
    adapter = BetaAdapter(n_voxels, hidden=hidden, dropout=dropout).to(device)
    if init_state is not None:
        adapter.load_state_dict(init_state)
    opt = torch.optim.AdamW(adapter.parameters(), lr=lr, weight_decay=1e-4)

    tr_img = torch.tensor(img_n[train_idx], dtype=torch.float32)
    tr_perc = torch.tensor(perc_n[train_idx], dtype=torch.float32)
    loader = DataLoader(
        TensorDataset(tr_img, tr_perc),
        batch_size=min(batch_size, len(train_idx)),
        shuffle=True,
        drop_last=len(train_idx) >= batch_size,
    )

    for _ in range(epochs):
        adapter.train()
        for x_img, x_perc in loader:
            x_img = x_img.to(device)
            x_perc = x_perc.to(device)
            adapted = adapter(x_img)
            if align_space == "mlp":
                assert encoder is not None
                with torch.no_grad():
                    tgt_repr = encoder_repr(encoder, x_perc)
                pred_repr = encoder_repr(encoder, adapted)
                loss = bimixco_loss(pred_repr, tgt_repr, temperature=temperature)
            elif align_space == "beta":
                loss = beta_align_loss(adapted, x_perc, temperature)
            else:
                assert align_space == "hybrid"
                assert encoder is not None
                with torch.no_grad():
                    tgt_repr = encoder_repr(encoder, x_perc)
                pred_repr = encoder_repr(encoder, adapted)
                loss_mlp = bimixco_loss(pred_repr, tgt_repr, temperature=temperature)
                loss_beta = beta_align_loss(adapted, x_perc, temperature)
                w = float(beta_weight)
                loss = (1.0 - w) * loss_mlp + w * loss_beta
            opt.zero_grad()
            loss.backward()
            opt.step()
    return adapter


@torch.no_grad()
def eval_adapter_fold(
    adapter: BetaAdapter,
    *,
    img_n: np.ndarray,
    perc_n: np.ndarray,
    eval_idx: np.ndarray,
    encoder: nn.Module | None,
    device: torch.device,
    align_space: str,
) -> dict:
    adapter.eval()
    x_img = torch.tensor(img_n[eval_idx], dtype=torch.float32, device=device)
    x_perc = torch.tensor(perc_n[eval_idx], dtype=torch.float32, device=device)
    adapted = adapter(x_img)
    beta_cos = float(
        F.cosine_similarity(
            F.normalize(adapted, dim=-1), F.normalize(x_perc, dim=-1), dim=-1,
        ).mean().item()
    )
    beta_ret = batch_retrieval_acc(adapted, x_perc) if len(eval_idx) > 1 else float("nan")

    if align_space in ("mlp", "hybrid"):
        assert encoder is not None
        pred_repr = encoder_repr(encoder, adapted)
        tgt_repr = encoder_repr(encoder, x_perc)
        repr_cos = float(F.cosine_similarity(pred_repr, tgt_repr, dim=-1).mean().item())
        repr_ret = batch_retrieval_acc(pred_repr, tgt_repr) if len(eval_idx) > 1 else float("nan")
        return {
            "cosine": repr_cos,
            "retrieval": repr_ret,
            "repr_cosine": repr_cos,
            "repr_retrieval": repr_ret,
            "beta_cosine": beta_cos,
            "beta_retrieval": beta_ret,
        }
    return {
        "cosine": beta_cos,
        "retrieval": beta_ret,
        "repr_cosine": float("nan"),
        "repr_retrieval": float("nan"),
        "beta_cosine": beta_cos,
        "beta_retrieval": beta_ret,
    }


def loo_evaluate(
    img_n: np.ndarray,
    perc_n: np.ndarray,
    paired_rows: list[dict],
    *,
    encoder: nn.Module | None,
    device: torch.device,
    align_space: str,
    subset: str,
    **train_kw,
) -> dict:
    if subset == "kneeland":
        use_idx = np.array([i for i, r in enumerate(paired_rows) if r["set"] in ("A", "B")])
    else:
        use_idx = np.arange(len(paired_rows))

    fold_metrics: list[dict] = []
    print(f"\n  LOO ({len(use_idx)} stimuli, train {len(use_idx)-1} / val 1):")
    for fold_i, holdout in enumerate(use_idx):
        tr = use_idx[use_idx != holdout]
        adapter = train_adapter_fold(
            img_n=img_n, perc_n=perc_n, train_idx=tr,
            encoder=encoder, device=device, align_space=align_space, **train_kw,
        )
        m = eval_adapter_fold(
            adapter, img_n=img_n, perc_n=perc_n, eval_idx=np.array([holdout]),
            encoder=encoder, device=device, align_space=align_space,
        )
        row = paired_rows[int(holdout)]
        m["holdout"] = f"set{row['set']}_cue{row['cue']}"
        fold_metrics.append(m)
        if align_space in ("mlp", "hybrid"):
            print(
                f"    holdout {m['holdout']}: repr_cos={m['repr_cosine']:.4f} "
                f"beta_cos={m['beta_cosine']:.4f}"
            )
        else:
            print(f"    holdout {m['holdout']}: beta_cos={m['beta_cosine']:.4f}")

    cosines = [m["cosine"] for m in fold_metrics]
    beta_cosines = [m["beta_cosine"] for m in fold_metrics]
    out = {
        "n_folds": len(fold_metrics),
        "mean_cosine": float(np.mean(cosines)),
        "mean_beta_cosine": float(np.mean(beta_cosines)),
        "folds": fold_metrics,
    }
    if align_space in ("mlp", "hybrid"):
        out["mean_repr_cosine"] = float(np.mean([m["repr_cosine"] for m in fold_metrics]))
    return out


def run_training(
    subj: str,
    root: Path,
    *,
    run_id: str | None = None,
    align_space: str = "mlp",
    dual_run_id: str = DEFAULT_DUAL_RUN,
    hidden: int = 1024,
    dropout: float = 0.1,
    lr: float = 1e-3,
    epochs: int = 200,
    loo_epochs: int = 120,
    batch_size: int = 32,
    temperature: float = 0.07,
    loo_subset: str = "kneeland",
    seed: int = 42,
    beta_weight: float = 0.35,
    init_adapter_run_id: str | None = None,
    exclude_sets_from_fit: frozenset[str] | None = None,
    fit_only_sets: frozenset[str] | None = None,
    skip_loo: bool = False,
) -> dict:
    if run_id:
        os.environ["MINDBRIDGE_RUN_ID"] = run_id
    tag = f"imagery_beta_adapter_{align_space}_{subj}"
    if align_space == "hybrid":
        tag = f"imagery_beta_adapter_hybrid{int(beta_weight * 100):02d}_{subj}"
    if exclude_sets_from_fit:
        tag = f"imagery_beta_adapter_{align_space}_honest_{subj}"
    run_id = resolve_run_id(tag)
    run_base = run_root(root, run_id)
    ckpt_dir = run_base / f"checkpoints_v{VARIANT}" / subj
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_b, perc_b, paired_rows = load_paired_betas(subj, root)
    n_voxels = img_b.shape[1]

    encoder = None
    dual_ckpt_path = None
    if align_space in ("mlp", "hybrid"):
        dual_mlp, vm, vs = load_frozen_dual_mlp(root, subj, dual_run_id, device)
        encoder = dual_mlp.encoder
        dual_ckpt_path = str(
            resolve_variant_checkpoint(root, "4H_CTR2", subj, dual_run_id, prefer="best_retrieval")
        )
    else:
        vm, vs = perception_voxel_stats(subj, root)

    img_n = normalize_raw(img_b, vm, vs)
    perc_n = normalize_raw(perc_b, vm, vs)

    print(f"\n{'='*60}")
    print(f"  Imagery Beta Adapter — {subj}  align={align_space}")
    if fit_only_sets is None and exclude_sets_from_fit == KNEELAND_SETS:
        fit_only_sets = frozenset({"C"})
    fit_idx, holdout_idx, honest_protocol = resolve_honest_fit_indices(
        paired_rows,
        exclude_sets_from_fit=exclude_sets_from_fit,
        fit_only_sets=fit_only_sets,
    )
    print(f"  run={run_id}  device={device}  pairs={len(paired_rows)}")
    if align_space == "hybrid":
        print(f"  hybrid loss: {(1-beta_weight):.2f} MLP BiMixCo + {beta_weight:.2f} beta")
    if dual_ckpt_path:
        print(f"  frozen dual MLP: {dual_ckpt_path}")
    print(f"{'='*60}")

    loo: dict = {"skipped": True}
    if skip_loo:
        print("  LOO skipped (skip_loo=True)")
    else:
        train_kw = dict(
            hidden=hidden, dropout=dropout, lr=lr, epochs=loo_epochs,
            batch_size=batch_size, temperature=temperature, beta_weight=beta_weight,
        )
        loo = loo_evaluate(
            img_n, perc_n, paired_rows,
            encoder=encoder, device=device, align_space=align_space,
            subset=loo_subset, **train_kw,
        )
        print(f"  LOO mean cosine: {loo['mean_cosine']:.4f}  beta_cos: {loo['mean_beta_cosine']:.4f}")
        if "mean_repr_cosine" in loo:
            print(f"  LOO mean repr cosine: {loo['mean_repr_cosine']:.4f}")

    init_state = None
    if init_adapter_run_id:
        init_path = resolve_beta_adapter_path(root, init_adapter_run_id, subj)
        init_ckpt = torch.load(init_path, map_location=device, weights_only=False)
        init_state = init_ckpt["model_state"]
        print(f"  Warm-start from {init_path}")

    np.random.seed(seed)
    all_idx = np.arange(len(img_n))
    fit_idx_arr = np.array(fit_idx, dtype=np.int64)
    adapter = train_adapter_fold(
        img_n=img_n, perc_n=perc_n, train_idx=fit_idx_arr,
        encoder=encoder, device=device, align_space=align_space,
        hidden=hidden, dropout=dropout, lr=lr, epochs=epochs,
        batch_size=batch_size, temperature=temperature, beta_weight=beta_weight,
        init_state=init_state,
    )

    full_metrics = eval_adapter_fold(
        adapter, img_n=img_n, perc_n=perc_n, eval_idx=all_idx,
        encoder=encoder, device=device, align_space=align_space,
    )
    holdout_metrics = None
    if holdout_idx:
        holdout_metrics = eval_adapter_fold(
            adapter, img_n=img_n, perc_n=perc_n, eval_idx=np.array(holdout_idx, dtype=np.int64),
            encoder=encoder, device=device, align_space=align_space,
        )
        print(
            f"  Holdout-only (unseen in final fit) repr_cos="
            f"{holdout_metrics.get('repr_cosine', holdout_metrics['cosine']):.4f} "
            f"beta_cos={holdout_metrics['beta_cosine']:.4f}"
        )

    ckpt_path = ckpt_dir / "best_adapter.pt"
    payload = {
        "model_state": adapter.state_dict(),
        "n_voxels": n_voxels,
        "hidden": hidden,
        "dropout": dropout,
        "voxel_mean": vm,
        "voxel_std": vs,
        "temperature": temperature,
        "align_space": align_space,
        "beta_weight": beta_weight if align_space == "hybrid" else 0.0,
        "init_adapter_run_id": init_adapter_run_id or "",
        "dual_run_id": dual_run_id if align_space in ("mlp", "hybrid") else "",
        "dual_ckpt": dual_ckpt_path or "",
        "n_pairs": len(paired_rows),
        "n_fit_pairs": len(fit_idx),
        "fit_row_indices": fit_idx,
        "holdout_row_indices": holdout_idx,
        "honest_protocol": honest_protocol,
        "exclude_sets_from_fit": sorted(exclude_sets_from_fit) if exclude_sets_from_fit else [],
        "paired_rows": paired_rows,
        "loo": loo,
        "holdout_metrics": holdout_metrics,
        "full_repr_cosine": full_metrics.get("repr_cosine", full_metrics["cosine"]),
        "full_repr_retrieval": full_metrics.get("repr_retrieval", full_metrics["retrieval"]),
        "full_beta_cosine": full_metrics["beta_cosine"],
        "full_beta_retrieval": full_metrics["beta_retrieval"],
        "variant": VARIANT,
        "run_id": run_id,
        "subj": subj,
    }
    torch.save(payload, ckpt_path)
    print(f"\n  Saved {ckpt_path}")
    print(
        f"  Full-set repr_cos={full_metrics.get('repr_cosine', full_metrics['cosine']):.4f} "
        f"beta_cos={full_metrics['beta_cosine']:.4f}"
    )

    results = {
        "subj": subj,
        "run_id": run_id,
        "align_space": align_space,
        "beta_weight": beta_weight if align_space == "hybrid" else None,
        "honest_protocol": honest_protocol,
        "n_fit_pairs": len(fit_idx),
        "loo_mean_cosine": loo.get("mean_cosine"),
        "loo_mean_beta_cosine": loo.get("mean_beta_cosine"),
        "full_repr_cosine": full_metrics.get("repr_cosine", full_metrics["cosine"]),
        "full_repr_retrieval": full_metrics.get("repr_retrieval", full_metrics["retrieval"]),
        "full_beta_cosine": full_metrics["beta_cosine"],
        "full_beta_retrieval": full_metrics["beta_retrieval"],
        "holdout_repr_cosine": (
            holdout_metrics.get("repr_cosine", holdout_metrics["cosine"]) if holdout_metrics else None
        ),
        "holdout_beta_cosine": holdout_metrics["beta_cosine"] if holdout_metrics else None,
        "checkpoint": str(ckpt_path),
        "dual_run_id": dual_run_id,
    }
    update_run_manifest(run_base, run_id=run_id, variant=VARIANT, subjects={subj: results})
    suffix = "_honest" if exclude_sets_from_fit else ""
    out_json = root / "results" / f"imagery_beta_adapter_{align_space}{suffix}_{subj}.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps({**results, "loo": loo}, indent=2))
    return results


def resolve_beta_adapter_path(
    root: Path | str,
    run_id: str,
    subj: str,
    *,
    ckpt_subj: str | None = None,
) -> Path:
    """CLIP adapter (vImgBetaAdapter) or VDVAE adapter (vImgVdvaeBetaAdapter).

    ckpt_subj: subject folder under the run (for cross-subject transfer, pass train subj
    while inference uses another subject's betas + dual model).
    """
    root = Path(root)
    subj_ckpt = ckpt_subj or subj
    for variant in ("vImgVdvaeBetaAdapter", "vImgBetaAdapter"):
        path = root / "runs" / run_id / f"checkpoints_{variant}" / subj_ckpt / "best_adapter.pt"
        if path.exists():
            return path
    raise FileNotFoundError(
        f"No beta adapter at runs/{run_id}/checkpoints_vImg*Adapter/{subj_ckpt}/best_adapter.pt"
    )


def apply_adapter(
    betas: np.ndarray,
    ckpt: dict,
    device: torch.device | None = None,
    *,
    strict_n_voxels: bool = True,
) -> np.ndarray:
    """Imagery betas (raw voxels) → aligned raw betas (train-subject norm + residual map)."""
    device = device or torch.device("cpu")
    n_ckpt = int(ckpt["n_voxels"])
    n_beta = betas.shape[-1]
    if n_beta != n_ckpt:
        msg = f"Adapter n_voxels={n_ckpt} but betas.shape[-1]={n_beta}"
        if strict_n_voxels:
            raise ValueError(msg)
        print(f"  Warning: {msg}")
    vm = np.asarray(ckpt["voxel_mean"]).reshape(1, -1)
    vs = np.asarray(ckpt["voxel_std"]).reshape(1, -1)
    x = normalize_raw(betas, vm, vs)
    model = BetaAdapter(int(ckpt["n_voxels"]), hidden=int(ckpt.get("hidden", 1024)))
    model.load_state_dict(ckpt["model_state"])
    model.eval().to(device)
    with torch.no_grad():
        out_n = model(torch.tensor(x, dtype=torch.float32, device=device))
    return (out_n.cpu().numpy().astype(np.float32) * vs + vm).astype(np.float32)


def apply_adapter_cross_subject(
    betas: np.ndarray,
    ckpt: dict,
    *,
    train_subj: str,
    test_subj: str,
    root: Path | str,
    device: torch.device | None = None,
) -> np.ndarray:
    """
    Apply adapter trained on train_subj to test_subj imagery/perception betas.

    Same n_voxels: direct apply_adapter. Otherwise embed via shared-grid ROI correspondence.
    """
    root = Path(root)
    n_ckpt = int(ckpt["n_voxels"])
    betas = np.asarray(betas, dtype=np.float32)
    squeeze = betas.ndim == 1
    if squeeze:
        betas = betas.reshape(1, -1)

    if betas.shape[1] == n_ckpt:
        out = apply_adapter(betas, ckpt, device=device)
    else:
        from roi_correspondence import embed_to_train_layout, extract_to_test_layout, load_correspondence

        corr = load_correspondence(train_subj, test_subj, root)
        if int(corr["n_train_voxels"]) != n_ckpt:
            raise ValueError(
                f"Adapter n_voxels={n_ckpt} but correspondence n_train={corr['n_train_voxels']}"
            )
        embedded = embed_to_train_layout(betas, corr)
        adapted_train = apply_adapter(embedded, ckpt, device=device)
        out = extract_to_test_layout(adapted_train, corr)
        print(
            f"  Cross-subject adapter {train_subj}→{test_subj}: "
            f"{corr['n_common_voxels']} shared voxels "
            f"({100 * corr['coverage_test']:.1f}% of {test_subj})"
        )

    return out[0] if squeeze else out


def apply_imagery_beta_adapter(
    betas: np.ndarray,
    root: Path | str,
    adapter_run_id: str,
    inference_subj: str,
    *,
    adapter_train_subj: str | None = None,
    device: torch.device | None = None,
) -> np.ndarray:
    """Load adapter checkpoint (from train subject) and apply to inference subject betas."""
    train_subj = adapter_train_subj or inference_subj
    path = resolve_beta_adapter_path(root, adapter_run_id, inference_subj, ckpt_subj=train_subj)
    ckpt = torch.load(path, map_location=device or "cpu", weights_only=False)
    if train_subj == inference_subj:
        return apply_adapter(betas, ckpt, device=device)
    return apply_adapter_cross_subject(
        betas, ckpt, train_subj=train_subj, test_subj=inference_subj, root=root, device=device,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Imagery→perception beta adapter")
    ap.add_argument("--subj", default=os.environ.get("NSD_SUBJ", "subj01"))
    ap.add_argument("--root", default=os.environ.get("MINDBRIDGE_ROOT", "/mnt/mindbridge"))
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--align-space", choices=("mlp", "beta", "hybrid"), default="mlp")
    ap.add_argument("--beta-weight", type=float, default=0.35, help="Hybrid: weight on beta loss")
    ap.add_argument("--init-adapter-run-id", default=None, help="Warm-start from prior adapter run")
    ap.add_argument(
        "--exclude-sets-from-fit",
        default="",
        help="Comma-separated sets excluded from final fit (e.g. A,B for honest Kneeland eval)",
    )
    ap.add_argument("--skip-loo", action="store_true", help="Skip LOO (much faster)")
    ap.add_argument("--dual-run-id", default=DEFAULT_DUAL_RUN)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--loo-epochs", type=int, default=120)
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=0.07)
    args = ap.parse_args()
    excl = frozenset(s.strip() for s in args.exclude_sets_from_fit.split(",") if s.strip())

    run_training(
        args.subj,
        get_root(args.root),
        run_id=args.run_id,
        align_space=args.align_space,
        dual_run_id=args.dual_run_id,
        hidden=args.hidden,
        lr=args.lr,
        epochs=args.epochs,
        loo_epochs=args.loo_epochs,
        batch_size=args.batch_size,
        temperature=args.temperature,
        beta_weight=args.beta_weight,
        init_adapter_run_id=args.init_adapter_run_id,
        exclude_sets_from_fit=excl or None,
        fit_only_sets=frozenset({"C"}) if excl == KNEELAND_SETS else None,
        skip_loo=args.skip_loo,
    )


if __name__ == "__main__":
    main()
