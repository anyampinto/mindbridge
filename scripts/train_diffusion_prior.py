# -*- coding: utf-8 -*-
"""
Train diffusion prior: 1024-d image projector → 768-d CLIP (perception trials).

Usage:
  python train_diffusion_prior.py --subj subj01 --root /mnt/mindbridge
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
sys.path = [p for p in sys.path if "share/software" not in p and "jupyterlab" not in p]

import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

from diffusion_prior_io import (
    ALL_SUBJECTS,
    DEFAULT_DUAL_RUN,
    POOLED_PRIOR_SUBJ,
    cache_perception_projector_pairs,
    dual_run_id_for_subj,
    load_trial_splits,
    prior_ckpt_path,
)
from models.diffusion_prior import DDPMSchedule, DiffusionPrior
from paths import get_root, resolve_run_id, run_root, update_run_manifest

VARIANT = "DiffusionPrior"


def mean_cosim(pred: torch.Tensor, tgt: torch.Tensor) -> float:
    p = F.normalize(pred.float(), dim=-1)
    t = F.normalize(tgt.float(), dim=-1)
    return float((p * t).sum(dim=-1).mean().item())


@torch.no_grad()
def val_epoch(
    prior: DiffusionPrior,
    schedule: DDPMSchedule,
    loader: DataLoader,
    device: torch.device,
    ddim_steps: int,
) -> float:
    from models.diffusion_prior import ddim_sample

    prior.eval()
    preds, tgts = [], []
    for proj, clip in loader:
        proj = proj.to(device)
        clip = clip.to(device)
        pred = ddim_sample(prior, schedule, proj, ddim_steps=ddim_steps)
        preds.append(pred.cpu())
        tgts.append(clip.cpu())
    return mean_cosim(torch.cat(preds), torch.cat(tgts))


def run_training(
    subj: str,
    root: Path,
    *,
    dual_run_id: str = DEFAULT_DUAL_RUN,
    run_id: str | None = None,
    epochs: int = 150,
    batch_size: int = 128,
    lr: float = 3e-4,
    num_timesteps: int = 1000,
    ddim_steps: int = 50,
    eval_every: int = 10,
    seed: int = 42,
    force_cache: bool = False,
) -> dict:
    if run_id:
        os.environ["MINDBRIDGE_RUN_ID"] = run_id
    run_id = resolve_run_id(f"diffusion_prior_{subj}")
    run_base = run_root(root, run_id)
    ckpt_dir = run_base / "checkpoints_prior" / subj
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    legacy_ckpt = root / "checkpoints_prior" / subj
    legacy_ckpt.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache_perception_projector_pairs(
        subj, root, dual_run_id=dual_run_id, force=force_cache,
    )
    proj = np.load(root / "nsd_meta" / f"proj_embs_{subj}.npy")
    clip = np.load(root / "nsd_meta" / f"clip_embs_trials_{subj}.npy")
    tr_idx, vl_idx, _ = load_trial_splits(subj, root, seed=seed)

    print(f"\n{'='*60}")
    print(f"  Diffusion prior — {subj}  run={run_id}")
    print(f"  trials={len(proj)}  train={len(tr_idx)}  val={len(vl_idx)}  device={device}")
    print(f"{'='*60}")

    train_ds = TensorDataset(
        torch.tensor(proj[tr_idx], dtype=torch.float32),
        torch.tensor(clip[tr_idx], dtype=torch.float32),
    )
    val_ds = TensorDataset(
        torch.tensor(proj[vl_idx], dtype=torch.float32),
        torch.tensor(clip[vl_idx], dtype=torch.float32),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    prior = DiffusionPrior().to(device)
    schedule = DDPMSchedule(num_timesteps=num_timesteps).to(device)
    opt = torch.optim.AdamW(prior.parameters(), lr=lr, weight_decay=1e-4)

    best_cos = -1.0
    history: list[dict] = []

    for epoch in range(1, epochs + 1):
        prior.train()
        total_loss = 0.0
        n_batches = 0
        for proj_b, clip_b in train_loader:
            proj_b = proj_b.to(device)
            clip_b = F.normalize(clip_b.to(device), dim=-1)
            b = clip_b.size(0)
            t = torch.randint(0, schedule.num_timesteps, (b,), device=device)
            noise = torch.randn_like(clip_b)
            x_t = schedule.q_sample(clip_b, t, noise)
            pred_x0 = prior(x_t, t, proj_b)
            loss = F.mse_loss(pred_x0, clip_b)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(prior.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        row = {"epoch": epoch, "train_mse": avg_loss}

        if epoch % eval_every == 0 or epoch == epochs:
            val_cos = val_epoch(prior, schedule, val_loader, device, ddim_steps)
            row["val_cosim_ddim"] = val_cos
            print(f"  ep {epoch:3d}  train_mse={avg_loss:.5f}  val_cosim={val_cos:.4f}")
            if val_cos > best_cos:
                best_cos = val_cos
                payload = {
                    "model_state": prior.state_dict(),
                    "epoch": epoch,
                    "val_cosim": val_cos,
                    "num_timesteps": num_timesteps,
                    "ddim_steps": ddim_steps,
                    "dual_run_id": dual_run_id,
                    "subj": subj,
                    "run_id": run_id,
                    "n_train": len(tr_idx),
                    "n_val": len(vl_idx),
                }
                for path in (ckpt_dir / "best_prior.pt", legacy_ckpt / "best_prior.pt"):
                    torch.save(payload, path)
                print(f"    → saved best_prior.pt (val_cosim={val_cos:.4f})")
        elif epoch % 5 == 0:
            print(f"  ep {epoch:3d}  train_mse={avg_loss:.5f}")

        history.append(row)

    results = {
        "subj": subj,
        "run_id": run_id,
        "best_val_cosim": best_cos,
        "checkpoint": str(ckpt_dir / "best_prior.pt"),
        "legacy_checkpoint": str(legacy_ckpt / "best_prior.pt"),
        "dual_run_id": dual_run_id,
        "epochs": epochs,
        "history": history[-20:],
    }
    update_run_manifest(run_base, run_id=run_id, variant=VARIANT, subjects={subj: results})
    out_json = root / "results" / f"diffusion_prior_{subj}.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results, indent=2))
    print(f"\n  Saved {out_json}")
    return results


def run_training_pooled(
    root: Path,
    *,
    dual_run_id: str = DEFAULT_DUAL_RUN,
    run_id: str | None = None,
    subjects: list[str] | None = None,
    epochs: int = 150,
    batch_size: int = 128,
    lr: float = 3e-4,
    num_timesteps: int = 1000,
    ddim_steps: int = 50,
    eval_every: int = 10,
    seed: int = 42,
    val_frac: float = 0.1,
    force_cache: bool = False,
) -> dict:
    """One diffusion prior on pooled perception (proj, CLIP) from all subjects."""
    if run_id:
        os.environ["MINDBRIDGE_RUN_ID"] = run_id
    run_id = resolve_run_id("diffusion_prior_all_subjects")
    run_base = run_root(root, run_id)
    subj_tag = POOLED_PRIOR_SUBJ
    ckpt_dir = run_base / "checkpoints_prior" / subj_tag
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    legacy_ckpt = root / "checkpoints_prior" / subj_tag
    legacy_ckpt.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    subjects = subjects or ALL_SUBJECTS
    proj_parts, clip_parts = [], []
    used_subjects: list[str] = []

    for subj in subjects:
        rid = dual_run_id_for_subj(dual_run_id, subj)
        try:
            cache_perception_projector_pairs(
                subj, root, dual_run_id=rid, force=force_cache,
            )
        except FileNotFoundError as e:
            print(f"  [{subj}] skip prior cache: {e}")
            continue
        proj_parts.append(np.load(root / "nsd_meta" / f"proj_embs_{subj}.npy"))
        clip_parts.append(np.load(root / "nsd_meta" / f"clip_embs_trials_{subj}.npy"))
        used_subjects.append(subj)

    if not proj_parts:
        raise RuntimeError("No subjects with cached projector pairs for pooled prior training")

    proj = np.concatenate(proj_parts, axis=0)
    clip = np.concatenate(clip_parts, axis=0)
    n = len(proj)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_val = max(1, int(val_frac * n))
    vl_idx = perm[:n_val]
    tr_idx = perm[n_val:]

    print(f"\n{'='*60}")
    print(f"  Pooled diffusion prior — subjects={used_subjects}")
    print(f"  run={run_id}  trials={n}  train={len(tr_idx)}  val={len(vl_idx)}")
    print(f"{'='*60}")

    train_ds = TensorDataset(
        torch.tensor(proj[tr_idx], dtype=torch.float32),
        torch.tensor(clip[tr_idx], dtype=torch.float32),
    )
    val_ds = TensorDataset(
        torch.tensor(proj[vl_idx], dtype=torch.float32),
        torch.tensor(clip[vl_idx], dtype=torch.float32),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    prior = DiffusionPrior().to(device)
    schedule = DDPMSchedule(num_timesteps=num_timesteps).to(device)
    opt = torch.optim.AdamW(prior.parameters(), lr=lr, weight_decay=1e-4)
    best_cos = -1.0
    history: list[dict] = []

    for epoch in range(1, epochs + 1):
        prior.train()
        total_loss = 0.0
        n_batches = 0
        for proj_b, clip_b in train_loader:
            proj_b = proj_b.to(device)
            clip_b = F.normalize(clip_b.to(device), dim=-1)
            b = clip_b.size(0)
            t = torch.randint(0, schedule.num_timesteps, (b,), device=device)
            noise = torch.randn_like(clip_b)
            x_t = schedule.q_sample(clip_b, t, noise)
            pred_x0 = prior(x_t, t, proj_b)
            loss = F.mse_loss(pred_x0, clip_b)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(prior.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        row = {"epoch": epoch, "train_mse": avg_loss}
        if epoch % eval_every == 0 or epoch == epochs:
            val_cos = val_epoch(prior, schedule, val_loader, device, ddim_steps)
            row["val_cosim_ddim"] = val_cos
            print(f"  ep {epoch:3d}  train_mse={avg_loss:.5f}  val_cosim={val_cos:.4f}")
            if val_cos > best_cos:
                best_cos = val_cos
                payload = {
                    "model_state": prior.state_dict(),
                    "epoch": epoch,
                    "val_cosim": val_cos,
                    "num_timesteps": num_timesteps,
                    "ddim_steps": ddim_steps,
                    "dual_run_id": dual_run_id,
                    "subj": subj_tag,
                    "subjects": used_subjects,
                    "run_id": run_id,
                    "n_train": len(tr_idx),
                    "n_val": len(vl_idx),
                    "n_trials": n,
                }
                for path in (ckpt_dir / "best_prior.pt", legacy_ckpt / "best_prior.pt"):
                    torch.save(payload, path)
                print(f"    → saved best_prior.pt (val_cosim={val_cos:.4f})")
        elif epoch % 5 == 0:
            print(f"  ep {epoch:3d}  train_mse={avg_loss:.5f}")
        history.append(row)

    results = {
        "subj": subj_tag,
        "subjects": used_subjects,
        "run_id": run_id,
        "best_val_cosim": best_cos,
        "checkpoint": str(ckpt_dir / "best_prior.pt"),
        "legacy_checkpoint": str(legacy_ckpt / "best_prior.pt"),
        "dual_run_id": dual_run_id,
        "epochs": epochs,
        "history": history[-20:],
    }
    update_run_manifest(run_base, run_id=run_id, variant=VARIANT, subjects={subj_tag: results})
    out_json = root / "results" / f"diffusion_prior_{subj_tag}.json"
    out_json.write_text(json.dumps(results, indent=2))
    print(f"\n  Saved {out_json}")
    return results


def run_training_all(
    root: Path,
    *,
    dual_run_id: str = DEFAULT_DUAL_RUN,
    run_id_prefix: str = "",
    epochs: int = 150,
    batch_size: int = 128,
    subjects: list[str] | None = None,
    force_cache: bool = False,
) -> dict:
    """Train one diffusion prior per subject (parallel-friendly via Modal map)."""
    subjects = subjects or ALL_SUBJECTS
    completed: dict = {}
    failed: dict = {}
    for subj in subjects:
        rid = dual_run_id_for_subj(dual_run_id, subj) if dual_run_id else ""
        per_run = run_id_prefix or None
        try:
            completed[subj] = run_training(
                subj,
                root,
                dual_run_id=rid or DEFAULT_DUAL_RUN,
                run_id=per_run,
                epochs=epochs,
                batch_size=batch_size,
                force_cache=force_cache,
            )
        except Exception as e:
            failed[subj] = str(e)
            print(f"  [{subj}] prior training failed: {e}")
    summary = {"completed": completed, "failed": failed}
    out = root / "results" / "diffusion_prior_all_subjects_per_subj.json"
    out.write_text(json.dumps(summary, indent=2, default=str))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Train CLIP diffusion prior on perception trials")
    ap.add_argument("--subj", default=os.environ.get("NSD_SUBJ", "subj01"))
    ap.add_argument("--root", default=os.environ.get("MINDBRIDGE_ROOT", "/mnt/mindbridge"))
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--dual-run-id", default=DEFAULT_DUAL_RUN)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--force-cache", action="store_true")
    ap.add_argument("--pooled", action="store_true", help="Train one prior on all subjects")
    ap.add_argument("--all-subjects", action="store_true", help="Train per-subject prior for subj01-08")
    args = ap.parse_args()

    root = get_root(args.root)
    if args.pooled:
        run_training_pooled(
            root,
            dual_run_id=args.dual_run_id,
            run_id=args.run_id,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            force_cache=args.force_cache,
        )
    elif args.all_subjects:
        run_training_all(
            root,
            dual_run_id=args.dual_run_id,
            run_id_prefix=args.run_id or "",
            epochs=args.epochs,
            batch_size=args.batch_size,
            force_cache=args.force_cache,
        )
    else:
        run_training(
            args.subj,
            root,
            dual_run_id=args.dual_run_id,
            run_id=args.run_id,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            force_cache=args.force_cache,
        )


if __name__ == "__main__":
    main()
