# -*- coding: utf-8 -*-
"""Retrieval metrics: raw cosine 2-way, inverted-softmax, multi-token similarity."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def _normalize_pred_target(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    pred = pred.detach().cpu().float()
    target = target.detach().cpu().float()
    if pred.dim() == 3:
        pred = F.normalize(pred, dim=-1)
        target = F.normalize(target, dim=-1)
    else:
        pred = F.normalize(pred, dim=-1)
        target = F.normalize(target, dim=-1)
    return pred, target


def multi_token_similarity(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """(N, T, D) normalized → (N, N) mean token cosine."""
    return (pred.unsqueeze(1) * target.unsqueeze(0)).sum(-1).mean(-1)


def pairwise_query_gallery_sim(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Query-gallery matrix (N, N): sim(q_i, t_j)."""
    if pred.dim() == 3:
        return multi_token_similarity(pred, target)
    return pred @ target.T


def retrieval_2way(
    pred: torch.Tensor,
    target: torch.Tensor,
    n_pairs: int = 10000,
    seed: int = 42,
) -> float:
    pred, target = _normalize_pred_target(pred, target)
    sim = pairwise_query_gallery_sim(pred, target)
    n = sim.size(0)
    if n < 2:
        return 0.0
    rng = np.random.default_rng(seed)
    i = rng.integers(0, n, size=n_pairs)
    j = rng.integers(0, n, size=n_pairs)
    collide = i == j
    while collide.any():
        j[collide] = rng.integers(0, n, size=int(collide.sum()))
        collide = i == j
    correct = sim[i, i] > sim[i, j]
    return float(correct.float().mean().item())


def retrieval_2way_inverted_softmax(
    pred: torch.Tensor,
    target: torch.Tensor,
    n_pairs: int = 10000,
    seed: int = 42,
    alpha: float = 1.0,
) -> float:
    """
    2-way with inverted-softmax correction (Radford / cross-modal retrieval).
    score(q, d) ∝ exp(sim(q,d)/α) / sum_{d'} exp(sim(d', d)/α)
    """
    pred, target = _normalize_pred_target(pred, target)
    sim_qd = pairwise_query_gallery_sim(pred, target) / alpha
    if target.dim() == 3:
        gallery_self = multi_token_similarity(target, target) / alpha
    else:
        gallery_self = (target @ target.T) / alpha
    log_denom = torch.logsumexp(gallery_self, dim=0)
    corrected = sim_qd - log_denom.unsqueeze(0)

    n = corrected.size(0)
    if n < 2:
        return 0.0
    rng = np.random.default_rng(seed)
    i = rng.integers(0, n, size=n_pairs)
    j = rng.integers(0, n, size=n_pairs)
    collide = i == j
    while collide.any():
        j[collide] = rng.integers(0, n, size=int(collide.sum()))
        collide = i == j
    correct = corrected[i, i] > corrected[i, j]
    return float(correct.float().mean().item())


def retrieval_2way_sinkhorn(
    pred: torch.Tensor,
    target: torch.Tensor,
    n_pairs: int = 10000,
    seed: int = 42,
    n_iters: int = 3,
    reg: float = 0.05,
) -> float:
    """2-way after Sinkhorn-style row/column normalization of similarity matrix."""
    pred, target = _normalize_pred_target(pred, target)
    sim = pairwise_query_gallery_sim(pred, target).clamp(min=1e-8)
    log_sim = torch.log(sim)
    u = torch.zeros(sim.size(0))
    v = torch.zeros(sim.size(1))
    for _ in range(n_iters):
        u = -torch.logsumexp(log_sim + v.unsqueeze(0), dim=1)
        v = -torch.logsumexp(log_sim + u.unsqueeze(1), dim=0)
    corrected = log_sim + u.unsqueeze(1) + v.unsqueeze(0)

    n = corrected.size(0)
    if n < 2:
        return 0.0
    rng = np.random.default_rng(seed)
    i = rng.integers(0, n, size=n_pairs)
    j = rng.integers(0, n, size=n_pairs)
    collide = i == j
    while collide.any():
        j[collide] = rng.integers(0, n, size=int(collide.sum()))
        collide = i == j
    correct = corrected[i, i] > corrected[i, j]
    return float(correct.float().mean().item())


def retrieval_2way_kneeland_subset(
    pred_query: torch.Tensor,
    target_gallery: torch.Tensor,
    gallery_positions: list[int],
    n_pairs: int = 1000,
    seed: int = 42,
) -> dict:
    """
    Kneeland 2WC when only a subset of gallery positions have brain/recon predictions.

    pred_query: (n_q, d) — one row per query stimulus
    target_gallery: (n_g, d) — full imagery gallery (e.g. 12 Kneeland stimuli)
    gallery_positions: query k uses target_gallery[gallery_positions[k]]
    Distractors j are uniform from all gallery indices != gallery_positions[k].
    """
    pred_query, target_gallery = _normalize_pred_target(pred_query, target_gallery)
    n_q = pred_query.size(0)
    n_g = target_gallery.size(0)
    if n_g < 2 or n_q < 1:
        return {"kneeland_2wc": float("nan"), "chance": 0.5, "n_pairs": n_pairs, "n_stimuli": n_q}

    rng = np.random.default_rng(seed)
    correct = []
    for _ in range(n_pairs):
        k = int(rng.integers(0, n_q))
        gi = int(gallery_positions[k])
        distractors = [j for j in range(n_g) if j != gi]
        j = int(rng.choice(distractors))
        sim_i = (pred_query[k] * target_gallery[gi]).sum()
        sim_j = (pred_query[k] * target_gallery[j]).sum()
        correct.append(sim_i > sim_j)
    acc = float(np.mean(correct))
    return {
        "kneeland_2wc": acc,
        "chance": 0.5,
        "n_pairs": n_pairs,
        "n_stimuli": n_q,
        "n_gallery": n_g,
        "distractor_pool": "other_imagery_stimuli_in_gallery",
        "protocol": "legacy_gt_distractor",
    }


def retrieval_2way_kneeland_paper(
    recon_gallery: torch.Tensor,
    gt_gallery: torch.Tensor,
    query_positions: list[int] | None = None,
    n_pairs: int = 1000,
    seed: int = 42,
) -> dict:
    """
    Kneeland Appendix A.2 reconstruction 2WC (chance = 50%).

    For each sampled query gallery index i, pick distractor j != i (uniform over gallery).
    Success when cos(recon_i, gt_i) > cos(recon_j, gt_i) — GT is the anchor; distractor is
    another stimulus's *reconstruction*, not another GT image.
    """
    recon_gallery, gt_gallery = _normalize_pred_target(recon_gallery, gt_gallery)
    n_g = recon_gallery.size(0)
    if n_g < 2:
        return {
            "kneeland_2wc": float("nan"),
            "chance": 0.5,
            "n_pairs": n_pairs,
            "n_stimuli": 0,
            "n_gallery": n_g,
            "distractor_pool": "other_reconstructions_in_gallery",
            "protocol": "paper_recon_distractor",
        }
    qpos = list(range(n_g)) if query_positions is None else list(query_positions)
    if not qpos:
        return {
            "kneeland_2wc": float("nan"),
            "chance": 0.5,
            "n_pairs": n_pairs,
            "n_stimuli": 0,
            "n_gallery": n_g,
            "distractor_pool": "other_reconstructions_in_gallery",
            "protocol": "paper_recon_distractor",
        }

    rng = np.random.default_rng(seed)
    correct = []
    for _ in range(n_pairs):
        i = int(rng.choice(qpos))
        distractors = [j for j in range(n_g) if j != i]
        j = int(rng.choice(distractors))
        sim_target = (recon_gallery[i] * gt_gallery[i]).sum()
        sim_distractor = (recon_gallery[j] * gt_gallery[i]).sum()
        correct.append(sim_target > sim_distractor)
    acc = float(np.mean(correct))
    return {
        "kneeland_2wc": acc,
        "chance": 0.5,
        "n_pairs": n_pairs,
        "n_stimuli": len(qpos),
        "n_gallery": n_g,
        "distractor_pool": "other_reconstructions_in_gallery",
        "protocol": "paper_recon_distractor",
    }


def retrieval_2way_kneeland_paper_multisample(
    recon_samples: torch.Tensor,
    gt_gallery: torch.Tensor,
    query_positions: list[int] | None = None,
    n_pairs: int = 1000,
    seed: int = 42,
) -> dict:
    """
    Paper protocol with K stochastic reconstructions per gallery stimulus (n_g, K, d).

    Each of n_pairs draws: query index i, sample index si, distractor j != i, sample sj.
    """
    if recon_samples.dim() != 3:
        raise ValueError("recon_samples must be (n_gallery, K, dim)")
    n_g, k, _ = recon_samples.size()
    recon_samples = F.normalize(recon_samples.detach().cpu().float(), dim=-1)
    gt_gallery = F.normalize(gt_gallery.detach().cpu().float(), dim=-1)
    qpos = list(range(n_g)) if query_positions is None else list(query_positions)
    if n_g < 2 or k < 1 or not qpos:
        return {
            "kneeland_2wc": float("nan"),
            "chance": 0.5,
            "n_pairs": n_pairs,
            "n_stimuli": len(qpos),
            "n_gallery": n_g,
            "n_samples_per_stim": k,
            "distractor_pool": "other_reconstructions_in_gallery",
            "protocol": "paper_recon_distractor_multisample",
        }

    rng = np.random.default_rng(seed)
    correct = []
    for _ in range(n_pairs):
        i = int(rng.choice(qpos))
        distractors = [j for j in range(n_g) if j != i]
        j = int(rng.choice(distractors))
        si = int(rng.integers(0, k))
        sj = int(rng.integers(0, k))
        sim_target = (recon_samples[i, si] * gt_gallery[i]).sum()
        sim_distractor = (recon_samples[j, sj] * gt_gallery[i]).sum()
        correct.append(sim_target > sim_distractor)
    acc = float(np.mean(correct))
    return {
        "kneeland_2wc": acc,
        "chance": 0.5,
        "n_pairs": n_pairs,
        "n_stimuli": len(qpos),
        "n_gallery": n_g,
        "n_samples_per_stim": k,
        "distractor_pool": "other_reconstructions_in_gallery",
        "protocol": "paper_recon_distractor_multisample",
    }


def retrieval_2way_kneeland(
    pred: torch.Tensor,
    target: torch.Tensor,
    n_pairs: int = 1000,
    seed: int = 42,
) -> dict:
    """
    Kneeland Table 1 CLIP column: 2-way forced choice (chance = 50%).

    For each sampled query i, draw distractor j uniformly from the other N-1
    imagery stimuli (same gallery as target). Success when
    cos(pred_i, gt_i) > cos(pred_i, gt_j). Average over n_pairs samples.

    Requires pred and target aligned on the same N imagery conditions (e.g. Set A+B, N=12).
    Distractors must NOT come from perception val images.
    """
    pred, target = _normalize_pred_target(pred, target)
    n = pred.size(0)
    acc = retrieval_2way(pred, target, n_pairs=n_pairs, seed=seed)
    return {
        "kneeland_2wc": acc,
        "chance": 0.5,
        "n_pairs": n_pairs,
        "n_stimuli": n,
        "distractor_pool": "other_imagery_stimuli_in_gallery",
        "protocol": "legacy_gt_distractor",
    }


def retrieval_nway(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> dict:
    """
    N-way identification (NOT Kneeland Table 1 — chance = 1/N, e.g. 8.3% for N=12).

    For each query i, success when cos(pred_i, target_i) exceeds cos(pred_i, target_j)
    for all j != i. With N stimuli, each query has N-1 in-set distractors (e.g. 11 when N=12).
    Chance accuracy = 1/N.
    """
    pred, target = _normalize_pred_target(pred, target)
    sim = pairwise_query_gallery_sim(pred, target)
    n = sim.size(0)
    if n < 2:
        return {
            "n_way_acc": float("nan"),
            "n_way_correct": 0,
            "n_stimuli": n,
            "chance": 1.0 / max(n, 1),
            "per_stim_correct": [],
        }
    correct = sim.argmax(dim=1) == torch.arange(n)
    per = correct.tolist()
    return {
        "n_way_acc": float(correct.float().mean().item()),
        "n_way_correct": int(correct.sum().item()),
        "n_stimuli": n,
        "chance": 1.0 / n,
        "per_stim_correct": per,
    }
