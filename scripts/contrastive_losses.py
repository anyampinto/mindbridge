# -*- coding: utf-8 -*-
"""BiMixCo, SoftCLIP, multi-token contrastive losses."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def mixco(
    x: torch.Tensor,
    beta: float = 0.15,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """MixCo: convex combo with Beta(beta,beta) coefficients (MindEye BiMixCo)."""
    b = x.size(0)
    perm = torch.randperm(b, device=x.device)
    lam = torch.from_numpy(
        np.random.beta(beta, beta, b).astype(np.float32),
    ).to(x.device)
    while (lam < 0.05).any() or (lam > 0.95).any():
        lam = torch.from_numpy(
            np.random.beta(beta, beta, b).astype(np.float32),
        ).to(x.device)
    if x.dim() == 3:
        lam_v = lam.view(b, 1, 1)
    else:
        lam_v = lam.view(b, 1)
    mixed = lam_v * x + (1.0 - lam_v) * x[perm]
    return mixed, perm, lam


def _sim_matrix(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if pred.dim() == 3:
        return (pred.unsqueeze(1) * target.unsqueeze(0)).sum(-1).mean(-1)
    p = F.normalize(pred.float(), dim=-1)
    t = F.normalize(target.float(), dim=-1)
    return p @ t.T


def bidirectional_clip_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    logits = _sim_matrix(pred, target) / temperature
    labels = torch.arange(pred.size(0), device=pred.device)
    return (
        F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)
    ) * 0.5


def soft_clip_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    logits = _sim_matrix(pred, target) / temperature
    if target.dim() == 3:
        soft = multi_token_soft_targets(target)
    else:
        t = F.normalize(target.float(), dim=-1)
        soft = (t @ t.T).clamp(min=0)
        soft = soft / soft.sum(dim=-1, keepdim=True).clamp(min=1e-8)

    log_p_i2t = F.log_softmax(logits, dim=-1)
    loss_i2t = -(soft * log_p_i2t).sum(dim=-1).mean()

    logits_t = logits.T
    soft_t = soft.T
    soft_t = soft_t / soft_t.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    log_p_t2i = F.log_softmax(logits_t, dim=-1)
    loss_t2i = -(soft_t * log_p_t2i).sum(dim=-1).mean()
    return (loss_i2t + loss_t2i) * 0.5


def multi_token_soft_targets(target: torch.Tensor) -> torch.Tensor:
    """(B, T, D) → (B, B) soft label matrix (mean token cos)."""
    sim = _sim_matrix(target, target).clamp(min=0)
    return sim / sim.sum(dim=-1, keepdim=True).clamp(min=1e-8)


def bimixco_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    temperature: float = 0.07,
    mix_beta: float = 0.15,
) -> torch.Tensor:
    """Bidirectional CLIP on MixCo-augmented pred and target."""
    pm, _, _ = mixco(pred, mix_beta)
    tm, _, _ = mixco(target, mix_beta)
    pm = F.normalize(pm, dim=-1) if pm.dim() == 2 else F.normalize(pm, dim=-1)
    tm = F.normalize(tm, dim=-1) if tm.dim() == 2 else F.normalize(tm, dim=-1)
    return bidirectional_clip_loss(pm, tm, temperature)


class TargetMemoryBank:
    """FIFO bank of normalized CLIP targets for extra contrastive negatives."""

    def __init__(self, dim: int, capacity: int = 4096, n_tokens: int = 1):
        self.capacity = capacity
        self.n_tokens = n_tokens
        self.dim = dim
        self.ptr = 0
        self.full = False
        if n_tokens > 1:
            self.bank = torch.zeros(capacity, n_tokens, dim)
        else:
            self.bank = torch.zeros(capacity, dim)

    @torch.no_grad()
    def enqueue(self, targets: torch.Tensor) -> None:
        t = targets.detach().cpu()
        if t.dim() == 2:
            t = F.normalize(t, dim=-1)
        else:
            t = F.normalize(t, dim=-1)
        n = t.size(0)
        end = self.ptr + n
        if end <= self.capacity:
            self.bank[self.ptr:end] = t
        else:
            first = self.capacity - self.ptr
            self.bank[self.ptr:] = t[:first]
            self.bank[: n - first] = t[first:]
            self.full = True
        self.ptr = end % self.capacity

    def get(self, device: torch.device) -> torch.Tensor:
        n = self.capacity if self.full else self.ptr
        if n == 0:
            return torch.empty(0, device=device)
        return self.bank[:n].to(device)

    def contrastive_with_bank(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        temperature: float,
        use_softclip: bool,
    ) -> torch.Tensor:
        bank = self.get(pred.device)
        if bank.numel() == 0:
            if use_softclip:
                return soft_clip_loss(pred, target, temperature)
            return bidirectional_clip_loss(pred, target, temperature)

        if pred.dim() == 3:
            all_t = torch.cat([target, bank], dim=0)
        else:
            all_t = torch.cat([target, bank], dim=0)
        labels = torch.arange(pred.size(0), device=pred.device)
        logits = _sim_matrix(pred, all_t) / temperature
        loss_i2t = F.cross_entropy(logits, labels)
        logits_t = logits[:, : pred.size(0)].T
        loss_t2i = F.cross_entropy(logits_t, labels)
        loss = (loss_i2t + loss_t2i) * 0.5
        if use_softclip:
            soft = multi_token_soft_targets(target) if target.dim() == 3 else (
                F.normalize(target, dim=-1) @ F.normalize(target, dim=-1).T
            ).clamp(min=0)
            soft = soft / soft.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            log_p = F.log_softmax(_sim_matrix(pred, target) / temperature, dim=-1)
            loss = loss + 0.5 * (-(soft * log_p).sum(dim=-1).mean())
        return loss
