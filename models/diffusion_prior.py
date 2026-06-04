# -*- coding: utf-8 -*-
"""Diffusion prior: 1024-d image projector conditioning → 768-d CLIP image manifold."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """(B,) int64 timesteps → (B, dim)."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, device=timesteps.device, dtype=torch.float32) / half
    )
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class DDPMSchedule:
    """Linear beta schedule (1000 steps), DDPM/DDIM helpers."""

    def __init__(self, num_timesteps: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02):
        betas = torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float64)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        self.num_timesteps = num_timesteps
        self.betas = betas.float()
        self.alphas = alphas.float()
        self.alpha_bar = alpha_bar.float()
        self.sqrt_alpha_bar = torch.sqrt(alpha_bar).float()
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar).float()
        self.sqrt_recip_alpha_bar = torch.sqrt(1.0 / alpha_bar).float()
        self.sqrt_recipm1_alpha_bar = torch.sqrt(1.0 / alpha_bar - 1.0).float()

    def to(self, device: torch.device) -> DDPMSchedule:
        for name in ("betas", "alphas", "alpha_bar", "sqrt_alpha_bar", "sqrt_one_minus_alpha_bar",
                     "sqrt_recip_alpha_bar", "sqrt_recipm1_alpha_bar"):
            setattr(self, name, getattr(self, name).to(device))
        return self

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Forward diffusion: x_t = sqrt(ab)*x0 + sqrt(1-ab)*noise."""
        sa = self.sqrt_alpha_bar[t].view(-1, 1)
        so = self.sqrt_one_minus_alpha_bar[t].view(-1, 1)
        return sa * x0 + so * noise


class PriorTransformerLayer(nn.Module):
    def __init__(self, hidden_dim: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.norm4 = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """x, cond: (B, 1, H)."""
        h, _ = self.self_attn(x, x, x)
        x = self.norm1(x + h)
        h, _ = self.cross_attn(x, cond, cond)
        x = self.norm2(x + h)
        x = self.norm3(x + self.ffn(x))
        return x


class DiffusionPrior(nn.Module):
    """
    Denoising transformer: noisy 768-d CLIP ← conditioned on 1024-d projector embedding.

    Predicts clean x0 (CLIP); training loss is MSE(pred_x0, true_clip).
    """

    def __init__(
        self,
        input_dim: int = 1024,
        output_dim: int = 768,
        hidden_dim: int = 512,
        n_layers: int = 4,
        time_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.cond_in = nn.Linear(input_dim, hidden_dim)
        self.x_in = nn.Linear(output_dim, hidden_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.layers = nn.ModuleList([
            PriorTransformerLayer(hidden_dim, dropout=dropout) for _ in range(n_layers)
        ])
        self.out = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, output_dim))
        self.time_dim = time_dim

    def forward(self, x_noisy: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        x_noisy: (B, 768) noisy CLIP
        t: (B,) timesteps
        cond: (B, 1024) projector embedding (L2-normalized)
        """
        t_emb = sinusoidal_timestep_embedding(t, self.time_dim)
        c = self.cond_in(cond).unsqueeze(1)
        h = self.x_in(x_noisy).unsqueeze(1) + self.time_mlp(t_emb).unsqueeze(1)
        for layer in self.layers:
            h = layer(h, c)
        return self.out(h.squeeze(1))


def ddim_sample_train(
    prior: DiffusionPrior,
    schedule: DDPMSchedule,
    cond: torch.Tensor,
    *,
    ddim_steps: int = 25,
    eta: float = 0.0,
) -> torch.Tensor:
    """
    Differentiable DDIM for training (gradients flow into cond / prior if unfrozen).
    No inference temperature noise — returns L2-normalized CLIP-768.
    """
    device = cond.device
    schedule = schedule.to(device)
    b = cond.size(0)
    x = torch.randn(b, prior.output_dim, device=device, dtype=cond.dtype)
    times = torch.linspace(
        schedule.num_timesteps - 1, 0, ddim_steps, device=device,
    ).long().tolist()

    for i, t_cur in enumerate(times):
        t_batch = torch.full((b,), t_cur, device=device, dtype=torch.long)
        pred_x0 = prior(x, t_batch, cond)
        if i == len(times) - 1:
            x = pred_x0
            break
        t_next = times[i + 1]
        alpha_cur = schedule.alpha_bar[t_cur]
        alpha_next = schedule.alpha_bar[t_next]
        sigma = (
            eta
            * torch.sqrt((1 - alpha_next) / (1 - alpha_cur) * (1 - alpha_cur / alpha_next))
        )
        dir_xt = torch.sqrt(1 - alpha_next - sigma**2) * pred_x0
        x = torch.sqrt(alpha_next) * pred_x0 + dir_xt + sigma * torch.randn_like(x)

    return F.normalize(x.float(), dim=-1)


@torch.no_grad()
def ddim_sample(
    prior: DiffusionPrior,
    schedule: DDPMSchedule,
    cond: torch.Tensor,
    *,
    ddim_steps: int = 50,
    eta: float = 0.0,
    temperature: float = 1.0,
    seed: int | None = None,
) -> torch.Tensor:
    """
    DDIM from Gaussian noise → clean 768-d CLIP, conditioned on projector emb.

    cond: (B, 1024)
    temperature: 1.0 = standard DDIM; >1 adds scaled noise each step to reduce
        mean-collapse (more diverse CLIP samples at inference).
    returns: (B, 768) L2-normalized CLIP
    """
    device = cond.device
    schedule = schedule.to(device)
    b = cond.size(0)
    if seed is not None:
        g = torch.Generator(device=device)
        g.manual_seed(int(seed))
        x = torch.randn(b, prior.output_dim, device=device, generator=g)
    else:
        g = None
        x = torch.randn(b, prior.output_dim, device=device)
    times = torch.linspace(
        schedule.num_timesteps - 1, 0, ddim_steps, device=device,
    ).long().tolist()
    temp = float(temperature)

    def _randn_like(t: torch.Tensor) -> torch.Tensor:
        if g is None:
            return torch.randn_like(t)
        return torch.randn(t.shape, device=t.device, dtype=t.dtype, generator=g)

    for i, t_cur in enumerate(times):
        t_batch = torch.full((b,), t_cur, device=device, dtype=torch.long)
        pred_x0 = prior(x, t_batch, cond)
        if i == len(times) - 1:
            if temp > 1.0:
                sigma_t = schedule.sqrt_one_minus_alpha_bar[t_cur].view(-1, 1)
                x = pred_x0 + temp * _randn_like(pred_x0) * sigma_t
            else:
                x = pred_x0
            break
        t_next = times[i + 1]
        alpha_cur = schedule.alpha_bar[t_cur]
        alpha_next = schedule.alpha_bar[t_next]
        sigma = (
            eta
            * torch.sqrt((1 - alpha_next) / (1 - alpha_cur) * (1 - alpha_cur / alpha_next))
        )
        dir_xt = torch.sqrt(1 - alpha_next - sigma**2) * pred_x0
        x = torch.sqrt(alpha_next) * pred_x0 + dir_xt + sigma * _randn_like(x)
        if temp > 1.0:
            sigma_t = torch.sqrt(1 - alpha_next).view(-1, 1)
            x = x + temp * _randn_like(x) * sigma_t

    return F.normalize(x.float(), dim=-1)
