"""Censoring-aware losses (Methodology Sec. 2.3).

All terms are evaluated in log-space; no IPCW weighting anywhere -- the exact
survival term of the likelihood handles censoring directly.
"""

from __future__ import annotations

import torch
from torch import Tensor


def right_censored_nll(model, t: Tensor, d: Tensor, x: Tensor) -> Tensor:
    """Negative mean right-censored log-likelihood.

    -mean_i [ d_i * log f(t_i|x_i) + (1 - d_i) * log S(t_i|x_i) ]
    """
    log_f = model.log_density(t, x)
    log_s = model.log_survival(t, x)
    ll = d * log_f + (1 - d) * log_s
    return -ll.mean()


def interval_censored_nll(model, left: Tensor, right: Tensor, x: Tensor) -> Tensor:
    """Negative mean interval-censored log-likelihood.

    Each subject is known to fail in (left, right] and contributes
    log[ S(left|x) - S(right|x) ], computed as
    log S(left) + log1p(-exp(log S(right) - log S(left))) for stability.
    """
    log_s_l = model.log_survival(left, x)
    log_s_r = model.log_survival(right, x)
    # S(right) <= S(left) since left <= right; clamp guards float rounding
    ratio = torch.exp(log_s_r - log_s_l).clamp_max(1.0 - 1e-7)
    ll = log_s_l + torch.log1p(-ratio)
    return -ll.mean()


def nelson_aalen(t: Tensor, d: Tensor, grid: Tensor) -> Tensor:
    """Nelson-Aalen cumulative hazard on ``grid`` (non-decreasing step function)."""
    grid = torch.as_tensor(grid, dtype=t.dtype, device=t.device)
    at_risk = (t.unsqueeze(0) >= grid.unsqueeze(1)).sum(dim=1).clamp_min(1)
    dN = ((t.unsqueeze(0) == grid.unsqueeze(1)) & (d.unsqueeze(0) > 0)).sum(dim=1)
    return torch.cumsum(dN / at_risk, dim=0)


def soft_na_wasserstein(model, t: Tensor, d: Tensor, x: Tensor, grid: Tensor | None = None) -> Tensor:
    """Optional Soft-Nelson-Aalen regularizer R(theta) (Methodology Sec. 2.3).

    Li & Cai (2026) match the model-implied cohort cumulative hazard to the
    empirical Nelson-Aalen curve by sampling. FlowSurv-AFT uses the exact
    cohort survival instead: H_model(tau) = -log mean_i S(tau|x_i), and R is
    the mean absolute deviation between H_model and H_NA on a grid of
    observed event times (an L1/Wasserstein-type distance between monotone
    hazard curves). Differentiable w.r.t. model parameters.
    """
    if grid is None:
        grid = torch.unique(t[d > 0])
    grid = grid.to(x.device)
    s_cohort = model.survival(grid.unsqueeze(1), x.unsqueeze(0).expand(grid.numel(), -1, -1))
    h_model = -s_cohort.mean(dim=1).clamp_min(torch.finfo(s_cohort.dtype).tiny).log()
    h_na = nelson_aalen(t, d, grid)
    return (h_model - h_na).abs().mean()


def total_nll(
    model,
    t: Tensor,
    d: Tensor,
    x: Tensor,
    lambda_softna: float = 0.0,
) -> Tensor:
    """L = -ll + lambda * R; lambda = 0 is the pre-registered primary (Sec. 2.3)."""
    loss = right_censored_nll(model, t, d, x)
    if lambda_softna > 0:
        loss = loss + lambda_softna * soft_na_wasserstein(model, t, d, x)
    return loss
