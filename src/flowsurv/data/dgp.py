"""Simulation data-generating processes (Methodology Sec. 3.1).

Phase 1 scope: S1 (Weibull sanity) only, needed by the gate tests and the
Phase 1 smoke fit. S2-S6 land in Phase 2 per the Implementation Plan.
"""

from __future__ import annotations

import torch
from torch import Tensor

#: active coefficients on the hazard (Methodology Sec. 3.1)
S1_BETA = (0.8, -0.6, 0.5, -0.4, 0.3)
S1_K = 1.5
S1_LAMBDA = 1.0


def simulate_s1(
    n: int,
    p: int = 10,
    beta: tuple[float, ...] = S1_BETA,
    k: float = S1_K,
    lam: float = S1_LAMBDA,
    seed: int | None = None,
) -> tuple[Tensor, Tensor]:
    """S1 Weibull AFT: h(t|x) = (k/lam) (t/lam)^(k-1) exp(beta'x), x ~ N(0, I_p).

    The first ``len(beta)`` covariates are active. Inversion method:
    T = lam * (-log U)^(1/k) * exp(-beta'x / k), U ~ Uniform(0, 1).

    In AFT form: log T = log lam - (beta'x)/k + W/k with W a standard
    minimum-extreme-value variate, so the true AFT location coefficients are
    ``-beta / k`` and the true AFT scale is ``1 / k``.

    Returns ``(t, x)`` -- event times (n,) and covariates (n, p).
    """
    if len(beta) > p:
        raise ValueError("more active coefficients than covariates")
    gen = torch.Generator().manual_seed(seed) if seed is not None else None
    x = torch.randn(n, p, generator=gen)
    u = torch.rand(n, generator=gen).clamp_min(torch.finfo(torch.float32).tiny)
    b = torch.tensor(beta)
    lin = x[:, : len(beta)] @ b
    t = lam * (-u.log()) ** (1.0 / k) * torch.exp(-lin / k)
    return t, x


def s1_true_log_density(t: Tensor, x: Tensor, beta=S1_BETA, k: float = S1_K, lam: float = S1_LAMBDA) -> Tensor:
    """True S1 log-density, for the likelihood comparison of gate test 1."""
    b = torch.tensor(beta, dtype=t.dtype)
    lin = x[:, : len(beta)] @ b
    return (
        torch.log(torch.tensor(k / lam, dtype=t.dtype))
        + (k - 1) * (t.log() - torch.log(torch.tensor(lam, dtype=t.dtype)))
        + lin
        - (t / lam) ** k * lin.exp()
    )
