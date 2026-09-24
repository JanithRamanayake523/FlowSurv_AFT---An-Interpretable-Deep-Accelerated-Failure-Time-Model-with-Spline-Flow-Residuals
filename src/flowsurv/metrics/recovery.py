"""Hazard recovery and distributional fidelity metrics (Methodology Sec. 3.3).

Simulation-only metrics: they require ground-truth hazards and CDFs. All
hazard/CDF inputs are (m, n) -- m grid points x n test subjects -- the
layout FlowSurvAFT.hazard / .cdf return when called with a time grid.

Inputs are torch tensors (float32 in); computations run in float64
internally. Scalars are python floats; per-subject diagnostics stay tensors.
"""

from __future__ import annotations

import torch
from torch import Tensor

_QUARTILES = (0.25, 0.5, 0.75)


def _default_weights(h_true: Tensor, grid: Tensor) -> Tensor:
    """w(t) proportional to the marginal event-time density of the TRUE model,
    normalized to integrate to 1 on the grid (trapezoid).

    f_true(t|x) = h_true(t|x) * exp(-int_0^t h_true(u|x) du); w(t) is the
    subject average of f_true (Methodology Sec. 3.3: "weights proportional to
    the marginal density of evaluation times").
    """
    cumhaz = torch.cumulative_trapezoid(h_true, grid, dim=0)
    cumhaz = torch.cat([torch.zeros_like(cumhaz[:1]), cumhaz], dim=0)  # H(0) = 0
    f_true = h_true * torch.exp(-cumhaz)
    w = f_true.mean(dim=1)
    area = torch.trapezoid(w, grid)
    return w / area.clamp_min(torch.finfo(w.dtype).tiny)


def hazard_recovery_error(
    h_pred: Tensor,
    h_true: Tensor,
    grid: Tensor,
    weights: Tensor | None = None,
) -> float:
    """HRE = mean over subjects of int [h_pred - h_true]^2 w(t) dt.

    Headline simulation metric (Methodology Sec. 3.3, pre-registration
    metric 4). Trapezoid rule on ``grid``. Default weights are the marginal
    true-model event-time density, normalized so int w dt = 1; caller-
    provided weights are used AS GIVEN (not renormalized).

    Methods without hazard support must be filtered by the caller: both
    hazard inputs are asserted finite here.
    """
    h_pred = torch.as_tensor(h_pred, dtype=torch.float64)
    h_true = torch.as_tensor(h_true, dtype=torch.float64)
    grid = torch.as_tensor(grid, dtype=torch.float64).flatten()
    assert torch.isfinite(h_pred).all() and torch.isfinite(h_true).all(), (
        "h_pred/h_true must be finite; filter methods without hazard support upstream"
    )

    if weights is None:
        w = _default_weights(h_true, grid)
    else:
        w = torch.as_tensor(weights, dtype=torch.float64).flatten()
    per_subject = torch.trapezoid((h_pred - h_true).pow(2) * w.unsqueeze(1), grid, dim=0)
    return float(per_subject.mean())


def cumulative_hazard_error(
    h_pred: Tensor,
    h_true: Tensor,
    grid: Tensor,
    weights: Tensor | None = None,
) -> float:
    """Scale-robust companion to HRE (exploratory): mean over subjects of
    int |H_pred(t) - H_true(t)| w(t) dt, with H the trapezoid cumulative
    hazard on ``grid``.

    HRE squares a hazard difference, so a handful of subjects with extreme
    true hazards (S5 up to ~1e5; S4 with k = 0.7 diverges as t -> 0) can
    decide a cell. The cumulative hazard is an integral of the hazard and its
    absolute error is far less dominated by such spikes. Same default weights
    as :func:`hazard_recovery_error`; not pre-registered, reported alongside
    HRE and never used for a confirmatory claim.
    """
    h_pred = torch.as_tensor(h_pred, dtype=torch.float64)
    h_true = torch.as_tensor(h_true, dtype=torch.float64)
    grid = torch.as_tensor(grid, dtype=torch.float64).flatten()
    assert torch.isfinite(h_pred).all() and torch.isfinite(h_true).all(), (
        "h_pred/h_true must be finite; filter methods without hazard support upstream"
    )
    w = _default_weights(h_true, grid) if weights is None else torch.as_tensor(weights, dtype=torch.float64).flatten()
    cum_pred = torch.cumulative_trapezoid(h_pred, grid, dim=0)
    cum_true = torch.cumulative_trapezoid(h_true, grid, dim=0)
    per_subject = torch.trapezoid((cum_pred - cum_true).abs() * w[1:].unsqueeze(1), grid[1:], dim=0)
    return float(per_subject.mean())


def cdf_fidelity(f_pred: Tensor, f_true: Tensor, grid: Tensor) -> dict:
    """Per-subject KS and W1 between estimated and true conditional CDFs.

    KS_n = max_t |F_pred - F_true|; W1_n = int |F_pred - F_true| dt (the 1-D
    Wasserstein-1 identity between CDFs), trapezoid on the grid. Returns the
    subject means plus per-subject tensors (Methodology Sec. 3.3,
    pre-registration metric 5).
    """
    f_pred = torch.as_tensor(f_pred, dtype=torch.float64)
    f_true = torch.as_tensor(f_true, dtype=torch.float64)
    grid = torch.as_tensor(grid, dtype=torch.float64).flatten()
    assert f_pred.shape == f_true.shape, "f_pred and f_true must have identical (m, n) shape"

    diff = (f_pred - f_true).abs()
    ks = diff.max(dim=0).values  # (n,)
    w1 = torch.trapezoid(diff, grid, dim=0)  # (n,)
    return {
        "ks": float(ks.mean()),
        "w1": float(w1.mean()),
        "ks_per_subject": ks,
        "w1_per_subject": w1,
    }


def _interp1(y: Tensor, grid: Tensor, x: Tensor) -> Tensor:
    """Linear interpolation of values ``y`` on ``grid`` at scalar ``x``,
    clamped to the grid endpoints."""
    x = x.clamp(grid[0], grid[-1])
    k = int(torch.searchsorted(grid, x).clamp(1, grid.numel() - 1))
    x0, x1 = grid[k - 1], grid[k]
    w = (x - x0) / (x1 - x0)
    return y[k - 1] + w * (y[k] - y[k - 1])


def quartile_cdf_deviation(f_pred: Tensor, f_true: Tensor, grid: Tensor) -> Tensor:
    """Signed F_pred - F_true at the TRUE conditional quartiles, per subject.

    Bias diagnostic (Methodology Sec. 3.3). Returns an (n, 3) tensor; column
    j is the deviation at the true (j+1)-th quartile time
    Q_true(quartile | x_n), found by linear inversion of F_true on the grid,
    with F_pred (and F_true) linearly interpolated at that time. If a
    quartile level lies beyond F_true's range on the grid, the deviation is
    evaluated at the grid endpoint (right-truncation of the diagnostic on the
    evaluation grid, not of the model).
    """
    f_pred = torch.as_tensor(f_pred, dtype=torch.float64)
    f_true = torch.as_tensor(f_true, dtype=torch.float64)
    grid = torch.as_tensor(grid, dtype=torch.float64).flatten()
    m = grid.numel()

    devs = []
    for n in range(f_true.shape[1]):
        ft = torch.cummax(f_true[:, n], dim=0).values  # guard numeric non-monotonicity
        row = []
        for q in _QUARTILES:
            k = int(torch.searchsorted(ft, torch.tensor(q, dtype=ft.dtype)))
            if k >= m:
                t_q = grid[-1]  # quartile beyond the grid range
            else:
                # linear inversion between grid[k - 1] and grid[k]; k == 0
                # (q below F_true at grid[0]) lands on the first segment
                k0 = max(k - 1, 0)
                f0, f1 = ft[k0], ft[k]
                w = (q - f0) / (f1 - f0).clamp_min(torch.finfo(torch.float64).tiny)
                t_q = grid[k0] + w.clamp(0.0, 1.0) * (grid[k] - grid[k0])
            dev = _interp1(f_pred[:, n], grid, t_q) - _interp1(f_true[:, n], grid, t_q)
            row.append(dev)
        devs.append(torch.stack(row))
    return torch.stack(devs)
