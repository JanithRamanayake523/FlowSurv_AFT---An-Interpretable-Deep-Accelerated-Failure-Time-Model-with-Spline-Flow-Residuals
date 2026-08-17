"""Discrimination metrics: Uno's C with IPCW weights (Methodology Sec. 3.3).

Uno's C is the ONLY concordance variant reported in the study
(pre-registration Sec. 3, anti-C-hacking per Sonabend 2022):

    C_U = [sum_{i != j} d_i * G(t_i)^{-2} * 1{t_i < t_j} * 1{r_i > r_j}]
          / [sum_{i != j} d_i * G(t_i)^{-2} * 1{t_i < t_j}]

with G the Kaplan-Meier estimate of the censoring distribution and r the
model risk score (higher = riskier).

Inputs are torch tensors (float32 in); computations run in float64
internally and results are returned as python floats.
"""

from __future__ import annotations

import torch
from torch import Tensor

_G_MIN = 1e-7  # clamp for the censoring KM estimate (G = 0 beyond the last censoring time)


def kaplan_meier_cdf(t: Tensor, d: Tensor, grid: Tensor) -> Tensor:
    """Kaplan-Meier estimate of the CENSORING distribution, evaluated on ``grid``.

    The event indicator of the censoring distribution is ``1 - d`` (flip of
    the event indicator): a subject with d = 0 is an "event" for the
    censoring law. Returns the CDF P(C <= s) on each grid point; the survival
    function needed for IPCW weights is ``1 - kaplan_meier_cdf(...)``.

    Right-continuous step function: the value at s uses all censoring times
    <= s. O(n log n) via sorting.
    """
    t = torch.as_tensor(t, dtype=torch.float64).flatten()
    d = torch.as_tensor(d, dtype=torch.float64).flatten()
    grid = torch.as_tensor(grid, dtype=torch.float64).flatten()

    cens = 1.0 - d  # event indicator for the censoring distribution
    order = torch.argsort(t)
    t_s, cens_s = t[order], cens[order]

    # censoring events and risk-set size at each unique observed time
    uniq, inverse, counts = torch.unique_consecutive(t_s, return_inverse=True, return_counts=True)
    d_c = torch.zeros_like(uniq).index_add_(0, inverse, cens_s)
    n_risk = torch.cumsum(counts.flip(0), dim=0).flip(0)  # suffix sums: #{t >= uniq_j}
    surv_u = torch.cumprod(1.0 - d_c / n_risk, dim=0)  # KM survival of C at unique times

    # evaluate on the grid: step function, value at s = survival at last uniq <= s
    idx = torch.searchsorted(uniq, grid, right=True) - 1
    surv = torch.where(idx >= 0, surv_u[idx.clamp_min(0)], torch.ones_like(grid))
    return 1.0 - surv


def unos_c(risk: Tensor, t: Tensor, d: Tensor) -> float:
    """Uno's C-statistic with IPCW weights G(t_i)^{-2} (Methodology Sec. 3.3).

    Tied risk scores count as half-concordant (standard convention, matches
    scikit-survival). Undefined on an all-censored fold or when no comparable
    pairs exist: returns NaN (the caller reports the fold as not evaluable).

    O(n^2) pairwise computation -- test folds are <= 1500 rows, so the
    pairwise matrices are small; clarity beats a sorted O(n log n) scan here.
    """
    risk = torch.as_tensor(risk, dtype=torch.float64).flatten()
    t = torch.as_tensor(t, dtype=torch.float64).flatten()
    d = torch.as_tensor(d, dtype=torch.float64).flatten()

    # G(t_i) per subject: survival of the censoring law at the subject's time
    g_hat = (1.0 - kaplan_meier_cdf(t, d, t)).clamp_min(_G_MIN)
    w = d / g_hat.pow(2)  # d_i * G(t_i)^{-2}; zero weight for censored subjects

    comparable = t.unsqueeze(1) < t.unsqueeze(0)  # [i, j]: t_i < t_j (excludes i == j)
    weight = w.unsqueeze(1) * comparable
    # [i, j]: 1 if risk_i > risk_j (earlier time -> higher risk score)
    score = (risk.unsqueeze(1) > risk.unsqueeze(0)).to(torch.float64)
    score = score + 0.5 * (risk.unsqueeze(1) == risk.unsqueeze(0)).to(torch.float64)

    denom = weight.sum()
    if denom <= 0:
        # all-censored fold / no comparable pairs: Uno's C is undefined
        return float("nan")
    return float((weight * score).sum() / denom)
