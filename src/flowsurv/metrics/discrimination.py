"""Discrimination metrics: Uno's C with IPCW weights (Methodology Sec. 3.3).

Uno's C is the ONLY concordance variant reported in the study
(pre-registration Sec. 3, anti-C-hacking per Sonabend 2022):

    C_U = [sum_{i != j} d_i * G(t_i-)^{-2} * 1{t_i < t_j} * 1{t_i <= tau} * 1{r_i > r_j}]
          / [sum_{i != j} d_i * G(t_i-)^{-2} * 1{t_i < t_j} * 1{t_i <= tau}]

with G the Kaplan-Meier estimate of the censoring distribution (G(t-) its
left limit), r the model risk score (higher = riskier) and tau the
truncation horizon (default: 90th percentile of observed times, the same
tau as IBS/ICI; Uno et al. 2011 define the estimator on [0, tau] with
G(tau) > 0 so that events near the end of follow-up cannot dominate).

Inputs are torch tensors (float32 in); computations run in float64
internally and results are returned as python floats.
"""

from __future__ import annotations

import torch
from torch import Tensor

_G_MIN = 1e-7  # clamp for the censoring KM estimate (G = 0 beyond the last censoring time)


def kaplan_meier_cdf(t: Tensor, d: Tensor, grid: Tensor, *, left: bool = False) -> Tensor:
    """Kaplan-Meier estimate of the CENSORING distribution, evaluated on ``grid``.

    The event indicator of the censoring distribution is ``1 - d`` (flip of
    the event indicator): a subject with d = 0 is an "event" for the
    censoring law. Returns the CDF P(C <= s) on each grid point; the survival
    function needed for IPCW weights is ``1 - kaplan_meier_cdf(...)``.

    Right-continuous step function: the value at s uses all censoring times
    <= s. With ``left=True`` the left limit is returned instead (censoring
    times strictly < s), i.e. G(s-) = 1 - F(s-), which is what Uno's and
    Graf's IPCW weights for an event at s require. O(n log n) via sorting.
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
    idx = torch.searchsorted(uniq, grid, right=not left) - 1
    surv = torch.where(idx >= 0, surv_u[idx.clamp_min(0)], torch.ones_like(grid))
    return 1.0 - surv


def unos_c(risk: Tensor, t: Tensor, d: Tensor, tau: float | None = None) -> float:
    """Uno's C-statistic with IPCW weights G(t_i-)^{-2}, truncated at ``tau``.

    Tied risk scores count as half-concordant (standard convention, matches
    scikit-survival). Undefined on an all-censored fold or when no comparable
    pairs exist: returns NaN (the caller reports the fold as not evaluable).

    O(n^2) pairwise computation -- test folds are <= 1500 rows, so the
    pairwise matrices are small; clarity beats a sorted O(n log n) scan here.
    """
    risk = torch.as_tensor(risk, dtype=torch.float64).flatten()
    t = torch.as_tensor(t, dtype=torch.float64).flatten()
    d = torch.as_tensor(d, dtype=torch.float64).flatten()

    if tau is None:
        tau = float(torch.quantile(t, 0.9))

    # G(t_i-) per subject: left limit of the censoring survival at the subject's time
    g_hat = (1.0 - kaplan_meier_cdf(t, d, t, left=True)).clamp_min(_G_MIN)
    # d_i * G(t_i-)^{-2}; zero weight for censored subjects and events after tau
    w = d * (t <= tau).to(torch.float64) / g_hat.pow(2)

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
