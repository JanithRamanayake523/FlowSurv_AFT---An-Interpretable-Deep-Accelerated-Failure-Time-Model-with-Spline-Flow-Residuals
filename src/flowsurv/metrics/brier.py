"""Integrated Brier score with IPCW weights (Methodology Sec. 3.3).

Gerds & Schumacher (2006) inverse-probability-of-censoring weighting:

    BS(s) = mean_i [ 1{t_i <= s, d_i = 1} * S(s|x_i)^2 / G(t_i-)
                    + 1{t_i > s} * (1 - S(s|x_i))^2 / G(s) ]
    IBS   = tau^{-1} * int_0^tau BS(s) ds

with G the Kaplan-Meier estimate of the censoring distribution (G(t-) the
left limit, per Graf et al. 1999). tau is the
90th percentile of observed test times (pre-registration Sec. 3).

Inputs are torch tensors (float32 in); computations run in float64
internally and the result is a python float.
"""

from __future__ import annotations

import torch
from torch import Tensor

from .discrimination import kaplan_meier_cdf

_G_MIN = 1e-7  # clamp for the censoring KM estimate (G = 0 beyond the last censoring time)


def integrated_brier_score(
    surv: Tensor,
    grid: Tensor,
    t: Tensor,
    d: Tensor,
    tau: float | None = None,
) -> float:
    """IPCW integrated Brier score on [0, tau].

    Args:
        surv: (m, n) predicted survival S(grid_a | x_i) -- grid points rows,
            subjects columns (the layout FlowSurvAFT.survival returns).
        grid: (m,) time points, ascending.
        t, d: (n,) observed times and event indicators of the test fold.
        tau: integration horizon; defaults to the 90th percentile of the
            observed times. Integration uses the trapezoid rule over the
            grid points <= tau, normalized by tau.

    Defined on all-censored folds: all weight then comes from the
    1{t_i > s} term with weights 1/G(s).
    """
    grid = torch.as_tensor(grid, dtype=torch.float64).flatten()
    t = torch.as_tensor(t, dtype=torch.float64).flatten()
    d = torch.as_tensor(d, dtype=torch.float64).flatten()
    if tau is None:
        tau = float(torch.quantile(torch.as_tensor(t, dtype=torch.float32), 0.9))
    surv = torch.as_tensor(surv, dtype=torch.float64)

    mask = grid <= tau
    grid_r = grid[mask]
    surv_r = surv[mask]

    # censoring KM: G(t_i) per subject and G(s) per grid point
    g_ti = (1.0 - kaplan_meier_cdf(t, d, t, left=True)).clamp_min(_G_MIN)
    g_s = (1.0 - kaplan_meier_cdf(t, d, grid_r)).clamp_min(_G_MIN)

    # (m, n) indicator matrices; a subject censored exactly at s enters
    # neither term (status at s unknown) per Gerds & Schumacher
    event_before = ((t.unsqueeze(0) <= grid_r.unsqueeze(1)) & (d.unsqueeze(0) > 0)).to(torch.float64)
    alive_after = (t.unsqueeze(0) > grid_r.unsqueeze(1)).to(torch.float64)

    term_event = event_before * surv_r.pow(2) / g_ti.unsqueeze(0)
    term_alive = alive_after * (1.0 - surv_r).pow(2) / g_s.unsqueeze(1)
    bs = (term_event + term_alive).mean(dim=1)  # (m,)
    return float(torch.trapezoid(bs, grid_r) / tau)
