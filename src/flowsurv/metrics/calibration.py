"""Calibration metrics: D-calibration, ICI, calibration slope (Methodology Sec. 3.3).

- D-calibration (Haider et al. 2020): chi-square uniformity test on deciles
  of the predicted survival probability at the observed time. Reported as
  statistic + pass/fail at alpha = 0.05; the pass rate across replications
  is the pre-registered summary (pre-registration Sec. 3).
- ICI (Austin et al. 2020): mean absolute deviation of the (non-robust)
  loess calibration curve of observed vs predicted event probabilities at
  tau. The observed outcome is the Kaplan-Meier jackknife pseudo-observation
  of 1{T <= tau}, which is valid under random censoring (the naive indicator
  1{t <= tau, d = 1} is biased upward in survival and the bias grows with
  censoring).
- Calibration slope: IPCW-weighted logistic regression of 1{T <= tau} on
  logit(F_hat(tau|x)); used by contrast C2 (pre-registration Sec. 2).

Inputs are torch tensors (float32 in); results are python floats.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy import stats
from torch import Tensor

DCAL_ALPHA = 0.05  # fixed significance level (pre-registration Sec. 3)

_S_MIN = 1e-7  # clamp for predicted probabilities where logit/division occurs


@dataclass
class DCalResult:
    """D-calibration outcome: chi-square statistic, p-value, pass at alpha=0.05."""

    statistic: float
    pvalue: float
    passed: bool


def _bin_index(s: Tensor, n_bins: int) -> Tensor:
    """Equal-width bin of S in [0, 1]: bin b = [b/k, (b+1)/k), top bin closed.

    Aligned with gate test 6 (tests/test_gate6_dcalibration.py).
    """
    return (s * n_bins).long().clamp(0, n_bins - 1)


def d_calibration(s_at_obs: Tensor, d: Tensor, n_bins: int = 10) -> DCalResult:
    """D-calibration after Haider et al. (2020), Algorithm 1.

    ``s_at_obs[i]`` is the predicted survival S(t_i|x_i) at the subject's
    observed time. Uncensored subjects count fully in the bin containing
    S(t_i|x_i). A subject censored at t_i has its event strictly after t_i,
    hence S(T_i|x_i) ~ Uniform(0, s_i): each bin entirely below s_i receives
    (1/k)/s_i of its unit mass and its own bin the remainder
    (s_i - b_i/k)/s_i, as in Haider et al. (2020), Algorithm 1.

    Defined on all-censored folds (counts are fractional). The chi-square
    test uses n_bins - 1 degrees of freedom with expected counts n / n_bins.
    """
    s = torch.as_tensor(s_at_obs, dtype=torch.float64).flatten().clamp(0.0, 1.0)
    d = torch.as_tensor(d, dtype=torch.float64).flatten()

    counts = torch.zeros(n_bins, dtype=torch.float64)
    b = _bin_index(s, n_bins)
    uncensored = d > 0
    counts.index_add_(0, b[uncensored], torch.ones(int(uncensored.sum()), dtype=torch.float64))
    # censored: S(T_i) is uniform on [0, s_i] (Haider et al. 2020), so bin j < b_i
    # receives (1/k) / s_i and the subject's own bin (s_i - b_i/k) / s_i.
    cen = ~uncensored
    if cen.any():
        s_c, b_c = s[cen].clamp_min(1e-12), b[cen]
        width = 1.0 / n_bins
        j = torch.arange(n_bins).unsqueeze(0)  # (1, k)
        below = (j < b_c.unsqueeze(1)).to(torch.float64) * (width / s_c).unsqueeze(1)
        own_frac = ((s_c - b_c * width) / s_c).clamp(0.0, 1.0)
        own = (j == b_c.unsqueeze(1)).to(torch.float64) * own_frac.unsqueeze(1)
        counts += (below + own).sum(dim=0)

    expected = s.numel() / n_bins
    statistic = float(((counts - expected) ** 2 / expected).sum())
    pvalue = float(stats.chi2.sf(statistic, df=n_bins - 1))
    return DCalResult(statistic=statistic, pvalue=pvalue, passed=pvalue > DCAL_ALPHA)


def _binned_mean_smooth(y: Tensor, p: Tensor, n_bins: int = 10) -> Tensor:
    """Equal-count binned-mean calibration curve (fallback when statsmodels
    is unavailable): each subject gets the mean observed rate of its
    predicted-probability decile."""
    order = torch.argsort(p)
    ranks = torch.empty_like(order)
    ranks[order] = torch.arange(y.numel())
    bin_id = (ranks * n_bins // y.numel()).clamp(max=n_bins - 1)
    y_sum = torch.zeros(n_bins, dtype=torch.float64).index_add_(0, bin_id, y)
    cnt = torch.zeros(n_bins, dtype=torch.float64).index_add_(0, bin_id, torch.ones_like(y))
    return (y_sum / cnt.clamp_min(1.0))[bin_id]


def _km_surv_at(t: np.ndarray, d: np.ndarray, tau: float) -> float:
    """Kaplan-Meier S(tau) for right-censored data (ties handled per unique time)."""
    uniq, inv = np.unique(t, return_inverse=True)
    events = np.bincount(inv, weights=d, minlength=uniq.size)
    n_risk = t.size - np.concatenate(([0.0], np.cumsum(np.bincount(inv, minlength=uniq.size))[:-1]))
    keep = uniq <= tau
    return float(np.prod(1.0 - events[keep] / n_risk[keep]))


def _pseudo_event_prob(t: np.ndarray, d: np.ndarray, tau: float) -> np.ndarray:
    """Jackknife pseudo-observations of 1{T <= tau} from the Kaplan-Meier estimate.

    theta_i = n * F_hat(tau) - (n - 1) * F_hat^{(-i)}(tau). Valid under random
    (covariate-independent) censoring, which holds for the simulated Type I/III
    schemes and is the usual approximation on real data.
    """
    n = t.size
    full = 1.0 - _km_surv_at(t, d, tau)
    out = np.empty(n)
    for i in range(n):
        keep = np.arange(n) != i
        out[i] = n * full - (n - 1) * (1.0 - _km_surv_at(t[keep], d[keep], tau))
    return out


def _ipcw_event_weights(t: Tensor, d: Tensor, tau: float) -> tuple[Tensor, Tensor]:
    """IPCW outcome y = 1{T <= tau} and weights: events by tau get 1/G(t_i-),
    subjects known to survive past tau get 1/G(tau-), subjects censored
    before tau get weight 0 (their outcome is unknown)."""
    from .discrimination import kaplan_meier_cdf

    g_ti = (1.0 - kaplan_meier_cdf(t, d, t, left=True)).clamp_min(_S_MIN)
    g_tau = (1.0 - kaplan_meier_cdf(t, d, torch.tensor([tau], dtype=torch.float64), left=True)).clamp_min(_S_MIN)
    event = (t <= tau) & (d > 0)
    # T > tau is known for subjects observed past tau, and for those censored exactly
    # at tau (censored at t means T > t); this matters under Type I censoring, where
    # tau is the common administrative censoring time.
    beyond = (t > tau) | ((t == tau) & (d == 0))
    y = event.to(torch.float64)
    w = torch.where(event, 1.0 / g_ti, torch.where(beyond, 1.0 / g_tau, torch.zeros_like(t)))
    return y, w


def ici(s_at_tau: Tensor, t: Tensor, d: Tensor, tau: float | None = None) -> float:
    """Integrated calibration index at tau (Austin et al. 2020).

    Observed: Kaplan-Meier jackknife pseudo-observation of 1{T <= tau}.
    Predicted: F(tau|x_i) = 1 - s_at_tau, clamped to [1e-7, 1 - 1e-7]. The
    calibration curve is a plain (non-robust, ``it=0``) loess smooth
    (frac = 0.75) of the pseudo-observations on F; ICI is the mean absolute
    deviation between smoothed observed and predicted. The default robust
    lowess iterations down-weight the minority class of a 0/1-like outcome
    and bias the oracle ICI to about 0.10, so they are disabled. Falls back
    to an equal-count binned-mean curve when statsmodels is not installed.
    """
    s = torch.as_tensor(s_at_tau, dtype=torch.float64).flatten().clamp(_S_MIN, 1.0 - _S_MIN)
    t = torch.as_tensor(t, dtype=torch.float64).flatten()
    d = torch.as_tensor(d, dtype=torch.float64).flatten()
    if tau is None:
        tau = float(torch.quantile(t, 0.9))

    p = 1.0 - s  # predicted event probability at tau
    y = torch.from_numpy(_pseudo_event_prob(t.numpy(), d.numpy(), tau))
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess

        smoothed = torch.from_numpy(lowess(y.numpy(), p.numpy(), frac=0.75, it=0, return_sorted=False))
    except ImportError:
        smoothed = _binned_mean_smooth(y, p)
    return float((smoothed - p).abs().mean())


def calibration_slope(s_at_tau: Tensor, t: Tensor, d: Tensor, tau: float | None = None) -> float:
    """Calibration slope at tau (contrast C2, pre-registration Sec. 2).

    IPCW-weighted logistic regression of 1{T <= tau} on logit(F_hat(tau|x_i))
    (subjects censored before tau carry zero weight, the rest are up-weighted
    by the inverse censoring probability); a slope of 1 indicates perfect
    calibration-in-the-large-adjusted spread. Returns NaN when the weighted
    outcome has no variation (slope undefined).
    """
    s = torch.as_tensor(s_at_tau, dtype=torch.float64).flatten().clamp(_S_MIN, 1.0 - _S_MIN)
    t = torch.as_tensor(t, dtype=torch.float64).flatten()
    d = torch.as_tensor(d, dtype=torch.float64).flatten()
    if tau is None:
        tau = float(torch.quantile(t, 0.9))

    eta = torch.logit(1.0 - s)  # linear predictor of the predicted probabilities
    y, w = _ipcw_event_weights(t, d, tau)
    used = w > 0
    if not used.any() or y[used].sum() == 0.0 or y[used].sum() == float(used.sum()):
        return float("nan")  # no variation in the observed outcome: slope undefined
    return _logistic_slope_newton(eta[used], y[used], w[used])


def _logistic_slope_newton(
    eta: Tensor, y: Tensor, w: Tensor | None = None, max_iter: int = 50, tol: float = 1e-10
) -> float:
    """Slope of y ~ sigmoid(beta0 + beta1 * eta) by weighted Newton-Raphson (IRLS)."""
    if w is None:
        w = torch.ones_like(y)
    x = torch.stack([torch.ones_like(eta), eta], dim=1)
    beta = torch.zeros(2, dtype=torch.float64)
    ridge = 1e-10 * torch.eye(2, dtype=torch.float64)  # guards near-separation
    for _ in range(max_iter):
        mu = torch.sigmoid(x @ beta)
        grad = x.T @ (w * (y - mu))
        hess = (x.T * (w * mu * (1.0 - mu))) @ x
        step = torch.linalg.solve(hess + ridge, grad)
        beta = beta + step
        if step.abs().max() < tol:
            break
    return float(beta[1])
