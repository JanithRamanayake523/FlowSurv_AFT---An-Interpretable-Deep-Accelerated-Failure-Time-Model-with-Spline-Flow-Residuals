"""Calibration metrics: D-calibration, ICI, calibration slope (Methodology Sec. 3.3).

- D-calibration (Haider et al. 2020): chi-square uniformity test on deciles
  of the predicted survival probability at the observed time. Reported as
  statistic + pass/fail at alpha = 0.05; the pass rate across replications
  is the pre-registered summary (pre-registration Sec. 3).
- ICI (Austin et al. 2020): mean absolute deviation of the loess calibration
  curve of observed vs predicted event probabilities at tau.
- Calibration slope: logistic regression of the observed event indicator at
  tau on logit(F_hat(tau|x)); used by contrast C2 (pre-registration Sec. 2).

Inputs are torch tensors (float32 in); results are python floats.
"""

from __future__ import annotations

from dataclasses import dataclass

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
    hence S(T_i|x_i) < S(t_i|x_i): its unit mass is distributed uniformly
    over the bins from its own bin down to bin 0 (lower S = later event
    times), each receiving 1/(b_i + 1).

    Defined on all-censored folds (counts are fractional). The chi-square
    test uses n_bins - 1 degrees of freedom with expected counts n / n_bins.
    """
    s = torch.as_tensor(s_at_obs, dtype=torch.float64).flatten().clamp(0.0, 1.0)
    d = torch.as_tensor(d, dtype=torch.float64).flatten()

    counts = torch.zeros(n_bins, dtype=torch.float64)
    b = _bin_index(s, n_bins)
    uncensored = d > 0
    counts.index_add_(0, b[uncensored], torch.ones(int(uncensored.sum()), dtype=torch.float64))
    for i in torch.nonzero(~uncensored).flatten().tolist():
        # censored: uniform fractional mass over bins 0..b_i (inclusive)
        counts[: b[i] + 1] += 1.0 / (int(b[i]) + 1)

    expected = s.numel() / n_bins
    statistic = float(((counts - expected) ** 2 / expected).sum())
    pvalue = float(stats.chi2.sf(statistic, df=n_bins - 1))
    return DCalResult(statistic=statistic, pvalue=pvalue, passed=pvalue > DCAL_ALPHA)


def _observed_event_indicator(t: Tensor, d: Tensor, tau: float) -> Tensor:
    """y_i = 1{t_i <= tau, d_i = 1}: event observed before tau."""
    return ((t <= tau) & (d > 0)).to(torch.float64)


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


def ici(s_at_tau: Tensor, t: Tensor, d: Tensor, tau: float | None = None) -> float:
    """Integrated calibration index at tau (Austin et al. 2020).

    Observed: y_i = 1{t_i <= tau, d_i = 1}. Predicted: F(tau|x_i) =
    1 - s_at_tau, clamped to [1e-7, 1 - 1e-7]. The calibration curve is a
    loess smooth (frac = 0.75) of y on F; ICI is the mean absolute deviation
    between smoothed observed and predicted. Falls back to an equal-count
    binned-mean curve when statsmodels is not installed.
    """
    s = torch.as_tensor(s_at_tau, dtype=torch.float64).flatten().clamp(_S_MIN, 1.0 - _S_MIN)
    t = torch.as_tensor(t, dtype=torch.float64).flatten()
    d = torch.as_tensor(d, dtype=torch.float64).flatten()
    if tau is None:
        tau = float(torch.quantile(t, 0.9))

    p = 1.0 - s  # predicted event probability at tau
    y = _observed_event_indicator(t, d, tau)
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess

        smoothed = torch.from_numpy(lowess(y.numpy(), p.numpy(), frac=0.75, return_sorted=False))
    except ImportError:
        smoothed = _binned_mean_smooth(y, p)
    return float((smoothed - p).abs().mean())


def calibration_slope(s_at_tau: Tensor, t: Tensor, d: Tensor, tau: float | None = None) -> float:
    """Calibration slope at tau (contrast C2, pre-registration Sec. 2).

    Logistic regression of the observed event indicator 1{t_i <= tau, d_i = 1}
    on logit(F_hat(tau|x_i)); a slope of 1 indicates perfect calibration-in-
    the-large-adjusted spread. Returns NaN when the observed indicator has no
    variation (slope undefined). Uses statsmodels Logit; falls back to a
    small Newton-Raphson solver when statsmodels is unavailable.
    """
    s = torch.as_tensor(s_at_tau, dtype=torch.float64).flatten().clamp(_S_MIN, 1.0 - _S_MIN)
    t = torch.as_tensor(t, dtype=torch.float64).flatten()
    d = torch.as_tensor(d, dtype=torch.float64).flatten()
    if tau is None:
        tau = float(torch.quantile(t, 0.9))

    eta = torch.logit(1.0 - s)  # linear predictor of the predicted probabilities
    y = _observed_event_indicator(t, d, tau)
    if y.sum() == 0.0 or y.sum() == y.numel():
        # no variation in the observed indicator: slope undefined
        return float("nan")

    try:
        import statsmodels.api as sm

        x = sm.add_constant(eta.numpy())
        res = sm.Logit(y.numpy(), x).fit(disp=False)
        return float(res.params[1])
    except ImportError:
        return _logistic_slope_newton(eta, y)


def _logistic_slope_newton(eta: Tensor, y: Tensor, max_iter: int = 50, tol: float = 1e-10) -> float:
    """Slope of y ~ sigmoid(beta0 + beta1 * eta) by Newton-Raphson (IRLS)."""
    x = torch.stack([torch.ones_like(eta), eta], dim=1)
    beta = torch.zeros(2, dtype=torch.float64)
    ridge = 1e-10 * torch.eye(2, dtype=torch.float64)  # guards near-separation
    for _ in range(max_iter):
        mu = torch.sigmoid(x @ beta)
        w = mu * (1.0 - mu)
        grad = x.T @ (y - mu)
        hess = (x.T * w) @ x
        step = torch.linalg.solve(hess + ridge, grad)
        beta = beta + step
        if step.abs().max() < tol:
            break
    return float(beta[1])
