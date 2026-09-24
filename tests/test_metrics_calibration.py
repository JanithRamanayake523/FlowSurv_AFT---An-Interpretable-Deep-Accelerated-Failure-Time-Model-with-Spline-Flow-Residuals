"""Regression tests for the supervisor-review metric fixes (2026-09-24).

- ICI must be ~0 for the true survival function at every censoring level
  (naive indicator + robust lowess gave 0.10 uncensored and 0.35 at 80%).
- Calibration slope of the oracle must be ~1 under censoring (IPCW).
- D-calibration censored mass follows Haider et al. (proportional split).
- Uno's C is truncated at tau; IPCW weights use the left limit G(t-).
"""

from __future__ import annotations

import numpy as np
import torch

from flowsurv.data.censoring import apply_censoring
from flowsurv.data.dgp import s1_true_cdf, simulate_s1
from flowsurv.metrics.calibration import calibration_slope, d_calibration, ici
from flowsurv.metrics.discrimination import kaplan_meier_cdf, unos_c


def _oracle(n, censoring, ctype, seed):
    t, x = simulate_s1(n, seed=seed)
    to, d = apply_censoring(t, censoring, ctype, seed=seed + 1)
    tau = float(torch.quantile(to, 0.9))
    s_tau = 1 - s1_true_cdf(torch.full_like(to, tau), x)
    return s_tau, to, d, tau


def test_oracle_ici_small_under_censoring():
    for censoring, ctype in [(0.0, "I"), (0.5, "III"), (0.8, "III")]:
        vals = []
        for seed in (1, 2, 3):
            s_tau, to, d, tau = _oracle(1000, censoring, ctype, seed)
            vals.append(ici(s_tau, to, d, tau=tau))
        assert np.mean(vals) < 0.06, (censoring, ctype, vals)


def test_oracle_calibration_slope_near_one_under_censoring():
    for ctype in ("I", "III"):  # Type I: tau equals the common censoring time
        slopes = []
        for seed in (1, 2, 3, 4):
            s_tau, to, d, tau = _oracle(1000, 0.5, ctype, seed)
            slopes.append(calibration_slope(s_tau, to, d, tau=tau))
        assert 0.7 < np.mean(slopes) < 1.3, (ctype, slopes)


def test_dcal_censored_mass_is_proportional():
    # one censored subject with S = 0.25 (bin 2 of 10): bins 0,1 get 0.1/0.25 = 0.4
    # each and bin 2 gets (0.25 - 0.2)/0.25 = 0.2; total mass 1.
    n_bins = 10
    s = torch.tensor([0.25])
    d = torch.tensor([0.0])
    res = d_calibration(s, d, n_bins=n_bins)
    expected = 1 / n_bins
    counts = np.array([0.4, 0.4, 0.2] + [0.0] * 7)
    stat = float(((counts - expected) ** 2 / expected).sum())
    assert np.isclose(res.statistic, stat)


def test_km_left_limit_excludes_ties_at_time():
    t = torch.tensor([1.0, 2.0, 3.0, 4.0])
    d = torch.tensor([1.0, 0.0, 1.0, 1.0])  # censoring "event" at t = 2
    right = kaplan_meier_cdf(t, d, torch.tensor([2.0]))
    left = kaplan_meier_cdf(t, d, torch.tensor([2.0]), left=True)
    assert float(left) == 0.0
    assert float(right) > 0.0


def test_unos_c_truncation_matches_sksurv():
    from sksurv.metrics import concordance_index_ipcw
    from sksurv.util import Surv

    t, x = simulate_s1(400, seed=7)
    to, d = apply_censoring(t, 0.5, "III", seed=8)
    risk = torch.as_tensor(x[:, 0], dtype=torch.float64) + 0.5 * torch.randn(400, generator=torch.Generator().manual_seed(0))
    tau = float(torch.quantile(to, 0.9))
    y = Surv.from_arrays(event=d.numpy().astype(bool), time=to.numpy())
    ref = concordance_index_ipcw(y, y, risk.numpy(), tau=tau)[0]
    assert abs(unos_c(risk, to, d, tau=tau) - ref) < 0.02
