"""Unit tests for Uno's C and the censoring KM helper (Phase 4, Methodology Sec. 3.3)."""

import math

import pytest
import torch

from flowsurv.metrics import kaplan_meier_cdf, unos_c


def test_perfect_ordering_no_censoring():
    """No censoring -> IPCW weights are 1 -> Uno's C = Harrell's C on event pairs."""
    t = torch.tensor([1.0, 2.0, 3.0, 4.0])
    d = torch.ones_like(t)
    risk = -t  # higher risk = earlier event: perfectly ordered
    assert unos_c(risk, t, d) == pytest.approx(1.0)


def test_reversed_ordering_no_censoring():
    t = torch.tensor([1.0, 2.0, 3.0, 4.0])
    d = torch.ones_like(t)
    assert unos_c(t, t, d) == pytest.approx(0.0)  # risk = t: exactly backwards


def test_all_tied_risk_scores_give_half():
    t = torch.tensor([1.0, 2.0, 3.0, 4.0])
    d = torch.ones_like(t)
    risk = torch.zeros_like(t)  # every comparable pair tied -> C = 0.5
    assert unos_c(risk, t, d) == pytest.approx(0.5)


def test_partial_ties_hand_computed():
    # pairs (i, j) with t_i < t_j: (1,2) tied risk, (1,3) and (2,3) concordant
    # -> C = (0.5 + 1 + 1) / 3 = 5/6
    t = torch.tensor([1.0, 2.0, 3.0])
    d = torch.ones_like(t)
    risk = torch.tensor([1.0, 1.0, 0.0])
    assert unos_c(risk, t, d) == pytest.approx(5.0 / 6.0)


def test_all_censored_fold_returns_nan():
    """Uno's C is undefined without events (Phase 4 exit-checklist edge case)."""
    t = torch.tensor([1.0, 2.0, 3.0])
    d = torch.zeros_like(t)
    risk = torch.tensor([3.0, 2.0, 1.0])
    assert math.isnan(unos_c(risk, t, d))


def test_kaplan_meier_cdf_no_censoring_is_zero():
    t = torch.tensor([1.0, 2.0, 3.0])
    d = torch.ones_like(t)
    grid = torch.tensor([0.5, 1.5, 3.5])
    assert torch.allclose(kaplan_meier_cdf(t, d, grid), torch.zeros_like(grid, dtype=torch.float64))


def test_kaplan_meier_cdf_hand_computed():
    # censoring "events" at t = 1 and t = 3; at t = 1: risk 3 -> G = 2/3;
    # at t = 3: risk 1 -> G = 0. CDF: [0, 1/3, 1/3, 1] on the grid below.
    t = torch.tensor([1.0, 2.0, 3.0])
    d = torch.tensor([0.0, 1.0, 0.0])
    grid = torch.tensor([0.5, 1.0, 2.0, 3.0])
    expected = torch.tensor([0.0, 1.0 / 3.0, 1.0 / 3.0, 1.0], dtype=torch.float64)
    assert torch.allclose(kaplan_meier_cdf(t, d, grid), expected)


def test_matches_sksurv_concordance_index_ipcw():
    pytest.importorskip("sksurv")
    from sksurv.metrics import concordance_index_ipcw
    from sksurv.util import Surv

    g = torch.Generator().manual_seed(0)
    n = 200
    t_event = torch.rand(n, generator=g) * 5 + 0.1
    t_cens = torch.rand(n, generator=g) * 5 + 0.1
    t = torch.minimum(t_event, t_cens)
    d = (t_event <= t_cens).double()
    risk = -t_event + 0.1 * torch.randn(n, generator=g)  # informative + noise

    y = Surv.from_arrays(d.numpy().astype(bool), t.numpy())
    # default sksurv estimate (Kaplan-Meier censoring distribution, tau=None)
    ref = concordance_index_ipcw(y, y, risk.numpy())[0]
    ours = unos_c(risk, t, d)
    # Tolerance 1e-2: sksurv's censoring-KM tie convention at exact censoring
    # times can shift weights by O(1/n); both use the default KM estimate.
    assert ours == pytest.approx(ref, abs=1e-2)
