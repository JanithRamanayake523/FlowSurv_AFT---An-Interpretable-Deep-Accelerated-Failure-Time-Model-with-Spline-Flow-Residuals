"""Unit tests for the IPCW integrated Brier score (Phase 4, Methodology Sec. 3.3)."""

import math

import pytest
import torch

from flowsurv.metrics import integrated_brier_score


def test_perfect_prediction_no_censoring_is_zero():
    """S = step at the true event times -> BS(s) = 0 everywhere -> IBS = 0."""
    t = torch.tensor([1.0, 2.0, 4.0])
    d = torch.ones_like(t)
    grid = torch.linspace(0.0, 5.0, 51)
    surv = (grid.unsqueeze(1) < t.unsqueeze(0)).double()  # S(s|x_i) = 1{s < t_i}
    ibs = integrated_brier_score(surv, grid, t, d, tau=5.0)
    assert ibs == pytest.approx(0.0, abs=1e-12)


def test_default_tau_is_90th_percentile_of_observed_times():
    t = torch.linspace(1.0, 10.0, 20)
    d = torch.ones_like(t)
    grid = torch.linspace(0.0, 10.0, 101)
    surv = torch.full((grid.numel(), t.numel()), 0.5, dtype=torch.float64)
    default = integrated_brier_score(surv, grid, t, d)
    explicit = integrated_brier_score(surv, grid, t, d, tau=float(torch.quantile(t, 0.9)))
    assert default == pytest.approx(explicit)


def test_all_censored_fold_is_defined():
    """All-censored test fold: IBS still defined, weights from G(s) only."""
    t = torch.tensor([1.0, 2.0, 3.0])
    d = torch.zeros_like(t)
    grid = torch.linspace(0.0, 3.0, 31)
    surv = torch.full((grid.numel(), 3), 0.5, dtype=torch.float64)
    ibs = integrated_brier_score(surv, grid, t, d, tau=3.0)
    assert math.isfinite(ibs) and ibs >= 0.0


def test_matches_sksurv_integrated_brier_score():
    pytest.importorskip("sksurv")
    from sksurv.metrics import integrated_brier_score as sksurv_ibs
    from sksurv.util import Surv

    g = torch.Generator().manual_seed(1)
    n = 100
    t_event = torch.rand(n, generator=g) * 5 + 0.1
    t_cens = torch.rand(n, generator=g) * 5 + 0.1
    t = torch.minimum(t_event, t_cens)
    d = (t_event <= t_cens).double()
    grid = torch.linspace(float(t.min()) + 1e-4, float(t.max()) - 1e-4, 50)
    # one smooth decreasing survival curve for all subjects, (m, n) layout
    surv = torch.sigmoid(1.0 - grid).unsqueeze(1).expand(-1, n).contiguous()

    y = Surv.from_arrays(d.numpy().astype(bool), t.numpy())
    # sksurv takes (n_samples, n_times); ours is (n_times, n_samples)
    ref = sksurv_ibs(y, y, surv.T.numpy(), grid.numpy())
    ours = integrated_brier_score(surv, grid, t, d, tau=float(grid[-1]))
    # Tolerance 1e-2: sksurv normalizes by (times[-1] - times[0]) with
    # times[0] = 0.01 while we normalize by tau, and censoring-KM tie
    # conventions at exact censoring times may differ by O(1/n) per weight.
    assert ours == pytest.approx(ref, abs=1e-2)
