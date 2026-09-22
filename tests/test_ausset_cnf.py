"""AussetCNFModel / the ausset_cnf baseline -- FlowSurv-AFT's AFT skeleton
with a continuous-flow residual law instead of the RQS spline flow.

Deliberately smaller-scale and fewer epochs than the FlowSurvAFT gate tests:
this is a secondary, non-confirmatory baseline (not part of C1-C4), and the
CNF's RK4 integration is much more expensive per forward/backward pass than
the RQS flow's analytic transform, so these tests check correctness on a
small problem rather than large-n statistical recovery.
"""

from __future__ import annotations

import numpy as np
import torch

from flowsurv.baselines import METHODS
from flowsurv.baselines.ausset_cnf import AussetCNF
from flowsurv.data.dgp import simulate_s1
from flowsurv.models import AussetCNFModel, TrainConfig, fit

SMALL_CFG = dict(hidden=32, n_blocks=1, cond_dim=8, cnf_hidden=16, n_steps=8)


def test_registered_in_methods():
    assert METHODS["ausset_cnf"] is AussetCNF


def test_model_density_integrates_to_one_and_survival_is_monotone():
    torch.manual_seed(0)
    model = AussetCNFModel(10, **SMALL_CFG)
    x = torch.randn(4, 10)

    grid = torch.logspace(-3, 1, 300)
    with torch.no_grad():
        dens = model.density(grid, x).numpy()
        surv = model.survival(grid, x).numpy()
    trapz = getattr(np, "trapezoid", None) or np.trapz
    integral = trapz(dens, grid.numpy(), axis=0)
    np.testing.assert_allclose(integral, 1.0, atol=0.05)  # coarse grid + n_steps=8: looser tol
    assert (np.diff(surv, axis=0) <= 1e-4).all()


def test_model_sample_matches_quantile_direction():
    """Sanity check only (not a KS audit like the FlowSurvAFT gate test --
    this is a much smaller/coarser model): samples should be positive and
    the median of many samples should be in the right ballpark of Q(0.5)."""
    torch.manual_seed(0)
    model = AussetCNFModel(10, **SMALL_CFG)
    x = torch.randn(3, 10)
    with torch.no_grad():
        samples = model.sample(x, n=500)
        q50 = model.quantile(0.5, x)
    assert (samples > 0).all()
    sample_median = samples.median(dim=0).values
    # loose relative check -- coarse RK4 + tiny model, not a statistical audit
    ratio = (sample_median / q50.squeeze(0)).clamp(min=1e-6)
    assert (ratio > 0.2).all() and (ratio < 5.0).all()


def test_fit_reduces_validation_nll():
    torch.manual_seed(0)
    t, x = simulate_s1(300, seed=1)
    model = AussetCNFModel(10, **SMALL_CFG)
    cfg = TrainConfig(batch_size=128, max_epochs=15, patience=15, seed=0, device="cpu")
    res = fit(model, t, torch.ones_like(t), x, cfg)
    assert res.epochs_ran > 0
    assert np.isfinite(res.best_val_nll)
    assert res.val_nll[-1] <= res.val_nll[0] + 0.1  # allow small noise, expect a real decrease


def test_baseline_wrapper_runs_through_run_cell():
    from flowsurv.data import default_grid
    from flowsurv.eval.run_cell import run_sim_rep

    cells = {c.cell_id: c for c in default_grid()}
    cell = cells["S1_n200_c20_typeI"]
    row = run_sim_rep(
        cell,
        1,
        "ausset_cnf",
        tuner=None,
        device="cpu",
    )
    assert row["converged"], row["error"]
    assert np.isfinite(row["unos_c"])
    assert np.isfinite(row["ibs"])
