"""Baseline hazards must be computed from float64 survival.

Cox / Weibull-AFT / log-normal AFT / RSF / DeepSurv / DeepHit derive their
hazard by differentiating the predicted survival curve. Passing that curve
through the float32 interface tensors corrupted the ~1e-30 late-time survival
of high-hazard subjects and produced spikes that dominated HRE (Weibull-AFT on
S1, n=5000, no censoring: HRE 801.7 with the float32 path vs 0.225 in float64).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from flowsurv.baselines.classical import CoxPH, WeibullAFT
from flowsurv.baselines.common import hazard_from_survival
from flowsurv.data.cells import cell_seed, default_grid, generate_dataset, split_train_val_test
from flowsurv.eval.run_cell import eval_grid
from flowsurv.metrics import hazard_recovery_error


def test_hazard_from_survival_accurate_for_tiny_survival_and_any_order():
    # S(t) = exp(-a t^2)  ->  h(t) = 2 a t ;  S reaches ~1e-200, far below float32 range
    t = np.linspace(0.05, 12.0, 400)
    a = np.array([0.5, 2.0, 5.0])
    surv = np.exp(-np.outer(t ** 2, a))
    assert surv.min() < 1e-100

    h = hazard_from_survival(t, surv)
    inner = slice(5, -5)
    np.testing.assert_allclose(h[inner], np.outer(2 * t, a)[inner], rtol=2e-2)

    rng = np.random.default_rng(0)
    perm = rng.permutation(t.size)
    np.testing.assert_allclose(hazard_from_survival(t[perm], surv[perm]), h[perm], rtol=1e-9)

    tied = np.concatenate([t, t[:10]])
    assert np.all(np.isfinite(hazard_from_survival(tied, np.exp(-np.outer(tied ** 2, a)))))


def _s1_high_hazard_case():
    cell = {c.cell_id: c for c in default_grid()}["S1_n5000_c0_typeI"]
    data = generate_dataset(cell, 1)
    seed = cell_seed(cell.cell_id, 1)
    sp = split_train_val_test(data["t"], data["d"], data["x"], seed=seed)
    grid = eval_grid(sp["test"]["t"])
    h_true = data["truth"]["hazard"](grid, sp["test"]["x"])
    return sp, grid, h_true, seed


@pytest.mark.parametrize("cls,max_hre", [(WeibullAFT, 2.0), (CoxPH, 5.0)])
def test_baseline_hre_has_no_float32_tail_spikes(cls, max_hre):
    sp, grid, h_true, seed = _s1_high_hazard_case()
    assert float(h_true.max()) > 100, "test needs subjects with very high true hazard"
    model = cls()
    assert model.fit(sp["train"]["t"], sp["train"]["d"], sp["train"]["x"], seed=seed).converged

    h = torch.as_tensor(model.predict_hazard(grid, sp["test"]["x"]), dtype=torch.float32)
    hre = float(hazard_recovery_error(h, h_true, grid))
    assert hre < max_hre, f"{cls.__name__} HRE {hre:.2f} (was ~800 with float32 survival)"
    assert float(h.max()) < 1000.0, "hazard must not sit at the old clip ceiling"
