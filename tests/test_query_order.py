"""Survival helpers must not depend on the order of the query times.

The runner passes each subject's own observed time (arbitrary order, with ties
under Type I censoring) to ``predict_surv`` for the D-calibration input and to
the tuner's validation NLL. ``interp_surv`` used to apply its monotone
cumulative-min in that caller order, which flattened every curve and made the
recorded D-calibration pass rate ~0 for the interpolating baselines.
"""

from __future__ import annotations

import numpy as np
import torch

from flowsurv.baselines.classical import CoxPH
from flowsurv.baselines.common import density_from_survival, interp_surv
from flowsurv.data.censoring import apply_censoring
from flowsurv.data.dgp import simulate_s1


def _curves(t, n_subjects=6):
    """Smooth, strictly decreasing survival curves S_j(t) = exp(-a_j t)."""
    a = np.linspace(0.3, 1.5, n_subjects)
    return np.exp(-np.outer(t, a))


def test_interp_surv_is_order_invariant():
    grid = np.linspace(0.05, 5.0, 60)
    surv = _curves(grid)
    rng = np.random.default_rng(0)
    t_sorted = np.sort(rng.uniform(0.1, 4.5, size=40))
    perm = rng.permutation(t_sorted.size)

    shuffled = interp_surv(grid, surv, t_sorted[perm])
    np.testing.assert_allclose(shuffled, interp_surv(grid, surv, t_sorted)[perm], atol=1e-12)
    # each subject's curve, read back in time order, is still monotone decreasing
    in_time_order = shuffled[np.argsort(perm)]
    assert np.all(np.diff(in_time_order, axis=0) <= 1e-12)


def test_interp_surv_shuffled_diagonal_matches_truth():
    grid = np.linspace(0.05, 5.0, 400)
    a = np.linspace(0.3, 1.5, 30)
    surv = np.exp(-np.outer(grid, a))
    rng = np.random.default_rng(1)
    t_own = rng.uniform(0.2, 4.0, size=30)  # arbitrary order, one time per subject
    diag = np.diag(interp_surv(grid, surv, t_own))
    np.testing.assert_allclose(diag, np.exp(-a * t_own), atol=5e-3)


def test_density_from_survival_is_order_invariant_and_handles_ties():
    t_unique = np.linspace(0.1, 3.0, 25)
    t = np.concatenate([t_unique, t_unique[:5]])  # ties, as under Type I censoring
    rng = np.random.default_rng(2)
    perm = rng.permutation(t.size)
    surv = _curves(t)

    dens_sorted_input = density_from_survival(np.sort(t), _curves(np.sort(t)))
    dens_shuffled = density_from_survival(t[perm], surv[perm])

    assert np.all(np.isfinite(dens_shuffled))
    assert (dens_shuffled > 0).mean() > 0.95, "density must not collapse to 0 for unsorted input"
    np.testing.assert_allclose(dens_shuffled, density_from_survival(t, surv)[perm], atol=1e-12)
    assert dens_sorted_input.shape == dens_shuffled.shape


def test_lifelines_predict_surv_matches_between_sorted_and_shuffled_times():
    t, x = simulate_s1(400, seed=3)
    t_obs, d = apply_censoring(t, 0.3, "I", seed=4)
    model = CoxPH()
    assert model.fit(t_obs, d, x, seed=0).converged

    t_q, x_q = t_obs[:60], x[:60]
    order = torch.argsort(t_q)
    s_shuffled = model.predict_surv(t_q, x_q).numpy()
    s_sorted = model.predict_surv(t_q[order], x_q).numpy()
    np.testing.assert_allclose(s_shuffled[order.numpy()], s_sorted, atol=1e-6)
