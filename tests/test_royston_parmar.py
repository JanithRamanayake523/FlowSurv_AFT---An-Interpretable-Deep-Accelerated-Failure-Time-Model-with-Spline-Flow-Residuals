"""Royston-Parmar flexible parametric baseline (native Python reimplementation).

Uses S1 (Weibull truth) to check the model is a numerically sound density
(integrates to 1, survival monotone, S(0)~=1) and that it is at least as
good a fit as Weibull-AFT on data the Weibull family generates -- the
classical model is nested inside a sufficiently flexible RP spline, so it
should not do meaningfully worse.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from lifelines import WeibullAFTFitter

from flowsurv.baselines.royston_parmar import RoystonParmar
from flowsurv.data.censoring import apply_censoring
from flowsurv.data.dgp import simulate_s1

N = 3000
COV_NAMES = [f"x{i}" for i in range(10)]


def _s1_censored(seed=1, censoring=0.3):
    t, x = simulate_s1(N, seed=seed)
    t_obs, d = apply_censoring(t, censoring, "I", seed=seed + 1)
    return t_obs, d, x


def test_fit_converges_and_predictions_are_sane():
    t, d, x = _s1_censored()
    model = RoystonParmar()
    res = model.fit(t, d, x, seed=0, df=4)
    assert res.converged
    assert res.info["df"] == 4

    t_test, x_test = t[:200], x[:200]
    surv = model.predict_surv(t_test, x_test).diagonal()
    haz = model.predict_hazard(t_test, x_test).diagonal()
    dens = model.predict_density(t_test, x_test).diagonal()
    assert torch.isfinite(surv).all() and (surv >= 0).all() and (surv <= 1).all()
    assert torch.isfinite(haz).all() and (haz >= 0).all()
    assert torch.isfinite(dens).all() and (dens >= 0).all()

    risk = model.predict_risk(x_test)
    assert torch.isfinite(risk).all() and (risk > 0).all()


def test_density_integrates_to_one_and_survival_is_monotone():
    t, x = simulate_s1(N, seed=2)
    model = RoystonParmar()
    res = model.fit(t, torch.ones_like(t), x, seed=0, df=4)
    assert res.converged

    x_sub = x[:5]
    grid = torch.logspace(-4, 2, 4000)
    dens = model.predict_density(grid, x_sub).numpy()
    trapz = getattr(np, "trapezoid", None) or np.trapz
    integral = trapz(dens, grid.numpy(), axis=0)
    np.testing.assert_allclose(integral, 1.0, atol=5e-3)

    surv = model.predict_surv(grid, x_sub).numpy()
    assert (np.diff(surv, axis=0) <= 1e-6).all()
    np.testing.assert_allclose(surv[0], 1.0, atol=1e-3)
    np.testing.assert_allclose(surv[-1], 0.0, atol=1e-3)


def test_matches_weibull_aft_likelihood_on_weibull_truth():
    """On S1 (Weibull DGP), a df=4 RP spline should not do meaningfully worse
    than Weibull-AFT itself -- the Weibull family is nested inside a
    sufficiently flexible hazard-scale spline (2-knot RP *is* Weibull)."""
    t, d, x = _s1_censored(seed=3, censoring=0.2)

    df_weibull = pd.DataFrame(x.numpy(), columns=COV_NAMES)
    df_weibull["duration"] = t.numpy()
    df_weibull["event"] = d.numpy().astype(int)
    wb = WeibullAFTFitter()
    wb.fit(df_weibull, duration_col="duration", event_col="event")
    weibull_nll = -wb.log_likelihood_ / t.shape[0]

    rp = RoystonParmar()
    res = rp.fit(t, d, x, seed=0, df=4)
    assert res.converged

    assert res.info["train_nll"] < weibull_nll + 0.05


def test_df_selection_uses_validation_split_when_provided():
    t, d, x = _s1_censored(seed=4)
    n_val = 400
    t_tr, d_tr, x_tr = t[:-n_val], d[:-n_val], x[:-n_val]
    t_val, d_val, x_val = t[-n_val:], d[-n_val:], x[-n_val:]

    model = RoystonParmar()
    res = model.fit(t_tr, d_tr, x_tr, val=(t_val, d_val, x_val), seed=0)
    assert res.converged
    assert res.info["df"] in (3, 4, 5)


def test_no_validation_split_defaults_to_df_four():
    t, d, x = _s1_censored(seed=5)
    model = RoystonParmar()
    res = model.fit(t, d, x, seed=0)
    assert res.converged
    assert res.info["df"] == 4


def test_predict_before_fit_raises():
    model = RoystonParmar()
    t = torch.linspace(0.1, 1.0, 5)
    x = torch.zeros(5, 10)
    try:
        model.predict_surv(t, x)
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass
