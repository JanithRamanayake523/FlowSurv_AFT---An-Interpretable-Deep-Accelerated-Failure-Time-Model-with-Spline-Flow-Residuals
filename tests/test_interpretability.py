"""Pillar C interpretability module (Methodology Sec. 5, Implementation Plan Phase 7).

Uses the shared ``s1_frozen_fit`` fixture (conftest.py): flow frozen at
identity, so FlowSurv-AFT reduces to a log-logistic AFT with mu(x) linear in
x -- directly comparable to a Weibull AFT fit on the same data, which is what
these tests check.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from lifelines import WeibullAFTFitter

from flowsurv.eval.interpretability import (
    case_study_curves,
    mu_pdp_ice,
    save_case_studies,
    save_time_ratios,
    time_ratio_table,
    weibull_lambda_coefs,
)

COV_NAMES = [f"x{i}" for i in range(10)]


def _weibull_fit(t, x):
    df = pd.DataFrame(x.numpy(), columns=COV_NAMES)
    df["duration"] = t.numpy()
    df["event"] = 1
    model = WeibullAFTFitter()
    model.fit(df, duration_col="duration", event_col="event")
    return model


def test_time_ratio_table_schema_and_positivity(s1_large, s1_frozen_fit):
    t, x = s1_large
    weibull = _weibull_fit(t, x)
    coefs = weibull_lambda_coefs(weibull)
    assert list(coefs.index) == COV_NAMES

    table = time_ratio_table(s1_frozen_fit, coefs, x[:200], COV_NAMES, dataset="s1")
    assert set(table.columns) == {"dataset", "covariate", "subject", "tr_flow", "tr_weibull"}
    assert set(table["covariate"]) == set(COV_NAMES)
    assert (table["tr_flow"] > 0).all()
    assert (table["tr_weibull"] > 0).all()
    # every subject at its own reference-crossing has TR == 1 by construction
    # is not guaranteed, but flow and weibull time ratios should be strongly
    # concordant on data the log-logistic/Weibull family fits well.
    corr = np.corrcoef(np.log(table["tr_flow"]), np.log(table["tr_weibull"]))[0, 1]
    assert corr > 0.8, f"flow vs weibull log-time-ratio correlation too low: {corr}"


def test_time_ratio_table_active_covariates_subset(s1_large, s1_frozen_fit):
    t, x = s1_large
    weibull = _weibull_fit(t, x)
    coefs = weibull_lambda_coefs(weibull)
    table = time_ratio_table(
        s1_frozen_fit, coefs, x[:50], COV_NAMES, dataset="s1", active_covariates=["x0", "x2"]
    )
    assert set(table["covariate"]) == {"x0", "x2"}
    assert len(table) == 50 * 2


def test_save_time_ratios_roundtrip(tmp_path, s1_large, s1_frozen_fit):
    t, x = s1_large
    weibull = _weibull_fit(t, x)
    coefs = weibull_lambda_coefs(weibull)
    table = time_ratio_table(s1_frozen_fit, coefs, x[:30], COV_NAMES, dataset="s1")
    path = save_time_ratios([table], path=tmp_path / "time_ratios.parquet")
    assert path.exists()
    reloaded = pd.read_parquet(path)
    pd.testing.assert_frame_equal(reloaded, table)


def test_mu_pdp_ice_shape_and_pdp_is_ice_mean(s1_large, s1_frozen_fit):
    t, x = s1_large
    x_sub = x[:40]
    grid = torch.linspace(-2.0, 2.0, 15)
    out = mu_pdp_ice(s1_frozen_fit, x_sub, covariate_idx=0, grid=grid, covariate_name="x0", dataset="s1")
    assert set(out.columns) == {"dataset", "covariate", "subject", "grid_value", "mu"}
    pdp = out[out["subject"] == -1].sort_values("grid_value")
    assert len(pdp) == len(grid)

    ice = out[out["subject"] != -1]
    assert set(ice["subject"]) == set(range(40))
    # PDP is, by construction, the mean of the ICE curves at each grid point
    ice_mean = ice.groupby("grid_value")["mu"].mean().sort_index()
    np.testing.assert_allclose(pdp.set_index("grid_value")["mu"].sort_index(), ice_mean, atol=1e-5)


def test_mu_pdp_ice_max_subjects_subsamples(s1_large, s1_frozen_fit):
    t, x = s1_large
    grid = torch.linspace(-1.0, 1.0, 5)
    out = mu_pdp_ice(s1_frozen_fit, x[:40], covariate_idx=1, grid=grid, max_subjects=5)
    ice = out[out["subject"] != -1]
    assert set(ice["subject"]) == set(range(5))


def test_case_study_curves_schema_and_positivity(s1_large, s1_frozen_fit):
    t, x = s1_large
    x_sub = x[:4]
    t_grid = torch.logspace(-3, 1, 50)
    out = case_study_curves(s1_frozen_fit, x_sub, t_grid, dataset="s1", subject_ids=["a", "b", "c", "d"])
    assert set(out.columns) == {"dataset", "subject", "t", "density", "survival", "hazard"}
    assert set(out["subject"]) == {"a", "b", "c", "d"}
    assert len(out) == 4 * len(t_grid)
    assert (out["density"] >= 0).all()
    assert (out["survival"] >= 0).all() and (out["survival"] <= 1).all()
    assert (out["hazard"] >= 0).all()
    # survival is non-increasing in t per subject
    for sid, g in out.groupby("subject"):
        s_sorted = g.sort_values("t")["survival"].to_numpy()
        assert np.all(np.diff(s_sorted) <= 1e-6)


def test_case_study_curves_subject_id_length_mismatch_raises(s1_large, s1_frozen_fit):
    t, x = s1_large
    t_grid = torch.logspace(-3, 1, 10)
    try:
        case_study_curves(s1_frozen_fit, x[:3], t_grid, subject_ids=["a", "b"])
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_save_case_studies_roundtrip(tmp_path, s1_large, s1_frozen_fit):
    t, x = s1_large
    t_grid = torch.logspace(-3, 1, 20)
    table = case_study_curves(s1_frozen_fit, x[:3], t_grid, dataset="s1")
    path = save_case_studies([table], path=tmp_path / "case_studies.parquet")
    reloaded = pd.read_parquet(path)
    pd.testing.assert_frame_equal(reloaded, table)


def test_quantile_time_ratios_and_aft_ness():
    """TR_q is flat in q (spread ~ 0) for the strict-AFT model and varies for a
    conditional flow whose spline depends on x."""
    from flowsurv.eval.interpretability import aft_ness_summary, quantile_time_ratio_table
    from flowsurv.models import FlowSurvAFT

    torch.manual_seed(0)
    x = torch.randn(40, 3)
    names = ["a", "b", "c"]

    strict = FlowSurvAFT(3, hidden=8, n_blocks=1, bins=4, conditional_flow=False)
    torch.nn.init.normal_(strict.encoder.mu_head.weight, std=0.5)
    torch.nn.init.normal_(strict.encoder.spline_head.p, std=0.3)
    cond = FlowSurvAFT(3, hidden=8, n_blocks=1, bins=4)
    torch.nn.init.normal_(cond.encoder.mu_head.weight, std=0.5)
    torch.nn.init.normal_(cond.encoder.spline_head.weight, std=0.5)

    s_tab = quantile_time_ratio_table(strict, x, names, "sim")
    c_tab = quantile_time_ratio_table(cond, x, names, "sim")
    assert set(s_tab.columns) == {"dataset", "covariate", "subject", "q", "tr_q"}
    assert len(s_tab) == 3 * 40 * 5
    assert aft_ness_summary(s_tab)["spread_mean"].max() < 1e-3
    assert aft_ness_summary(c_tab)["spread_mean"].min() > 1e-2
    # exp(mu_a - mu_b) equals TR_q for the strict-AFT model at every q
    with torch.no_grad():
        mu_a = strict.location(x)
        xb = x.clone()
        xb[:, 0] = x[:, 0].mean()
        tr = torch.exp(mu_a - strict.location(xb)).numpy()
    sub = s_tab[(s_tab.covariate == "a") & (s_tab.q == 0.5)].sort_values("subject")
    np.testing.assert_allclose(sub["tr_q"].to_numpy(), tr, rtol=1e-3)
