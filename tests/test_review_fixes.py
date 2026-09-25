"""Regression tests for the supervisor-review model/training fixes (2026-09-24).

External validation split, explicit paired/grid mode, log-time centring,
spline-derivative penalty, strict-AFT and Gumbel ablations, quantile helper,
cumulative-hazard companion metric.
"""

from __future__ import annotations

import numpy as np
import torch

from flowsurv.baselines import ABLATION_METHODS, METHODS
from flowsurv.baselines.flowsurv_wrappers import FlowSurvAFTMethod, FlowSurvGumbelMethod
from flowsurv.data.censoring import apply_censoring
from flowsurv.data.dgp import simulate_s1
from flowsurv.metrics import cumulative_hazard_error
from flowsurv.models import FlowSurvAFT, FlowSurvGumbel, TrainConfig, fit, right_censored_nll


def _data(n=200, seed=0):
    t, x = simulate_s1(n, seed=seed)
    t, d = apply_censoring(t, 0.3, "I", seed=seed + 1)
    return t.float(), d.float(), x.float()


# --------------------------------------------------------------- item 4: external val
def test_external_val_is_used_for_early_stopping_and_all_train_data_trains():
    t, d, x = _data(300)
    tr, va = slice(0, 210), slice(210, 300)
    model = FlowSurvAFT(x.shape[1], hidden=16, n_blocks=1)
    cfg = TrainConfig(max_epochs=8, patience=8, seed=0)
    res = fit(model, t[tr], d[tr], x[tr], cfg, val=(t[va], d[va], x[va]))
    with torch.no_grad():
        val_nll = right_censored_nll(model, t[va], d[va], x[va]).item()
    assert abs(val_nll - res.best_val_nll) < 1e-4  # best weights restored on the external val
    assert len(res.val_nll) == res.epochs_ran


def test_wrapper_passes_external_val_through():
    t, d, x = _data(300)
    method = FlowSurvAFTMethod()
    out = method.fit(t[:210], d[:210], x[:210], val=(t[210:], d[210:], x[210:]), seed=0, max_epochs=5, hidden=16, n_blocks=1)
    assert out.converged and np.isfinite(out.info["best_val_nll"])


# ----------------------------------------------------- item 8: explicit paired/grid mode
def test_grid_and_paired_are_distinguishable_when_grid_size_equals_n():
    t, d, x = _data(200)  # 200 subjects == 200 grid points
    method = FlowSurvAFTMethod()
    method.fit(t, d, x, seed=0, max_epochs=3, hidden=16, n_blocks=1)
    grid = torch.logspace(-2, 1, 200)
    s_grid = torch.as_tensor(method.predict_surv(grid, x, paired=False))
    s_pair = torch.as_tensor(method.predict_surv(t, x, paired=True))
    assert s_grid.shape == (200, 200)
    assert s_pair.shape == (200,)
    # the paired values are the diagonal of the full (times x subjects) matrix
    full = torch.as_tensor(method.predict_surv(t, x, paired=False))
    assert torch.allclose(full.diagonal(), s_pair, atol=1e-6)


# ------------------------------------------------------------ item 10: log-time centring
def test_mu_offset_centres_the_warm_start_and_keeps_exact_density():
    model = FlowSurvAFT(3, hidden=8, n_blocks=1)
    model.encoder.mu_offset.fill_(7.0)  # e.g. log-days
    x = torch.zeros(1, 3)
    med = model.quantile(torch.tensor([0.5]), x)
    assert torch.allclose(med, torch.exp(torch.tensor(7.0)).reshape(1, 1), rtol=1e-4)
    t = torch.logspace(0, 9, 4000, dtype=torch.float64).float()
    dens = model.density(t, x, paired=False).squeeze(1).detach().double()
    mass = torch.trapezoid(dens, t.double())
    assert abs(float(mass) - 1.0) < 0.02


def test_wrapper_sets_offset_to_mean_log_time():
    t, d, x = _data(200)
    method = FlowSurvAFTMethod()
    method.fit(t * 1000.0, d, x, seed=0, max_epochs=2, hidden=16, n_blocks=1)
    expected = float((t * 1000.0).log().mean())
    assert abs(float(method.model.encoder.mu_offset) - expected) < 1e-4


# ------------------------------------------------- item 5: spline penalty, bound, ablations
def test_spline_penalty_zero_at_identity_and_positive_when_perturbed():
    model = FlowSurvAFT(3, hidden=8, n_blocks=1, bins=4)
    x = torch.randn(16, 3)
    assert float(model.spline_derivative_penalty(x)) == 0.0
    torch.nn.init.normal_(model.encoder.spline_head.weight, std=0.5)
    assert float(model.spline_derivative_penalty(x)) > 0.0


def test_wrapper_defaults_match_preregistration():
    m = FlowSurvAFTMethod()._make_model(5)
    assert m.flow.bound == 4.0
    assert FlowSurvAFTMethod._spline_l2 == 1e-5


def test_strict_aft_flow_is_unconditional():
    model = FlowSurvAFT(3, hidden=8, n_blocks=1, bins=4, conditional_flow=False)
    torch.nn.init.normal_(model.encoder.spline_head.p, std=0.5)
    _mu, _sig, params = model.encoder(torch.randn(10, 3))
    assert torch.allclose(params, params[0].expand_as(params))  # same spline for every subject
    # quantile ratios are then constant in q: an exact AFT time ratio
    xa, xb = torch.randn(1, 3), torch.randn(1, 3)
    torch.nn.init.normal_(model.encoder.mu_head.weight, std=0.5)
    torch.nn.init.normal_(model.encoder.sigma_head.p, std=0.5)  # global sigma, still x-free
    model.eval()  # dropout off: mu must be identical across the tiled quantile levels
    q = torch.tensor([0.1, 0.5, 0.9])
    ratio = model.quantiles(q, xa) / model.quantiles(q, xb)
    assert torch.allclose(ratio, ratio[0].expand_as(ratio), rtol=1e-4)
    # and the conditional model does NOT have this property once sigma varies with x
    cond = FlowSurvAFT(3, hidden=8, n_blocks=1, bins=4)
    torch.nn.init.normal_(cond.encoder.sigma_head.weight, std=0.5)
    cond.eval()
    r2 = cond.quantiles(q, xa) / cond.quantiles(q, xb)
    assert not torch.allclose(r2, r2[0].expand_as(r2), rtol=1e-3)


def test_ablations_registered_but_not_in_default_grid():
    assert {"flowsurv_strict_aft", "flowsurv_gumbel"} <= set(METHODS)
    assert ABLATION_METHODS == {"flowsurv_strict_aft", "flowsurv_gumbel"}


# ------------------------------------------------------------- Gumbel ablation (item 2)
def test_gumbel_is_a_weibull_aft_and_internally_consistent():
    model = FlowSurvGumbel(2, hidden=8, n_blocks=1)
    torch.nn.init.normal_(model.encoder.mu_head.weight, std=0.3)
    torch.nn.init.normal_(model.encoder.sigma_head.weight, std=0.3)
    model.eval()
    x = torch.randn(4, 2)
    t = torch.logspace(-3, 3, 3000)
    with torch.no_grad():
        dens = model.density(t, x, paired=False).double()
        surv = model.survival(t, x, paired=False).double()
        cdf = model.cdf(t, x, paired=False).double()
        assert torch.allclose(surv + cdf, torch.ones_like(surv), atol=1e-5)
        mass = torch.trapezoid(dens, t.double(), dim=0)
        assert (mass - (1 - surv[-1])).abs().max() < 0.03  # density integrates to F(t_max)
        q = torch.tensor([0.25, 0.5, 0.75])
        tq = model.quantiles(q, x)
        for k, qk in enumerate(q):
            f = model.cdf(tq[k], x, paired=True)
            assert torch.allclose(f, torch.full_like(f, float(qk)), atol=1e-4)


def test_gumbel_wrapper_fits():
    t, d, x = _data(150)
    m = FlowSurvGumbelMethod()
    res = m.fit(t, d, x, seed=0, max_epochs=3, hidden=16, n_blocks=1)
    assert res.converged


# ------------------------------------------------------------------ item 3: quantile helper
def test_quantiles_matches_scalar_quantile_calls():
    model = FlowSurvAFT(3, hidden=8, n_blocks=1, bins=4)
    torch.nn.init.normal_(model.encoder.spline_head.weight, std=0.2)
    model.eval()
    x = torch.randn(5, 3)
    q = torch.tensor([0.1, 0.5, 0.9])
    stacked = torch.stack([model.quantile(qk.reshape(1), x).reshape(-1) for qk in q])
    assert torch.allclose(model.quantiles(q, x), stacked, rtol=1e-5)


# ------------------------------------------------------------- item 9: robust HRE companion
def test_cumulative_hazard_error_zero_at_truth_and_robust_to_spike():
    grid = torch.logspace(-2, 1, 200)
    h_true = torch.ones(200, 6)
    assert cumulative_hazard_error(h_true, h_true, grid) == 0.0
    h_spike = h_true.clone()
    h_spike[100, 0] = 1e6  # one grid point of one subject
    from flowsurv.metrics import hazard_recovery_error

    hre = hazard_recovery_error(h_spike, h_true, grid)
    che = cumulative_hazard_error(h_spike, h_true, grid)
    assert che < hre  # the spike is far less dominant on the cumulative scale


# ------------------------------------------- Deviation 6: HRE truncated at each subject's own Q_true(0.9)
def test_truncated_hre_ignores_hazard_error_beyond_each_subjects_window():
    from flowsurv.metrics import hazard_recovery_error, truncated_hazard_recovery_error

    grid = torch.linspace(0.01, 10.0, 400)
    n = 5
    h_true = torch.ones(400, n)
    cdf_true = 1 - torch.exp(-grid.unsqueeze(1) * torch.linspace(0.5, 2.0, n).unsqueeze(0))  # subject-specific
    q90 = -torch.log(torch.tensor(0.1)) / torch.linspace(0.5, 2.0, n)  # true 0.9-quantile per subject
    # hazard error only where every subject's true CDF is already >= 0.9 (t > max Q90 ~ 4.6)
    h_pred = h_true.clone()
    h_pred[grid > float(q90.max()) + 0.05] += 50.0
    assert truncated_hazard_recovery_error(h_pred, h_true, cdf_true, grid) < 1e-9
    assert hazard_recovery_error(h_pred, h_true, grid) > 1e-3  # pooled HRE does see it


def test_truncated_hre_is_a_weighted_average_over_the_window():
    from flowsurv.metrics import truncated_hazard_recovery_error

    grid = torch.linspace(0.01, 10.0, 400)
    h_true = torch.ones(400, 3)
    cdf_true = 1 - torch.exp(-grid.unsqueeze(1) * torch.tensor([0.5, 1.0, 2.0]).unsqueeze(0))
    h_pred = h_true + 0.3  # constant error 0.3 everywhere -> squared error 0.09 in any window
    val = truncated_hazard_recovery_error(h_pred, h_true, cdf_true, grid)
    assert abs(val - 0.09) < 5e-3  # renormalized per subject, so window length does not matter


# ------------------------------------------ item 4: predictions saved, metrics recomputable offline
def test_saved_predictions_reproduce_the_metrics(tmp_path):
    from flowsurv.data import default_grid
    from flowsurv.eval.predictions import load_predictions
    from flowsurv.eval.run_cell import run_sim_rep
    from flowsurv.metrics import d_calibration, ici, integrated_brier_score, unos_c

    cell = next(c for c in default_grid() if c.cell_id == "S1_n200_c20_typeI")
    row = run_sim_rep(cell, 1, "cox_ph", predictions_dir=str(tmp_path))
    assert row["converged"]
    p = load_predictions(tmp_path, cell.cell_id, 1, "cox_ph")
    assert {"grid", "surv", "s_at_obs", "s_at_tau", "hazard", "risk", "t", "d", "tau"} <= set(p)
    t, d, tau = torch.tensor(p["t"]), torch.tensor(p["d"]), float(p["tau"])
    assert np.isclose(unos_c(torch.tensor(p["risk"]), t, d, tau=tau), row["unos_c"], atol=1e-6)
    assert np.isclose(integrated_brier_score(torch.tensor(p["surv"]), torch.tensor(p["grid"]), t, d, tau=tau), row["ibs"], atol=1e-6)
    assert np.isclose(d_calibration(torch.tensor(p["s_at_obs"]), d).statistic, row["dcal_stat"], atol=1e-5)
    assert np.isclose(ici(torch.tensor(p["s_at_tau"]), t, d, tau=tau), row["ici"], atol=1e-6)


# ------------------------------- DeepHit risk score must not depend on a fixed 0-10 time grid
def test_median_from_curves_interpolates_and_orders_long_survivors():
    from flowsurv.baselines.deep import _median_from_curves

    times = np.array([100.0, 200.0, 300.0, 400.0])
    surv = np.array(
        [
            [0.9, 0.6, 0.99, 0.95],
            [0.7, 0.4, 0.98, 0.90],
            [0.4, 0.2, 0.97, 0.85],
            [0.2, 0.1, 0.96, 0.80],
        ]
    )
    med = _median_from_curves(times, surv)
    assert np.isclose(med[0], 200.0 + 100.0 * (0.7 - 0.5) / (0.7 - 0.4))  # crosses 0.5 between t=200 and 300
    assert np.isclose(med[1], 100.0 + 100.0 * (0.6 - 0.5) / (0.6 - 0.4))
    # columns 2 and 3 never reach 0.5: beyond the last cut, ordered by S_last (0.96 > 0.80)
    assert med[2] > med[3] > 400.0


def test_deephit_risk_is_not_constant_when_times_are_in_days():
    from flowsurv.baselines import DeepHit

    t, d, x = _data(300)
    t = t * 1000.0  # days-scale times: the old fixed grid (0-10) tied every subject
    m = DeepHit()
    res = m.fit(t, d, x, seed=0, max_epochs=5, hidden=16, n_blocks=1, num_durations=20)
    assert res.converged
    risk = torch.as_tensor(m.predict_risk(x)).numpy()
    assert len(np.unique(risk)) > 10


# ---------------- validation folds with no events: explicit failure, and the tuner survives them
def _split_with_event_free_val(n=300):
    t, d, x = _data(n)
    tr, va = slice(0, 210), slice(210, n)
    return (t[tr], d[tr], x[tr]), (t[va], torch.zeros_like(d[va]), x[va])


def test_deepsurv_and_deephit_report_failure_when_val_fold_has_no_events():
    from flowsurv.baselines import DeepHit, DeepSurv

    train, val = _split_with_event_free_val()
    for cls in (DeepSurv, DeepHit):
        res = cls().fit(*train, val=val, seed=0, max_epochs=3, hidden=16, n_blocks=1)
        assert not res.converged
        assert "no events" in res.info["error"]


def test_tuner_survives_a_fold_without_validation_events():
    from flowsurv.eval.tune import tune_method

    (t, d, x), (tv, dv, xv) = _split_with_event_free_val()
    good_t, good_d, good_x = _data(300, seed=5)
    datasets = [
        {"train": {"t": t, "d": d, "x": x}, "val": {"t": good_t[210:], "d": good_d[210:], "x": good_x[210:]}},
        {"train": {"t": t, "d": d, "x": x}, "val": {"t": tv, "d": dv, "x": xv}},  # no events
    ]
    best, log = tune_method("deepsurv", datasets, n_configs=2)
    assert best  # a config was chosen from the good fold
    assert not log["converged_all"].any()  # the bad fold is flagged, not fatal
    assert np.isfinite(log["mean_val_score"]).all()


def test_failed_fit_row_keeps_the_real_reason(monkeypatch):
    from flowsurv.data import default_grid
    from flowsurv.data.cells import cell_seed, generate_dataset, split_train_val_test
    from flowsurv.eval import run_cell

    cell = next(c for c in default_grid() if c.cell_id == "S1_n200_c80_typeI")
    # force an event-free validation fold through the public path
    real_split = run_cell.split_train_val_test

    def event_free_val(t, d, x, seed):
        sp = real_split(t, d, x, seed=seed)
        sp["val"]["d"] = torch.zeros_like(sp["val"]["d"])
        return sp

    monkeypatch.setattr(run_cell, "split_train_val_test", event_free_val)
    row = run_cell.run_sim_rep(cell, 1, "deepsurv")
    assert row["converged"] is False
    assert "no events" in row["error"]
