"""One (cell, replication, method) -> one metrics row (Implementation Plan Phase 3/6).

Evaluation grid convention (Methodology Sec. 3.3): ``METHOD_EVAL_GRID = 200``
time points, log-spaced from 1e-4 to tau, where tau is the 90th percentile of
the observed test times. Log spacing concentrates points at early times where
hazards/densities vary fastest; tau matches the IBS/ICI horizon.

Every failure (fit or evaluation) is captured in the row with
``converged=False`` and ``error=repr(e)`` -- failure rates are a pre-registered
finding (prereg Sec. 5), so this module never raises.

CLI:
    python -m flowsurv.eval.run_cell --cell S1_weibull_n200_c0_typeI --rep 1 --method flowsurv_aft
    python -m flowsurv.eval.run_cell --dataset support --rep 1 --method flowsurv_aft
"""

from __future__ import annotations

import argparse
import json
import math
import time

import torch

from ..baselines import METHODS
from ..baselines.flowsurv_wrappers import _FlowSurvWrapper
from ..data import (
    REAL_DATASETS,
    CellConfig,
    cell_seed,
    default_grid,
    generate_dataset,
    real_split,
    split_train_val_test,
)
from ..metrics import (
    cdf_fidelity,
    d_calibration,
    hazard_recovery_error,
    ici,
    integrated_brier_score,
    unos_c,
)
from .tune import FrozenTuner, RealCell, inner_val_split

METHOD_EVAL_GRID = 200
GRID_LO = 1e-4
TAU_QUANTILE = 0.90

# Single source of truth for metrics.parquet (see eval design doc / prereg Sec. 3).
METRICS_COLUMNS = [
    "cell_id",
    "scenario",
    "n",
    "censoring",
    "censoring_type",
    "rep",
    "method",
    "unos_c",
    "ibs",
    "dcal_stat",
    "dcal_p",
    "dcal_pass",
    "ici",
    "hre",
    "ks",
    "w1",
    "time_fit_s",
    "time_eval_s",
    "time_sample_s",
    "converged",
    "error",
    "tuned",
]


def eval_grid(t_obs, n_grid: int = METHOD_EVAL_GRID) -> torch.Tensor:
    """Log-spaced evaluation grid from 1e-4 to tau = 90th pct of observed times."""
    t = torch.as_tensor(t_obs, dtype=torch.float32).flatten()
    tau = float(torch.quantile(t, TAU_QUANTILE))
    return torch.logspace(math.log10(GRID_LO), math.log10(tau), n_grid)


def _empty_row(
    cell_id: str,
    scenario: str,
    n: float,
    censoring: float,
    censoring_type: str,
    rep: int,
    method_name: str,
) -> dict:
    row = {col: math.nan for col in METRICS_COLUMNS}
    row.update(
        cell_id=cell_id,
        scenario=scenario,
        n=int(n),
        censoring=float(censoring),
        censoring_type=censoring_type,
        rep=int(rep),
        method=method_name,
        converged=False,
        error="",
        tuned=False,
    )
    return row


def _per_subject(pred) -> torch.Tensor:
    """Diagonal of an (n, n) prediction at the subjects' own observed times."""
    pred = torch.as_tensor(pred, dtype=torch.float32)
    if pred.dim() == 2:
        pred = pred.diagonal()
    return pred


def _evaluate(method, fit_result, test: dict, truth: dict | None, row: dict) -> None:
    """Fill the metric columns of ``row`` on the test fold. May raise."""
    t, d, x = test["t"], test["d"], test["x"]
    grid = eval_grid(t)
    tau = float(grid[-1])

    t0 = time.perf_counter()
    risk = method.predict_risk(x)
    surv = method.predict_surv(grid, x)
    s_at_obs = _per_subject(method.predict_surv(t, x))
    s_at_tau = torch.as_tensor(
        method.predict_surv(torch.tensor([tau]), x), dtype=torch.float32
    ).reshape(-1)
    h_pred = method.predict_hazard(grid, x) if getattr(method, "supports_hazard", False) else None
    row["time_eval_s"] = time.perf_counter() - t0

    row["time_fit_s"] = float(fit_result.wall_time_s)
    row["unos_c"] = float(unos_c(risk, t, d))
    row["ibs"] = float(integrated_brier_score(surv, grid, t, d, tau=tau))
    dcal = d_calibration(s_at_obs, d)
    row["dcal_stat"] = float(dcal.statistic)
    row["dcal_p"] = float(dcal.pvalue)
    row["dcal_pass"] = float(dcal.passed)  # float so group means are pass rates
    row["ici"] = float(ici(s_at_tau, t, d, tau=tau))

    if truth is not None:
        if h_pred is not None:
            h_true = truth["hazard"](grid, x)
            row["hre"] = float(hazard_recovery_error(h_pred, h_true, grid))
        f_pred = 1.0 - torch.as_tensor(surv, dtype=torch.float32)
        f_true = truth["cdf"](grid, x)
        fidelity = cdf_fidelity(f_pred, f_true, grid)
        row["ks"] = float(fidelity["ks"])
        row["w1"] = float(fidelity["w1"])

    sample_fn = getattr(method, "sample", None)
    if callable(sample_fn):
        t0 = time.perf_counter()
        sample_fn(x, 1000)  # 1,000 samples per subject (prereg Sec. 3, cost metric)
        row["time_sample_s"] = time.perf_counter() - t0


def _fit_config(method_name: str, method, tuner: FrozenTuner | None, cell, rep: int, device: str) -> dict:
    """Frozen config for deep methods; {} otherwise (prereg Sec. 4)."""
    config: dict = {}
    if tuner is not None and getattr(method, "is_deep", False):
        config = dict(tuner.tuned_config(method_name, cell, rep=rep))
    # _FlowSurvWrapper subclasses (flowsurv_aft/gauss, ausset_cnf) map hypers
    # onto TrainConfig, which owns `device`; name-prefix matching missed
    # ausset_cnf and silently left it on the wrapper's CPU default.
    if isinstance(method, _FlowSurvWrapper):
        config.setdefault("device", device)
    return config


def run_sim_rep(
    cell: CellConfig,
    rep: int,
    method_name: str,
    tuner: FrozenTuner | None = None,
    device: str = "cpu",
) -> dict:
    """Run one (simulation cell, replication, method) and return a metrics row.

    Never raises: any exception is captured as ``converged=False`` +
    ``error=repr(e)`` with NaN metrics.
    """
    row = _empty_row(cell.cell_id, cell.scenario, cell.n, cell.censoring, cell.censoring_type, rep, method_name)
    try:
        data = generate_dataset(cell, rep)
        seed = cell_seed(cell.cell_id, rep)  # common seeds across methods (prereg Sec. 4)
        splits = split_train_val_test(data["t"], data["d"], data["x"], seed=seed)
        method = METHODS[method_name]()
        row["tuned"] = tuner is not None and getattr(method, "is_deep", False)
        config = _fit_config(method_name, method, tuner, cell, rep, device)
        fit_result = method.fit(
            splits["train"]["t"],
            splits["train"]["d"],
            splits["train"]["x"],
            val=(splits["val"]["t"], splits["val"]["d"], splits["val"]["x"]),
            seed=seed,
            **config,
        )
        row["converged"] = bool(fit_result.converged)
        _evaluate(method, fit_result, splits["test"], data.get("truth"), row)
    except Exception as e:  # noqa: BLE001 -- failures are data (prereg Sec. 5)
        row["converged"] = False
        row["error"] = repr(e)
    return row


def run_real_rep(
    dataset_name: str,
    rep: int,
    method_name: str,
    tuner: FrozenTuner | None = None,
    device: str = "cpu",
) -> dict:
    """Run one (real dataset, split, method) and return a metrics row.

    Same row schema as :func:`run_sim_rep`; no ground truth, so hre/ks/w1 are
    NaN. 80/20 stratified split via ``real_split``; 15% of the train fold is
    the inner validation split (seeded by rep; prereg Sec. 4). Never raises.
    """
    data = REAL_DATASETS[dataset_name]()
    t, d, x = data["t"], data["d"], data["x"]
    n = int(len(t))
    censoring = float(1.0 - torch.as_tensor(d, dtype=torch.float32).mean())
    row = _empty_row(f"{dataset_name}_real", dataset_name, n, censoring, "real", rep, method_name)
    try:
        splits = real_split(t, d, x, rep)
        train2, val = inner_val_split(splits["train"], val_frac=0.15, seed=rep)
        cell = RealCell(cell_id=row["cell_id"], name=dataset_name)
        method = METHODS[method_name]()
        row["tuned"] = tuner is not None and getattr(method, "is_deep", False)
        config = _fit_config(method_name, method, tuner, cell, rep, device)
        fit_result = method.fit(
            train2["t"],
            train2["d"],
            train2["x"],
            val=(val["t"], val["d"], val["x"]),
            seed=rep,
            **config,
        )
        row["converged"] = bool(fit_result.converged)
        _evaluate(method, fit_result, splits["test"], None, row)
    except Exception as e:  # noqa: BLE001
        row["converged"] = False
        row["error"] = repr(e)
    return row


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--cell", help="simulation cell id, e.g. S1_weibull_n200_c0_typeI")
    source.add_argument("--dataset", help="real dataset name, e.g. support")
    parser.add_argument("--rep", type=int, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--tune",
        action="store_true",
        help="use a FrozenTuner (frozen per macro-cell); default is untuned defaults",
    )
    parser.add_argument("--tuning-dir", default="experiments/tuning")
    args = parser.parse_args(argv)

    tuner = FrozenTuner(args.tuning_dir) if args.tune else None
    if args.dataset:
        row = run_real_rep(args.dataset, args.rep, args.method, tuner=tuner, device=args.device)
    else:
        cells = {c.cell_id: c for c in default_grid()}
        if args.cell not in cells:
            raise SystemExit(f"unknown cell {args.cell!r}; known: {sorted(cells)[:5]} ...")
        row = run_sim_rep(cells[args.cell], args.rep, args.method, tuner=tuner, device=args.device)
    print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
