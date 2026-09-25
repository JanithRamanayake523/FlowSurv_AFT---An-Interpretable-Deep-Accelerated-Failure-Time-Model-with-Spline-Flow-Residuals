"""30-config random search + freeze-then-audit tuning (Methodology Sec. 2.4, prereg Sec. 4).

Protocol (pre-registered, do not deviate):

- Budget: 30 random hyperparameter configurations per DL method per
  macro-cell, identical for every DL method (fairness constraint).
- Selection objective: validation NLL. FlowSurv methods use the exact
  right-censored NLL on the validation fold (Methodology Sec. 2.3); other DL
  methods use the package-equivalent objective (``FitResult.info["val_loss"]``);
  Royston-Parmar uses its df-selection path (validation NLL over df = 3-5).
- Freeze-then-audit: within each simulation macro-cell, hyperparameters are
  tuned on replications 1-5 and frozen for replications 6-100. Full nested
  (per-rep) tuning on a random 10% of cells serves as the audit.
- Real data (Methodology Sec. 4.2): every train fold gets the same 30-config
  budget with a 15% inner validation split -- no freezing across splits.

Chosen configs, tuning logs, and audit results are persisted as parquet under
``experiments/tuning/`` so interrupted grid runs resume with frozen configs.
"""

from __future__ import annotations

import itertools
import json
import math
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..baselines import METHODS
from ..baselines.flowsurv_wrappers import mode_kwargs
from ..data import (
    REAL_DATASETS,
    CellConfig,
    cell_seed,
    generate_dataset,
    real_split,
    split_train_val_test,
)

TUNING_DIR = Path("experiments/tuning")
N_CONFIGS = 30
TUNE_REPS = (1, 2, 3, 4, 5)  # freeze-then-audit: tune on reps 1-5, freeze 6-100
AUDIT_FRAC = 0.1


# ---------------------------------------------------------------------------
# config sampling
# ---------------------------------------------------------------------------


def sample_configs(tuning_space: dict[str, list], n: int = N_CONFIGS, seed: int = 0) -> list[dict]:
    """Uniform random sample of ``n`` configs without replacement.

    If the cartesian product of ``tuning_space`` has fewer than ``n`` points,
    the full product is returned (order preserved). Deterministic given
    ``seed``.
    """
    if not tuning_space:
        return []
    keys = list(tuning_space)
    cartesian = [
        dict(zip(keys, combo))
        for combo in itertools.product(*(list(tuning_space[k]) for k in keys))
    ]
    if len(cartesian) <= n:
        return cartesian
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(cartesian), size=n, replace=False)
    return [cartesian[int(i)] for i in idx]


# ---------------------------------------------------------------------------
# validation objective
# ---------------------------------------------------------------------------


def _per_subject(pred) -> torch.Tensor:
    """Reduce a prediction to one value per subject.

    ``predict_density``/``predict_surv`` map (t (m,), x (n,p)) -> (m,n); when
    called with the observed times of the same n subjects (m == n) the
    per-subject values are the diagonal.
    """
    pred = torch.as_tensor(pred, dtype=torch.float32)
    if pred.dim() == 2:
        if pred.shape[0] != pred.shape[1]:
            raise ValueError(f"cannot reduce non-square prediction of shape {tuple(pred.shape)}")
        pred = pred.diagonal()
    return pred


def censored_val_nll(method, val: dict) -> float:
    """Exact right-censored validation NLL from density/survival predictions."""
    log_f = _per_subject(method.predict_density(val["t"], val["x"], **mode_kwargs(method, True))).clamp_min(1e-12).log()
    log_s = _per_subject(method.predict_surv(val["t"], val["x"], **mode_kwargs(method, True))).clamp_min(1e-12).log()
    d = torch.as_tensor(val["d"], dtype=torch.float32)
    ll = d * log_f + (1.0 - d) * log_s
    return float(-ll.mean())


def config_score(method_name: str, method, fit_result, val: dict) -> float:
    """Validation score of one (method, config) fit; lower is better.

    - FlowSurv methods: exact right-censored NLL on the validation fold.
    - Royston-Parmar: df-selection path (validation NLL over df configs).
    - Other DL methods: package-equivalent objective, ``info["val_loss"]``.
    - Fallback for any density-capable method: exact validation NLL.
    """
    if method_name.startswith("flowsurv"):
        return censored_val_nll(method, val)
    info = getattr(fit_result, "info", None) or {}
    if method_name == "royston_parmar" and getattr(method, "supports_density", False):
        return censored_val_nll(method, val)
    if "val_loss" in info:
        return float(info["val_loss"])
    if getattr(method, "supports_density", False):
        return censored_val_nll(method, val)
    warnings.warn(f"no validation objective available for {method_name!r}; scoring NaN")
    return math.nan


# ---------------------------------------------------------------------------
# random search
# ---------------------------------------------------------------------------

TUNING_LOG_COLUMNS = ["config_id", "config_json", "mean_val_score", "converged_all"]


def tune_method(
    method_name: str,
    datasets: list[dict],
    seed: int = 0,
    n_configs: int = N_CONFIGS,
    verbose: bool = False,
    cell_id: str = "",
) -> tuple[dict, pd.DataFrame]:
    """30-config random search for ``method_name`` over ``datasets``.

    ``datasets`` is a list of ``{"train": {...}, "val": {...}}`` splits (for a
    simulation macro-cell: the train/val splits of replications 1-5). Each
    config is fit fresh on every dataset's train fold with its val fold; the
    selection score is the mean validation NLL (or package equivalent).

    Returns ``(best_config, tuning_log)``. Classical methods without a tuning
    space return ``({}, empty_log)`` immediately (package defaults).

    ``verbose``: print a ``[tune config_id/n_configs]`` line per config (this
    search runs once per macro-cell before that cell's first scored fit, and
    is otherwise silent -- a multi-minute gap that looks like a hang from the
    grid driver's per-fit progress line alone).
    """
    cls = METHODS[method_name]
    probe = cls()
    space = getattr(probe, "tuning_space", None) or {}
    empty_log = pd.DataFrame(columns=TUNING_LOG_COLUMNS)
    if not space:
        return {}, empty_log

    configs = sample_configs(space, n=n_configs, seed=seed)
    if verbose:
        print(f"[tune] {method_name} {cell_id}: {len(configs)}-config search over {len(datasets)} datasets", flush=True)
    rows = []
    for config_id, cfg in enumerate(configs):
        t0 = time.perf_counter()
        scores, converged_all = [], True
        for rep_offset, ds in enumerate(datasets):
            method = cls()
            # A fold whose fit fails (e.g. a validation fold with no events under heavy
            # censoring) must not abort the whole macro-cell: it is scored NaN, ignored in the
            # mean over folds, and flagged through ``converged_all``.
            try:
                fit_result = method.fit(
                    ds["train"]["t"],
                    ds["train"]["d"],
                    ds["train"]["x"],
                    val=(ds["val"]["t"], ds["val"]["d"], ds["val"]["x"]),
                    seed=seed + rep_offset,
                    **cfg,
                )
            except Exception as e:  # noqa: BLE001 -- failures are data
                if verbose:
                    print(f"[tune] {method_name} {cell_id} config {config_id + 1} fold {rep_offset}: fit raised {e!r}", flush=True)
                converged_all = False
                scores.append(math.nan)
                continue
            converged_all = converged_all and bool(fit_result.converged)
            if not fit_result.converged:
                scores.append(math.nan)
                continue
            scores.append(config_score(method_name, method, fit_result, ds["val"]))
        row = {
            "config_id": config_id,
            "config_json": json.dumps(cfg, sort_keys=True),
            "mean_val_score": float(np.nanmean(scores)) if scores else math.nan,
            "converged_all": converged_all,
        }
        row.update({f"cfg_{k}": v for k, v in cfg.items()})
        rows.append(row)
        if verbose:
            print(
                f"[tune] {method_name} {cell_id} config {config_id + 1}/{len(configs)} "
                f"score={row['mean_val_score']:.4f} conv={converged_all!s:5s} "
                f"({time.perf_counter() - t0:.1f}s)",
                flush=True,
            )

    log = pd.DataFrame(rows, columns=TUNING_LOG_COLUMNS + sorted({k for r in rows for k in r if k.startswith("cfg_")}))
    if log["mean_val_score"].isna().all():
        warnings.warn(f"all validation scores NaN for {method_name!r}; keeping first config")
        best = configs[0]
    else:
        best = json.loads(log.loc[log["mean_val_score"].idxmin(), "config_json"])
    return best, log


# ---------------------------------------------------------------------------
# inner validation split (real data, prereg Sec. 4: 15% of the train fold)
# ---------------------------------------------------------------------------


def _take(v, idx: np.ndarray):
    if torch.is_tensor(v):
        return v[torch.from_numpy(np.asarray(idx)).long()]
    return np.asarray(v)[idx]


def inner_val_split(train: dict, val_frac: float = 0.15, seed: int = 0) -> tuple[dict, dict]:
    """Seeded random inner validation split of a training fold."""
    n = int(len(train["t"]))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_val = max(1, int(round(val_frac * n)))
    idx_val, idx_tr = perm[:n_val], perm[n_val:]
    train2 = {k: _take(train[k], idx_tr) for k in ("t", "d", "x")}
    val = {k: _take(train[k], idx_val) for k in ("t", "d", "x")}
    return train2, val


# ---------------------------------------------------------------------------
# freeze-then-audit tuner
# ---------------------------------------------------------------------------


@dataclass
class RealCell:
    """Duck-typed stand-in for ``CellConfig`` so ``FrozenTuner`` can key and
    tune real datasets (which have no simulation macro-cell). Real data is
    always tuned per split (Methodology Sec. 4.2)."""

    cell_id: str
    name: str
    is_real: bool = True


def audit_cells(cells, frac: float = AUDIT_FRAC, seed: int = 0) -> list[str]:
    """Deterministic selection of the audit subset of cells (prereg Sec. 4).

    Accepts ``CellConfig`` objects or plain cell-id strings; returns sorted
    cell ids.
    """
    ids = sorted(str(getattr(c, "cell_id", c)) for c in cells)
    if not ids:
        return []
    rng = np.random.default_rng(seed)
    k = max(1, int(round(frac * len(ids))))
    chosen = rng.permutation(ids)[:k]
    return sorted(str(c) for c in chosen)


class FrozenTuner:
    """Pre-registered freeze-then-audit tuning protocol (prereg Sec. 4).

    - Normal mode (``audit=False``): on first request for a (method,
      macro-cell) pair, tune on replications 1-5, persist the chosen config to
      ``experiments/tuning/``, and return the frozen config for all subsequent
      replications (6-100). Interrupted runs reload the frozen config from
      disk instead of re-tuning.
    - Audit mode (``audit=True``): re-tune per replication (full nested
      tuning) -- used on the 10% audit subset of cells from :func:`audit_cells`;
      per-rep configs are appended to ``*_audit.parquet``.
    - Real cells (``RealCell``): always tuned per split.

    Classical non-deep methods with no tuning space get ``{}`` immediately.
    """

    def __init__(
        self,
        tuning_dir: str | Path = TUNING_DIR,
        n_configs: int = N_CONFIGS,
        seed: int = 0,
        audit: bool = False,
        verbose: bool = False,
    ) -> None:
        self.tuning_dir = Path(tuning_dir)
        self.n_configs = n_configs
        self.seed = seed
        self.audit = audit
        self.verbose = verbose
        self._cache: dict[tuple[str, str, int], dict] = {}

    # -- paths -------------------------------------------------------------

    def _path(self, method_name: str, cell_id: str, kind: str) -> Path:
        return self.tuning_dir / f"{method_name}__{cell_id}__{kind}.parquet"

    # -- dataset construction ----------------------------------------------

    def _build_datasets(self, cell, reps: list[int]) -> list[dict]:
        if getattr(cell, "is_real", False):
            data = REAL_DATASETS[cell.name]()
            datasets = []
            for r in reps:
                splits = real_split(data["t"], data["d"], data["x"], r)
                train2, val = inner_val_split(splits["train"], val_frac=0.15, seed=r)
                datasets.append({"train": train2, "val": val})
            return datasets
        datasets = []
        for r in reps:
            data = generate_dataset(cell, r)
            splits = split_train_val_test(
                data["t"], data["d"], data["x"], seed=cell_seed(cell.cell_id, r)
            )
            datasets.append({"train": splits["train"], "val": splits["val"]})
        return datasets

    # -- persistence ---------------------------------------------------------

    def _persist_frozen(self, method_name: str, cell_id: str, best: dict, log: pd.DataFrame) -> None:
        self.tuning_dir.mkdir(parents=True, exist_ok=True)
        chosen = pd.DataFrame(
            [
                {
                    "method": method_name,
                    "cell_id": cell_id,
                    "config_json": json.dumps(best, sort_keys=True),
                    "tune_reps": ",".join(str(r) for r in TUNE_REPS),
                    "n_configs": self.n_configs,
                    "seed": self.seed,
                }
            ]
        )
        chosen.to_parquet(self._path(method_name, cell_id, "chosen"), index=False)
        log.to_parquet(self._path(method_name, cell_id, "log"), index=False)

    def _persist_audit(self, method_name: str, cell_id: str, rep: int, best: dict, log: pd.DataFrame) -> None:
        self.tuning_dir.mkdir(parents=True, exist_ok=True)
        row = pd.DataFrame(
            [
                {
                    "method": method_name,
                    "cell_id": cell_id,
                    "rep": int(rep),
                    "config_json": json.dumps(best, sort_keys=True),
                    "mean_val_score": float(log["mean_val_score"].min()) if len(log) else math.nan,
                }
            ]
        )
        path = self._path(method_name, cell_id, "audit")
        if path.exists():
            old = pd.read_parquet(path)
            old = old[old["rep"] != int(rep)]  # idempotent re-tune of the same rep
            row = pd.concat([old, row], ignore_index=True)
        row.to_parquet(path, index=False)
        log.assign(rep=int(rep)).to_parquet(
            self._path(method_name, cell_id, f"audit_log_rep{int(rep)}"), index=False
        )

    def _load_or_tune_frozen(self, method_name: str, cell) -> dict:
        path = self._path(method_name, cell.cell_id, "chosen")
        if path.exists():
            row = pd.read_parquet(path).iloc[0]
            return json.loads(row["config_json"])
        datasets = self._build_datasets(cell, list(TUNE_REPS))
        best, log = tune_method(
            method_name, datasets, seed=self.seed, n_configs=self.n_configs,
            verbose=self.verbose, cell_id=str(cell.cell_id),
        )
        self._persist_frozen(method_name, cell.cell_id, best, log)
        return best

    # -- public API ----------------------------------------------------------

    def tuned_config(self, method_name: str, cell, rep: int = 1) -> dict:
        """Frozen (or per-rep audit) hyperparameter config for a fit.

        ``cell`` is a ``CellConfig`` (simulation) or ``RealCell`` (real data).
        """
        probe = METHODS[method_name]()
        space = getattr(probe, "tuning_space", None) or {}
        if not space:  # classical / package defaults: nothing to tune
            return {}

        is_real = bool(getattr(cell, "is_real", False))
        if self.audit or is_real:
            key = (method_name, str(cell.cell_id), int(rep))
            if key not in self._cache:
                datasets = self._build_datasets(cell, [int(rep)])
                best, log = tune_method(
                    method_name, datasets, seed=self.seed + int(rep), n_configs=self.n_configs,
                    verbose=self.verbose, cell_id=str(cell.cell_id),
                )
                self._persist_audit(method_name, str(cell.cell_id), rep, best, log)
                self._cache[key] = best
            return dict(self._cache[key])

        key = (method_name, str(cell.cell_id), 0)
        if key not in self._cache:
            self._cache[key] = self._load_or_tune_frozen(method_name, cell)
        return dict(self._cache[key])
