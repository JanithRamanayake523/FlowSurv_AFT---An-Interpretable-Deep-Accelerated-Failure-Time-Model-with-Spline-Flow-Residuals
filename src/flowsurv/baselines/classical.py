"""Classical survival baselines (Implementation Plan Phase 3, task 1).

Implements the pre-registered classical/ML methods behind the common
:class:`~flowsurv.baselines.common.SurvivalMethod` interface:

- Cox PH and Weibull/log-normal AFT (lifelines)
- Random Survival Forest (scikit-survival, optional)

All methods return float32 CPU tensors at the interface boundary and implement
``predict_density``/``predict_hazard`` by numerical differentiation of the
predicted survival curve, so the shared runner never needs special-casing.
"""

from __future__ import annotations

import time
import warnings
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from .common import FitResult, SurvivalMethod, as_output, density_from_survival, hazard_from_survival, interp_surv, to_numpy


class _LifelinesMethod(SurvivalMethod):
    """Shared harness for lifelines regression models."""

    is_deep: bool = False
    supports_density: bool = True
    supports_hazard: bool = True
    _estimator_cls: type | None = None
    _duration_col: str = "duration"
    _event_col: str = "event"

    def __init__(self) -> None:
        self.model: Any | None = None
        self.tuning_space: dict[str, list] = {}

    def _make_df(self, t: np.ndarray, d: np.ndarray, x: np.ndarray) -> pd.DataFrame:
        df = pd.DataFrame(x)
        df.columns = [f"x{i}" for i in range(x.shape[1])]
        df[self._duration_col] = t
        df[self._event_col] = d.astype(int)
        return df

    def _fit_estimator(self, df: pd.DataFrame, **hyper) -> Any:
        raise NotImplementedError

    def _predict_surv_frame(
        self,
        df: pd.DataFrame,
        times: np.ndarray,
        x: np.ndarray,
    ) -> pd.DataFrame:
        raise NotImplementedError

    def _risk(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def fit(
        self,
        t: Tensor,
        d: Tensor,
        x: Tensor,
        *,
        val: tuple[Tensor, Tensor, Tensor] | None = None,
        seed: int = 0,
        **hyper,
    ) -> FitResult:
        try:
            import lifelines
        except ImportError as e:  # pragma: no cover
            return FitResult(wall_time_s=0.0, converged=False, info={"error": f"lifelines not installed: {e}"})

        t_np, d_np, x_np = to_numpy(t, d, x)
        df = self._make_df(t_np, d_np, x_np)
        start = time.perf_counter()
        try:
            self.model = self._fit_estimator(df, **hyper)
            converged = True
            info: dict = {}
        except Exception as e:
            converged = False
            info = {"error": repr(e)}
            warnings.warn(f"{self.name} fit failed: {e!r}")
        return FitResult(wall_time_s=time.perf_counter() - start, converged=converged, info=info)

    def predict_surv(self, t: Tensor, x: Tensor) -> Tensor:
        if self.model is None or getattr(self.model, "params_", None) is None:
            raise RuntimeError(f"{self.name} has not been fit successfully")
        t_np, _, x_np = to_numpy(t, torch.zeros_like(t), x)
        times = np.sort(np.unique(t_np))
        df = self._make_df(np.zeros(x_np.shape[0]), np.ones(x_np.shape[0]), x_np)
        surv = self._predict_surv_frame(df, times, x_np)
        surv_arr = interp_surv(surv.index.to_numpy(), surv.to_numpy(), t_np)
        return as_output(surv_arr)

    def predict_density(self, t: Tensor, x: Tensor) -> Tensor:
        surv = self.predict_surv(t, x)
        t_np, _, _ = to_numpy(t, torch.zeros_like(t), x)
        dens = density_from_survival(t_np, surv.numpy())
        return as_output(dens)

    def predict_hazard(self, t: Tensor, x: Tensor) -> Tensor:
        dens = self.predict_density(t, x)
        surv = self.predict_surv(t, x)
        h = dens.numpy() / np.clip(surv.numpy(), np.finfo(np.float64).tiny, 1.0)
        return as_output(np.clip(h, 0.0, 1e3))

    def predict_risk(self, x: Tensor) -> Tensor:
        _, _, x_np = to_numpy(torch.zeros(x.shape[0]), torch.zeros(x.shape[0]), x)
        return as_output(self._risk(x_np))


class CoxPH(_LifelinesMethod):
    """Cox proportional hazards (lifelines.CoxPHFitter)."""

    name: str = "cox_ph"

    def __init__(self) -> None:
        super().__init__()
        self.tuning_space = {"penalizer": [0.0, 0.01, 0.1]}

    def _fit_estimator(self, df: pd.DataFrame, **hyper) -> Any:
        from lifelines import CoxPHFitter

        penalizer = hyper.get("penalizer", 0.0)
        model = CoxPHFitter(penalizer=penalizer)
        model.fit(df, duration_col=self._duration_col, event_col=self._event_col)
        return model

    def _predict_surv_frame(
        self,
        df: pd.DataFrame,
        times: np.ndarray,
        x: np.ndarray,
    ) -> pd.DataFrame:
        return self.model.predict_survival_function(df, times=times)

    def _risk(self, x: np.ndarray) -> np.ndarray:
        df = self._make_df(np.zeros(x.shape[0]), np.ones(x.shape[0]), x)
        return self.model.predict_partial_hazard(df).to_numpy().ravel()


class WeibullAFT(_LifelinesMethod):
    """Parametric Weibull AFT (lifelines.WeibullAFTFitter)."""

    name: str = "weibull_aft"

    def __init__(self) -> None:
        super().__init__()
        self.tuning_space = {"penalizer": [0.0, 0.01, 0.1]}

    def _fit_estimator(self, df: pd.DataFrame, **hyper) -> Any:
        from lifelines import WeibullAFTFitter

        model = WeibullAFTFitter(penalizer=hyper.get("penalizer", 0.0))
        model.fit(df, duration_col=self._duration_col, event_col=self._event_col)
        return model

    def _predict_surv_frame(
        self,
        df: pd.DataFrame,
        times: np.ndarray,
        x: np.ndarray,
    ) -> pd.DataFrame:
        return self.model.predict_survival_function(df, times=times)

    def _risk(self, x: np.ndarray) -> np.ndarray:
        df = self._make_df(np.zeros(x.shape[0]), np.ones(x.shape[0]), x)
        med = self.model.predict_percentile(df, p=0.5).to_numpy().ravel()
        return -np.asarray(med, dtype=np.float64)


class LogNormalAFT(_LifelinesMethod):
    """Parametric log-normal AFT (lifelines.LogNormalAFTFitter)."""

    name: str = "log_normal_aft"

    def __init__(self) -> None:
        super().__init__()
        self.tuning_space = {"penalizer": [0.0, 0.01, 0.1]}

    def _fit_estimator(self, df: pd.DataFrame, **hyper) -> Any:
        from lifelines import LogNormalAFTFitter

        model = LogNormalAFTFitter(penalizer=hyper.get("penalizer", 0.0))
        model.fit(df, duration_col=self._duration_col, event_col=self._event_col)
        return model

    def _predict_surv_frame(
        self,
        df: pd.DataFrame,
        times: np.ndarray,
        x: np.ndarray,
    ) -> pd.DataFrame:
        return self.model.predict_survival_function(df, times=times)

    def _risk(self, x: np.ndarray) -> np.ndarray:
        df = self._make_df(np.zeros(x.shape[0]), np.ones(x.shape[0]), x)
        med = self.model.predict_percentile(df, p=0.5).to_numpy().ravel()
        return -np.asarray(med, dtype=np.float64)


class RandomSurvivalForest(SurvivalMethod):
    """Random Survival Forest (scikit-survival; optional dependency)."""

    name: str = "rsf"
    is_deep: bool = False
    supports_density: bool = True
    supports_hazard: bool = True

    def __init__(self) -> None:
        self.model: Any | None = None
        self.tuning_space = {
            "n_estimators": [100, 500],
            "min_samples_split": [10, 20],
            "min_samples_leaf": [5, 10],
        }

    def fit(
        self,
        t: Tensor,
        d: Tensor,
        x: Tensor,
        *,
        val: tuple[Tensor, Tensor, Tensor] | None = None,
        seed: int = 0,
        **hyper,
    ) -> FitResult:
        try:
            from sksurv.ensemble import RandomSurvivalForest
        except ImportError as e:  # pragma: no cover
            return FitResult(
                wall_time_s=0.0,
                converged=False,
                info={"error": f"scikit-survival not installed: {e}"},
            )

        t_np, d_np, x_np = to_numpy(t, d, x)
        # sksurv uses structured array: (event, time)
        y = np.array(
            [(bool(e), float(tt)) for e, tt in zip(d_np, t_np)],
            dtype=[("event", bool), ("time", float)],
        )
        start = time.perf_counter()
        try:
            self.model = RandomSurvivalForest(
                n_estimators=hyper.get("n_estimators", 500),
                min_samples_split=hyper.get("min_samples_split", 10),
                min_samples_leaf=hyper.get("min_samples_leaf", 5),
                n_jobs=-1,
                random_state=seed,
            )
            self.model.fit(x_np, y)
            converged = True
            info: dict = {}
        except Exception as e:
            converged = False
            info = {"error": repr(e)}
            warnings.warn(f"RSF fit failed: {e!r}")
        return FitResult(wall_time_s=time.perf_counter() - start, converged=converged, info=info)

    def _surv_grid(self, t: Tensor, x: Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.model is None:
            raise RuntimeError("RSF has not been fit successfully")
        t_np, _, x_np = to_numpy(t, torch.zeros_like(t), x)
        times = np.sort(np.unique(t_np))
        funcs = list(self.model.predict_survival_function(x_np))
        # funcs is iterable of 1-D step functions; evaluate at times and orient
        # as (n_times, n_subjects). np.atleast_1d keeps the time axis when m==1.
        # Each StepFunction's domain is fixed by the fitted (training) event
        # times, so query times past it (e.g. the eval grid's tau, or a test
        # subject's observed time, when they exceed the largest training
        # time) raise ValueError; clip to the domain -- survival is flat
        # beyond the last observed event, which is what clipping gives.
        lo, hi = funcs[0].domain
        times_eval = np.clip(times, lo, hi)
        surv = np.stack([np.atleast_1d(fn(times_eval)) for fn in funcs], axis=0).T
        return times, surv, t_np

    def predict_surv(self, t: Tensor, x: Tensor) -> Tensor:
        times, surv, t_query = self._surv_grid(t, x)
        surv_arr = interp_surv(times, surv, t_query)
        return as_output(surv_arr)

    def predict_density(self, t: Tensor, x: Tensor) -> Tensor:
        surv = self.predict_surv(t, x)
        t_np, _, _ = to_numpy(t, torch.zeros_like(t), x)
        dens = density_from_survival(t_np, surv.numpy())
        return as_output(dens)

    def predict_hazard(self, t: Tensor, x: Tensor) -> Tensor:
        dens = self.predict_density(t, x)
        surv = self.predict_surv(t, x)
        h = dens.numpy() / np.clip(surv.numpy(), np.finfo(np.float64).tiny, 1.0)
        return as_output(np.clip(h, 0.0, 1e3))

    def predict_risk(self, x: Tensor) -> Tensor:
        if self.model is None:
            raise RuntimeError("RSF has not been fit successfully")
        _, _, x_np = to_numpy(torch.zeros(x.shape[0]), torch.zeros(x.shape[0]), x)
        try:
            risk = self.model.predict(x_np)
        except Exception:
            # Fallback: cumulative hazard at last observed time
            times = self.model.unique_times_
            funcs = self.model.predict_cumulative_hazard_function(x_np)
            risk = np.array([fn(times[-1]) for fn in funcs])
        return as_output(risk)
