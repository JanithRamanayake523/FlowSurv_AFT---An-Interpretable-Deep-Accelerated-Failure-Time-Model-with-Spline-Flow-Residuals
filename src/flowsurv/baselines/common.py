"""Common baseline interface and shared helpers (Implementation Plan Phase 3, task 2).

Every method in :mod:`flowsurv.baselines` implements :class:`SurvivalMethod`
and returns :class:`FitResult` from ``fit``. Tensors are float32 CPU at the
interface boundary; methods that use GPUs or external libraries (lifelines,
scikit-survival, pycox, R/flexsurv) convert internally.

Prediction orientation contract (used by eval/metrics):

    predict_surv / predict_density / predict_hazard:
        (m,) evaluation times x (n, p) covariates -> (m, n) array
    predict_risk: (n, p) -> (n,), higher = riskier.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np
import torch
from torch import Tensor


@dataclass
class FitResult:
    """Outcome of a ``fit`` call.

    ``converged=False`` records a failed fit (e.g. lifelines did not
    converge); per the pre-registration, failure rates are a reported
    finding, so ``fit`` must return this instead of raising whenever the
    failure is in the underlying estimator.
    """

    wall_time_s: float
    converged: bool = True
    info: dict = field(default_factory=dict)


@runtime_checkable
class SurvivalMethod(Protocol):
    """Common method interface (Implementation Plan Phase 3, task 2).

    ``is_deep=True`` marks the methods that receive the identical 30-config
    tuning budget of Methodology Sec. 2.4; ``tuning_space`` holds the
    candidate lists the random search samples uniformly from.
    """

    name: str
    is_deep: bool
    supports_density: bool
    supports_hazard: bool
    tuning_space: dict[str, list]

    def fit(
        self,
        t: Tensor,
        d: Tensor,
        x: Tensor,
        *,
        val: tuple[Tensor, Tensor, Tensor] | None = None,
        seed: int = 0,
        **hyper,
    ) -> FitResult: ...

    def predict_surv(self, t: Tensor, x: Tensor) -> Tensor: ...

    def predict_density(self, t: Tensor, x: Tensor) -> Tensor: ...

    def predict_hazard(self, t: Tensor, x: Tensor) -> Tensor: ...

    def predict_risk(self, x: Tensor) -> Tensor: ...


# ----------------------------------------------------------------------
# helpers


def to_numpy(t: Tensor, d: Tensor, x: Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interface tensors -> float64 numpy ``(t (n,), d (n,), x (n, p))``."""

    def _a(v) -> np.ndarray:
        if torch.is_tensor(v):
            v = v.detach().cpu().numpy()
        return np.asarray(v, dtype=np.float64)

    return _a(t).ravel(), _a(d).ravel(), _a(x)


def interp_surv(
    times_src: np.ndarray,
    surv_src: np.ndarray,
    t_query: np.ndarray,
) -> np.ndarray:
    """Interpolate per-subject survival curves onto a common query grid.

    Args:
        times_src: (m_src,) ascending source time grid, shared by all subjects
            (as returned by lifelines/pycox predictions).
        surv_src: (m_src, n) survival values on ``times_src``.
        t_query: (m,) ascending query times.

    Returns:
        (m, n) float64 array: linear interpolation, constant extrapolation at
        both ends, clamped to [0, 1], with monotonicity enforced by a
        cumulative minimum along the time axis (interpolation can introduce
        tiny upward wiggles in near-flat tails).
    """
    ts = np.asarray(times_src, dtype=np.float64).ravel()
    s = np.asarray(surv_src, dtype=np.float64)
    tq = np.asarray(t_query, dtype=np.float64).ravel()
    if ts.ndim != 1:
        raise ValueError("times_src must be a 1-D grid")
    if ts.size == 1:
        # degenerate grid: constant survival for every query time
        return np.clip(np.broadcast_to(s[0][None, :], (tq.size, s.shape[1])), 0.0, 1.0)
    if ts.size < 1:
        raise ValueError("times_src must be non-empty")
    if s.shape[0] != ts.size:
        raise ValueError(f"surv_src has {s.shape[0]} rows but times_src has {ts.size}")
    if np.any(np.diff(ts) <= 0):
        raise ValueError("times_src must be strictly increasing")

    lo = np.clip(np.searchsorted(ts, tq, side="right") - 1, 0, ts.size - 2)
    hi = lo + 1
    span = np.maximum(ts[hi] - ts[lo], np.finfo(np.float64).tiny)
    w = ((tq - ts[lo]) / span)[:, None]
    out = s[lo] * (1.0 - w) + s[hi] * w
    # constant extrapolation outside the source grid
    out[tq <= ts[0]] = s[0]
    out[tq >= ts[-1]] = s[-1]
    out = np.minimum.accumulate(out, axis=0)
    return np.clip(out, 0.0, 1.0)


def hazard_from_survival(t_grid: np.ndarray, surv: np.ndarray) -> np.ndarray:
    """Numerical hazard h = -d log S / dt on a grid (m,) x (m, n) -> (m, n).

    Used for smooth baselines whose survival is available but whose hazard is
    not (Implementation Plan Phase 4, task 4: "differentiation for smooth
    baselines"). First-order edges; noise in ``surv`` is amplified by the
    differentiation, so callers should pass a sufficiently fine grid and/or
    smooth S first -- this is documented per method wherever used.
    """
    tg = np.asarray(t_grid, dtype=np.float64).ravel()
    s = np.asarray(surv, dtype=np.float64)
    if s.shape[0] != tg.size:
        raise ValueError("surv must have one row per grid time")
    if tg.size <= 1:
        return np.zeros_like(s)
    log_s = np.log(np.clip(s, np.finfo(np.float64).tiny, 1.0))
    dlog = np.gradient(log_s, tg, axis=0, edge_order=1)
    return np.clip(-dlog, 0.0, None)


def density_from_survival(t_grid: np.ndarray, surv: np.ndarray) -> np.ndarray:
    """Numerical density f = -dS/dt from a survival grid -> (m, n).

    Computed directly by finite-differencing ``surv`` so that division by
    near-zero survival values does not explode. The result is clipped to be
    non-negative.
    """
    tg = np.asarray(t_grid, dtype=np.float64).ravel()
    s = np.asarray(surv, dtype=np.float64)
    if s.shape[0] != tg.size:
        raise ValueError("surv must have one row per grid time")
    if tg.size <= 1:
        return np.zeros_like(s)
    ds = np.gradient(s, tg, axis=0, edge_order=1)
    return np.clip(-ds, 0.0, None)


def as_output(arr: np.ndarray) -> Tensor:
    """Prediction helper output -> float32 CPU tensor at the interface boundary."""
    return torch.from_numpy(np.ascontiguousarray(arr)).float()
