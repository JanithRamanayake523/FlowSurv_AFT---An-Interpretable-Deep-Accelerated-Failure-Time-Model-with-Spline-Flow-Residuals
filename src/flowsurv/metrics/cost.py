"""Wall-clock cost audit (Methodology Sec. 3.3, pre-registration metric 6).

Quantifies the bidirectional-exactness claim: wall-clock for training,
metric-relevant evaluation (survival + density predictions), and 1,000-sample
generation per subject -- one analytic pass for FlowSurv-AFT, numerical
inversion/solver loops for the flow competitors.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import torch
from torch import Tensor


def time_call(fn, *args, repeats: int = 1, warmup: int = 0, **kwargs) -> float:
    """Best-of-``repeats`` wall-clock seconds of fn(*args, **kwargs).

    ``warmup`` un-timed calls run first (cache/JIT warm-up). Best-of rather
    than mean: the audit measures achievable cost, not scheduler noise.
    """
    for _ in range(warmup):
        fn(*args, **kwargs)
    best = math.inf
    for _ in range(max(int(repeats), 1)):
        start = time.perf_counter()
        fn(*args, **kwargs)
        best = min(best, time.perf_counter() - start)
    return float(best)


@dataclass
class CostReport:
    """Wall-clock seconds for fit / evaluation / n-sample generation."""

    fit_s: float
    eval_s: float
    sample_s: float


def audit_cost(
    method,
    t_train: Tensor,
    d_train: Tensor,
    x_train: Tensor,
    t_grid: Tensor,
    x_eval: Tensor,
    n_samples: int = 1000,
    seed: int = 0,
) -> CostReport:
    """Time fit, evaluation, and sampling of a SurvivalMethod-interface object.

    Expected interface (shared with the baseline harness, Phase 3):
    ``fit(t, d, x)``, ``predict_surv(t_grid, x) -> (m, n)``; optional
    ``predict_density(t_grid, x)`` and ``sample(x, n) -> (n_samples, n)``.

    Methods without a callable ``sample`` get ``sample_s = NaN`` -- the
    bidirectional-exactness cost claim of Methodology Sec. 3.3 is then simply
    not applicable to them (recorded as missing, not zero).
    """
    fit_s = time_call(method.fit, t_train, d_train, x_train)

    def _evaluate() -> None:
        method.predict_surv(t_grid, x_eval)
        if callable(getattr(method, "predict_density", None)):
            method.predict_density(t_grid, x_eval)

    eval_s = time_call(_evaluate)

    if callable(getattr(method, "sample", None)):

        def _sample() -> None:
            torch.manual_seed(seed)  # reproducible sampling-cost measurement
            method.sample(x_eval, n_samples)

        sample_s = time_call(_sample)
    else:
        sample_s = float("nan")  # no sampling support: cost claim not applicable
    return CostReport(fit_s=fit_s, eval_s=eval_s, sample_s=sample_s)
