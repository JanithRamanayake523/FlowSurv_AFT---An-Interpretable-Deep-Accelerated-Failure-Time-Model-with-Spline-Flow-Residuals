"""Stub implementations for optional baselines that rely on external dependencies.

These keep the :data:`~flowsurv.baselines.METHODS` registry stable when the
optional Phase-3 packages (R/flexsurv for Royston-Parmar, torchdiffeq for
Ausset-CNF) are not installed.  A fit call returns ``converged=False`` so the
shared runner records the method as missing rather than crashing the grid.
"""

from __future__ import annotations

from .common import FitResult, SurvivalMethod


class _UnavailableMethod(SurvivalMethod):
    """Placeholder baseline whose ``fit`` always reports non-convergence."""

    is_deep: bool = False
    supports_density: bool = True
    supports_hazard: bool = True
    name: str = "unavailable"
    tuning_space: dict[str, list] = {}

    def __init__(self) -> None:
        self.reason: str = "optional dependency not installed"

    def fit(self, t, d, x, *, val=None, seed=0, **hyper) -> FitResult:
        return FitResult(
            wall_time_s=0.0,
            converged=False,
            info={"error": f"{self.name} unavailable: {self.reason}"},
        )

    def predict_surv(self, t, x) -> None:
        raise RuntimeError(f"{self.name} was not fit (unavailable)")

    def predict_density(self, t, x) -> None:
        raise RuntimeError(f"{self.name} was not fit (unavailable)")

    def predict_hazard(self, t, x) -> None:
        raise RuntimeError(f"{self.name} was not fit (unavailable)")

    def predict_risk(self, x) -> None:
        raise RuntimeError(f"{self.name} was not fit (unavailable)")
