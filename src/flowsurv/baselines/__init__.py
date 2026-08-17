"""Baselines: classical (Cox/AFT/RSF/Royston-Parmar) and deep (DeepSurv/DeepHit/DSM) (Phase 3)."""

from __future__ import annotations

from .classical import CoxPH, LogNormalAFT, RandomSurvivalForest, WeibullAFT
from .deep import DeepHit, DeepSurv
from .dsm import DSM
from .flowsurv_wrappers import FlowSurvAFTMethod, FlowSurvGaussMethod

#: Royston-Parmar and Ausset-CNF are optional (external R/ODE dependencies).
#: Import them where available; otherwise provide a no-op placeholder so the
#: registry remains stable.
try:
    from .royston_parmar import RoystonParmar
except Exception:  # pragma: no cover
    from .common import SurvivalMethod
    from .stubs import _UnavailableMethod

    class RoystonParmar(_UnavailableMethod):  # type: ignore[no-redef]
        name: str = "royston_parmar"


try:
    from .ausset_cnf import AussetCNF
except Exception:  # pragma: no cover
    from .stubs import _UnavailableMethod

    class AussetCNF(_UnavailableMethod):  # type: ignore[no-redef]
        name: str = "ausset_cnf"


METHODS: dict[str, type[SurvivalMethod]] = {
    "flowsurv_aft": FlowSurvAFTMethod,
    "flowsurv_gauss": FlowSurvGaussMethod,
    "cox_ph": CoxPH,
    "weibull_aft": WeibullAFT,
    "log_normal_aft": LogNormalAFT,
    "rsf": RandomSurvivalForest,
    "deepsurv": DeepSurv,
    "deephit": DeepHit,
    "dsm": DSM,
    "royston_parmar": RoystonParmar,
    "ausset_cnf": AussetCNF,
}

__all__ = [
    "METHODS",
    "AussetCNF",
    "CoxPH",
    "DSM",
    "DeepHit",
    "DeepSurv",
    "FlowSurvAFTMethod",
    "FlowSurvGaussMethod",
    "LogNormalAFT",
    "RandomSurvivalForest",
    "RoystonParmar",
    "WeibullAFT",
]
