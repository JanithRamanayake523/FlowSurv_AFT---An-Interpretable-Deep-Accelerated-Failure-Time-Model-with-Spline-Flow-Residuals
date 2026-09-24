"""Baselines: classical (Cox/AFT/RSF/Royston-Parmar), deep (DeepSurv/DeepHit/DSM) and FlowSurv."""

from __future__ import annotations

from .ausset_cnf import AussetCNF
from .classical import CoxPH, LogNormalAFT, RandomSurvivalForest, WeibullAFT
from .common import SurvivalMethod
from .deep import DeepHit, DeepSurv
from .dsm import DSM
from .flowsurv_wrappers import (
    FlowSurvAFTMethod,
    FlowSurvGaussMethod,
    FlowSurvGumbelMethod,
    FlowSurvStrictAFTMethod,
)
from .royston_parmar import RoystonParmar

METHODS: dict[str, type[SurvivalMethod]] = {
    "flowsurv_aft": FlowSurvAFTMethod,
    "flowsurv_gauss": FlowSurvGaussMethod,
    "flowsurv_strict_aft": FlowSurvStrictAFTMethod,
    "flowsurv_gumbel": FlowSurvGumbelMethod,
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

#: Diagnostic/ablation methods: registered (runnable via ``--methods``) but
#: excluded from the default full-grid method list.
ABLATION_METHODS: frozenset[str] = frozenset({"flowsurv_strict_aft", "flowsurv_gumbel"})

__all__ = [
    "ABLATION_METHODS",
    "METHODS",
    "AussetCNF",
    "CoxPH",
    "DSM",
    "DeepHit",
    "DeepSurv",
    "FlowSurvAFTMethod",
    "FlowSurvGaussMethod",
    "FlowSurvGumbelMethod",
    "FlowSurvStrictAFTMethod",
    "LogNormalAFT",
    "RandomSurvivalForest",
    "RoystonParmar",
    "WeibullAFT",
]
