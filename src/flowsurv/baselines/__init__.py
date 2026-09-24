"""Baselines: classical (Cox/AFT/RSF/Royston-Parmar), deep (DeepSurv/DeepHit/DSM) and FlowSurv."""

from __future__ import annotations

from .ausset_cnf import AussetCNF
from .classical import CoxPH, LogNormalAFT, RandomSurvivalForest, WeibullAFT
from .common import SurvivalMethod
from .deep import DeepHit, DeepSurv
from .dsm import DSM
from .flowsurv_wrappers import FlowSurvAFTMethod, FlowSurvGaussMethod
from .royston_parmar import RoystonParmar

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
