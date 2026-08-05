"""Data: S1-S6 simulation DGPs, censoring mechanisms, real-data loaders (Phase 2/7)."""

from .censoring import censor_type1
from .dgp import S1_BETA, S1_K, S1_LAMBDA, s1_true_log_density, simulate_s1

__all__ = [
    "S1_BETA",
    "S1_K",
    "S1_LAMBDA",
    "censor_type1",
    "s1_true_log_density",
    "simulate_s1",
]
