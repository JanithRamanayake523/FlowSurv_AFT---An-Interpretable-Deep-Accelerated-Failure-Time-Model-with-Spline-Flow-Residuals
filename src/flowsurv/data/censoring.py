"""Censoring mechanisms (Methodology Sec. 3.1).

Phase 1 scope: plain Type I (administrative) censoring at an empirical
quantile, needed by the Phase 1 smoke fit. Per-cell calibrated Type I /
Type III censoring with +-1% tolerance lands in Phase 2.
"""

from __future__ import annotations

import torch
from torch import Tensor


def censor_type1(t: Tensor, target: float) -> tuple[Tensor, Tensor]:
    """Type I censoring at the empirical (1 - target) quantile of ``t``.

    Returns ``(t_obs, delta)`` with t_obs = min(T, c) and delta = 1{T <= c},
    giving approximately ``target`` censoring proportion.
    """
    if not 0.0 <= target < 1.0:
        raise ValueError("target censoring proportion must be in [0, 1)")
    c = torch.quantile(t, 1.0 - target)
    delta = (t <= c).float()
    return torch.minimum(t, c), delta
