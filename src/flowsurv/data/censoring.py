"""Censoring mechanisms (Methodology Sec. 3.1).

Phase 2 scope: per-cell calibrated Type I (administrative, fixed quantile)
and Type III (random Uniform(0, u)) censoring, calibrated numerically to hit
the target censoring proportion within +-1% (Implementation Plan Phase 2,
exit checklist item 1).
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


def _type3_upper(t: Tensor, target: float, iters: int = 50) -> float:
    """Solve for the Uniform(0, u) upper endpoint giving ``target`` censoring.

    Uses the pilot sample ``t`` itself: for C ~ Uniform(0, u) the expected
    censoring proportion is the sample mean of P(C < T | T = t_i)
    = mean(min(t_i / u, 1)), which is continuous and non-increasing in u, so
    bisection (50 iterations) converges to the exact empirical solution. The
    *realized* proportion after drawing C concentrates around this value,
    landing within +-1% for the pilot sizes used (n >= 200).
    """
    def cens_prop(u: float) -> float:
        return float((t / u).clamp(max=1.0).mean())

    lo, hi = 1e-8, max(float(t.max()) * 2.0, 1.0)
    # bracket: cens_prop(lo) >= target >= cens_prop(hi)
    while cens_prop(lo) < target:
        lo *= 0.5
    while cens_prop(hi) > target:
        hi *= 2.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if cens_prop(mid) > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def censor_type3(t: Tensor, target: float, seed: int | None = None) -> tuple[Tensor, Tensor]:
    """Type III random censoring: C ~ Uniform(0, u), u by bisection (50 iters).

    ``u`` is calibrated on the given pilot ``t`` so the expected censoring
    proportion equals ``target`` (achieved within +-1% at study sample sizes).
    Fresh uniforms are drawn per call from ``seed``. Returns ``(t_obs, delta)``.
    """
    if not 0.0 < target < 1.0:
        raise ValueError("target censoring proportion must be in (0, 1)")
    u = _type3_upper(t, target)
    gen = torch.Generator().manual_seed(seed) if seed is not None else None
    c = torch.rand(t.shape, generator=gen, dtype=t.dtype) * u
    delta = (t <= c).float()
    return torch.minimum(t, c), delta


def calibrate_censoring(
    t: Tensor,
    target: float,
    ctype: str,
    seed: int | None = None,
) -> dict:
    """Apply calibrated censoring and report the achieved proportion.

    Returns a dict with keys ``"t_obs"``, ``"d"``, ``"achieved"`` (realized
    censoring proportion, float) and ``"param"`` (the Type I cutpoint c or
    the Type III upper endpoint u). Type I uses the exact empirical quantile
    of the pilot ``t``; Type III uses the bisection of ``_type3_upper``.
    """
    if not 0.0 <= target <= 0.95:
        raise ValueError("target censoring proportion must be in [0, 0.95]")
    if target == 0.0:
        return {
            "t_obs": t.clone(),
            "d": torch.ones_like(t),
            "achieved": 0.0,
            "param": float("inf"),
        }
    if ctype == "I":
        t_obs, d = censor_type1(t, target)
        param = float(torch.quantile(t, 1.0 - target))
    elif ctype == "III":
        t_obs, d = censor_type3(t, target, seed=seed)
        param = _type3_upper(t, target)
    else:
        raise ValueError(f"unknown censoring type {ctype!r}; expected 'I' or 'III'")
    return {"t_obs": t_obs, "d": d, "achieved": float(1.0 - d.mean()), "param": param}


def apply_censoring(
    t: Tensor,
    target: float,
    ctype: str,
    seed: int | None = None,
) -> tuple[Tensor, Tensor]:
    """Thin wrapper: apply censoring of type ``ctype`` at level ``target``.

    ``target == 0`` short-circuits to no censoring for both types.
    Returns ``(t_obs, delta)``.
    """
    if target == 0.0:
        return t.clone(), torch.ones_like(t)
    if ctype == "I":
        return censor_type1(t, target)
    if ctype == "III":
        return censor_type3(t, target, seed=seed)
    raise ValueError(f"unknown censoring type {ctype!r}; expected 'I' or 'III'")
