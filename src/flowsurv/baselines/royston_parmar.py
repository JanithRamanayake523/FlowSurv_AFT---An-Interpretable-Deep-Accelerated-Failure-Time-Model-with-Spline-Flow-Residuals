"""Royston-Parmar flexible parametric survival model (native Python, hazard scale).

Royston & Parmar (2002): on the log cumulative-hazard scale,

    log H(t|x) = s(log t; gamma) + x . beta            (proportional hazards)

where ``s`` is a restricted cubic spline (Durrleman & Simon 1989) in log t
with knots at quantiles of the observed *event* log-times. This is the same
model ``flexsurv::flexsurvspline(..., scale="hazard")`` fits via R -- the
original design here used ``rpy2`` (Methodology Sec. 7 flags this as fragile
on Windows). Reimplemented directly in PyTorch (MLE by L-BFGS on the exact
right-censored log-likelihood) so the method needs no R/rpy2 install and no
CSV export/import bridge.

Hazard-scale outputs, given eta(t|x) = s(log t) + x.beta:

    H(t|x) = exp(eta)                     cumulative hazard
    S(t|x) = exp(-H)                      survival
    h(t|x) = H * ds/d(log t) / t          hazard (chain rule; ds/d log t does
                                           not depend on x under the PH form)
    f(t|x) = h * S                        density

``df`` (default 4) sets the number of knots (2 boundary + df-2 interior,
placed at equally spaced quantiles of the log event-times, as in flexsurv);
if a validation split is supplied, ``fit`` selects df from ``{3, 4, 5}`` by
validation NLL (Methodology Sec. 2.4's "documented tuning" for classical
methods) instead of the general 30-config DL protocol, matching the
prereg's "package defaults with documented tuning where applicable".
"""

from __future__ import annotations

import time
import warnings
from typing import Any

import numpy as np
import torch
from torch import Tensor

from .common import FitResult, SurvivalMethod, as_output, to_numpy

_DF_CANDIDATES = (3, 4, 5)
_MIN_LOG_T = -20.0  # clamp floor for log(t) with t -> 0


def _rcs_basis(u: Tensor, knots: Tensor) -> Tensor:
    """Restricted cubic spline basis (Durrleman & Simon 1989 / flexsurv).

    ``u`` (m,), ``knots`` (K,) ascending, K >= 2. Returns (m, K): columns
    ``[1, u, v_2(u), ..., v_{K-1}(u)]``, linear outside ``[knots[0],
    knots[-1]]`` and cubic-spline interior (the "restricted" / natural
    boundary condition).
    """
    K = knots.shape[0]
    k1, kK = knots[0], knots[-1]
    cols = [torch.ones_like(u), u]
    for j in range(1, K - 1):
        kj = knots[j]
        lam = (kK - kj) / (kK - k1)
        term = (
            torch.clamp(u - kj, min=0) ** 3
            - lam * torch.clamp(u - k1, min=0) ** 3
            - (1 - lam) * torch.clamp(u - kK, min=0) ** 3
        )
        cols.append(term)
    return torch.stack(cols, dim=1)


def _rcs_basis_deriv(u: Tensor, knots: Tensor) -> Tensor:
    """d/du of :func:`_rcs_basis`, same (m, K) shape."""
    K = knots.shape[0]
    k1, kK = knots[0], knots[-1]
    cols = [torch.zeros_like(u), torch.ones_like(u)]
    for j in range(1, K - 1):
        kj = knots[j]
        lam = (kK - kj) / (kK - k1)
        term = (
            3 * torch.clamp(u - kj, min=0) ** 2
            - lam * 3 * torch.clamp(u - k1, min=0) ** 2
            - (1 - lam) * 3 * torch.clamp(u - kK, min=0) ** 2
        )
        cols.append(term)
    return torch.stack(cols, dim=1)


def _knots_from_events(log_t_events: np.ndarray, df: int) -> Tensor:
    """K = df knots at equally spaced quantiles of the observed log event-times."""
    n_knots = max(2, df)
    qs = np.linspace(0.0, 1.0, n_knots)
    knots = np.quantile(log_t_events, qs)
    # guard degenerate ties (e.g. very few distinct event times): nudge apart
    knots = np.maximum.accumulate(knots)
    eps = 1e-4
    for i in range(1, len(knots)):
        if knots[i] <= knots[i - 1]:
            knots[i] = knots[i - 1] + eps
    return torch.tensor(knots, dtype=torch.float32)


def _neg_log_lik(
    gamma: Tensor, beta: Tensor, basis: Tensor, dbasis: Tensor, u: Tensor, xx: Tensor, dd: Tensor
) -> Tensor:
    eta = basis @ gamma + xx @ beta  # (n,) log cumulative hazard
    dsdu = dbasis @ gamma  # (n,) d s / d(log t); does not involve beta (PH form)
    H = eta.exp()
    # log hazard: h = H * dsdu / t = exp(eta + log(dsdu) - u); clamp guards the
    # (rare, off-optimum) region where the unconstrained spline's slope dips
    # non-positive -- flexsurv has the same well-known limitation.
    log_h = eta + dsdu.clamp_min(1e-6).log() - u
    return -(dd * log_h - H).mean()


def _fit_one_df(t_np: np.ndarray, d_np: np.ndarray, x_np: np.ndarray, df: int, seed: int) -> dict:
    """MLE for one df; returns state dict + achieved training NLL."""
    log_t = np.log(np.clip(t_np, np.exp(_MIN_LOG_T), None))
    event_log_t = log_t[d_np.astype(bool)]
    if event_log_t.size < 2:
        event_log_t = log_t  # degenerate (no/near-no events): fall back to all times
    knots = _knots_from_events(event_log_t, df)

    u = torch.tensor(log_t, dtype=torch.float32)
    xx = torch.tensor(x_np, dtype=torch.float32)
    dd = torch.tensor(d_np, dtype=torch.float32)
    basis = _rcs_basis(u, knots)
    dbasis = _rcs_basis_deriv(u, knots)

    torch.manual_seed(seed)
    K, p = knots.shape[0], xx.shape[1]
    gamma = torch.zeros(K, requires_grad=True)
    with torch.no_grad():
        gamma[1] = 1.0  # init s(u) ~= gamma0 + u: an Exponential-ish baseline hazard
    beta = torch.zeros(p, requires_grad=True)

    opt = torch.optim.LBFGS(
        [gamma, beta], lr=0.5, max_iter=300, tolerance_grad=1e-8, tolerance_change=1e-10, line_search_fn="strong_wolfe"
    )

    def closure():
        opt.zero_grad()
        loss = _neg_log_lik(gamma, beta, basis, dbasis, u, xx, dd)
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        nll = float(_neg_log_lik(gamma, beta, basis, dbasis, u, xx, dd))
    if not np.isfinite(nll):
        raise RuntimeError(f"Royston-Parmar df={df} fit diverged (NLL={nll})")
    return {"gamma": gamma.detach(), "beta": beta.detach(), "knots": knots, "df": df, "train_nll": nll}


class RoystonParmar(SurvivalMethod):
    """Royston-Parmar flexible parametric baseline (hazard-scale, native Python)."""

    name: str = "royston_parmar"
    is_deep: bool = False
    supports_density: bool = True
    supports_hazard: bool = True

    def __init__(self) -> None:
        self.gamma: Tensor | None = None
        self.beta: Tensor | None = None
        self.knots: Tensor | None = None
        self.df: int | None = None
        self.tuning_space: dict[str, list] = {"df": list(_DF_CANDIDATES)}

    def fit(
        self,
        t: Tensor,
        d: Tensor,
        x: Tensor,
        *,
        val: tuple[Tensor, Tensor, Tensor] | None = None,
        seed: int = 0,
        **hyper: Any,
    ) -> FitResult:
        t_np, d_np, x_np = to_numpy(t, d, x)
        start = time.perf_counter()
        try:
            if "df" in hyper:
                candidates = [int(hyper["df"])]
            elif val is not None:
                candidates = list(_DF_CANDIDATES)  # internal df-selection (Methodology Sec. 2.4)
            else:
                candidates = [4]

            fits = [_fit_one_df(t_np, d_np, x_np, df, seed) for df in candidates]

            if len(fits) > 1 and val is not None:
                t_v, d_v, x_v = to_numpy(*val)
                scored = []
                for state in fits:
                    self._load(state)
                    log_f = self._log_density_np(t_v, x_v)
                    log_s = self._log_survival_np(t_v, x_v)
                    val_nll = -float(np.mean(d_v * log_f + (1.0 - d_v) * log_s))
                    scored.append((val_nll, state))
                best = min(scored, key=lambda pair: pair[0])[1]
            else:
                best = fits[0]

            self._load(best)
            converged = True
            info = {"df": best["df"], "train_nll": best["train_nll"]}
        except Exception as e:  # noqa: BLE001 -- failures are a reported finding (prereg Sec. 5)
            converged = False
            info = {"error": repr(e)}
            warnings.warn(f"{self.name} fit failed: {e!r}")
        return FitResult(wall_time_s=time.perf_counter() - start, converged=converged, info=info)

    def _load(self, state: dict) -> None:
        self.gamma, self.beta, self.knots, self.df = state["gamma"], state["beta"], state["knots"], state["df"]

    def _require_fit(self) -> None:
        if self.gamma is None:
            raise RuntimeError(f"{self.name} has not been fit successfully")

    def _eta_grid(self, t: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(m, n) log cumulative hazard, d s / d log t (m,), and u = log t (m,)."""
        self._require_fit()
        u = torch.tensor(np.log(np.clip(t, np.exp(_MIN_LOG_T), None)), dtype=torch.float32)
        xx = torch.tensor(x, dtype=torch.float32)
        with torch.no_grad():
            basis = _rcs_basis(u, self.knots)  # (m, K)
            dbasis = _rcs_basis_deriv(u, self.knots)  # (m, K)
            s_grid = basis @ self.gamma  # (m,)
            dsdu = dbasis @ self.gamma  # (m,)
            xb = xx @ self.beta  # (n,)
            eta = s_grid.unsqueeze(1) + xb.unsqueeze(0)  # (m, n)
        return eta.numpy(), dsdu.numpy(), u.numpy()

    def predict_surv(self, t: Tensor, x: Tensor) -> Tensor:
        t_np, _, x_np = to_numpy(t, torch.zeros_like(t), x)
        eta, _, _ = self._eta_grid(t_np, x_np)
        return as_output(np.exp(-np.exp(eta)))

    def predict_density(self, t: Tensor, x: Tensor) -> Tensor:
        t_np, _, x_np = to_numpy(t, torch.zeros_like(t), x)
        eta, dsdu, u = self._eta_grid(t_np, x_np)
        H = np.exp(eta)
        S = np.exp(-H)
        h = H * np.clip(dsdu, 1e-6, None)[:, None] / np.exp(u)[:, None]
        return as_output(h * S)

    def predict_hazard(self, t: Tensor, x: Tensor) -> Tensor:
        t_np, _, x_np = to_numpy(t, torch.zeros_like(t), x)
        eta, dsdu, u = self._eta_grid(t_np, x_np)
        H = np.exp(eta)
        h = H * np.clip(dsdu, 1e-6, None)[:, None] / np.exp(u)[:, None]
        return as_output(h)

    def _log_density_np(self, t: np.ndarray, x: np.ndarray) -> np.ndarray:
        eta, dsdu, u = self._eta_grid(t, x)
        eta_diag = np.diagonal(eta)
        log_h = eta_diag + np.log(np.clip(dsdu, 1e-6, None)) - u
        return log_h - np.exp(eta_diag)

    def _log_survival_np(self, t: np.ndarray, x: np.ndarray) -> np.ndarray:
        eta, _, _ = self._eta_grid(t, x)
        return -np.exp(np.diagonal(eta))

    def predict_risk(self, x: Tensor) -> Tensor:
        self._require_fit()
        _, _, x_np = to_numpy(torch.zeros(x.shape[0]), torch.zeros(x.shape[0]), x)
        xb = (torch.tensor(x_np, dtype=torch.float32) @ self.beta).numpy()
        return as_output(np.exp(xb))  # PH scale: higher x.beta = higher hazard = riskier
