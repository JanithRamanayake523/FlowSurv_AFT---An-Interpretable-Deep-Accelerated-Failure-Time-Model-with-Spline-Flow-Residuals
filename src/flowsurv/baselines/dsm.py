"""Deep Survival Machines (DSM) baseline: a parametric Weibull mixture.

Implements a lightweight PyTorch Weibull mixture model that follows the same
:class:`~flowsurv.baselines.common.SurvivalMethod` interface.  Covariate
dependence enters through the mixture weights, shapes, and scales via a shared
MLP.  This is not the full ``auton-survival`` package, but it captures the
core distributional idea and keeps the stack self-contained.
"""

from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .common import FitResult, SurvivalMethod, as_output, hazard_from_survival, interp_surv, to_numpy


class _WeibullMixture(nn.Module):
    """Covariate-conditioned Weibull mixture.

    For each subject the network outputs mixture weights ``pi(x)``,
    log-shapes ``log_s(x)`` and log-scales ``log_l(x)``.  The conditional
    survival and density are

        S(t|x) = sum_k pi_k * exp(-(t/lambda_k)^shape_k)
        f(t|x) = sum_k pi_k * f_k(t|shape_k, lambda_k)
    """

    def __init__(
        self,
        in_features: int,
        n_components: int = 4,
        hidden: int = 128,
        n_blocks: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_components = n_components
        layers: list[nn.Module] = [nn.Linear(in_features, hidden), nn.ReLU(), nn.Dropout(dropout)]
        for _ in range(max(0, n_blocks - 1)):
            layers += [nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout)]
        self.trunk = nn.Sequential(*layers)
        self.logit_pi = nn.Linear(hidden, n_components)
        self.log_shape = nn.Linear(hidden, n_components)
        self.log_scale = nn.Linear(hidden, n_components)

    def _params(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        h = self.trunk(x)
        pi = F.softmax(self.logit_pi(h), dim=-1)
        shape = F.softplus(self.log_shape(h)).clamp_min(1e-3)
        scale = F.softplus(self.log_scale(h)).clamp_min(1e-3)
        return pi, shape, scale

    def _compute(self, t: Tensor, x: Tensor) -> tuple[Tensor, Tensor]:
        """Return (log_f, log_S) with shapes broadcastable as (m, n)."""
        pi, shape, scale = self._params(x)  # (n, K)
        t_b = t.unsqueeze(-1).unsqueeze(-1)  # (m, 1, 1)
        shape_b = shape.unsqueeze(0)  # (1, n, K)
        scale_b = scale.unsqueeze(0)
        pi_b = pi.unsqueeze(0)  # (1, n, K)

        y = (t_b / scale_b).clamp_min(1e-6)
        log_surv_i = -(y ** shape_b)
        log_f_i = (
            torch.log(shape_b)
            - torch.log(scale_b)
            + (shape_b - 1.0) * torch.log(y)
            + log_surv_i
        )
        log_s = torch.logsumexp(torch.log(pi_b) + log_surv_i, dim=-1)  # (m, n)
        log_f = torch.logsumexp(torch.log(pi_b) + log_f_i, dim=-1)
        return log_f, log_s

    def forward(self, t: Tensor, x: Tensor) -> Tensor:
        """Right-censored log-likelihood contribution (per subject)."""
        # expects t shape (n,) and x shape (n, p)
        log_f, log_s = self._compute(t, x)
        # diagonal of (n,n) -> per subject
        return torch.diagonal(log_f) - torch.diagonal(log_s)

    def predict_surv(self, t: Tensor, x: Tensor) -> Tensor:
        log_f, log_s = self._compute(t, x)
        return torch.exp(log_s)

    def predict_density(self, t: Tensor, x: Tensor) -> Tensor:
        log_f, log_s = self._compute(t, x)
        return torch.exp(log_f)

    def predict_hazard(self, t: Tensor, x: Tensor) -> Tensor:
        log_f, log_s = self._compute(t, x)
        return torch.exp(log_f - log_s)

    def predict_risk(self, x: Tensor) -> Tensor:
        # negative median lifetime on a coarse grid
        with torch.no_grad():
            pi, shape, scale = self._params(x)
            # rough range: scale * (log 2)^(1/shape) per component, use max
            t_grid = torch.linspace(1e-3, float((scale * 3.0).max()), 200, device=x.device, dtype=x.dtype)
            surv = self.predict_surv(t_grid, x)
            medians = []
            for j in range(x.shape[0]):
                s_j = surv[:, j]
                k = torch.searchsorted(-s_j, torch.tensor(-0.5, device=x.device, dtype=x.dtype))
                k = k.clamp(0, len(t_grid) - 1)
                medians.append(t_grid[k].item())
        return torch.tensor(medians, dtype=torch.float32)


@dataclass
class _DSMConfig:
    n_components: int = 4
    hidden: int = 128
    n_blocks: int = 2
    dropout: float = 0.1
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 256
    max_epochs: int = 500
    patience: int = 30
    device: str = "cpu"
    seed: int = 0


def _fit_dsm(
    model: _WeibullMixture,
    t: Tensor,
    d: Tensor,
    x: Tensor,
    cfg: _DSMConfig,
    val: tuple[Tensor, Tensor, Tensor] | None = None,
) -> FitResult:
    """Internal training loop for the DSM Weibull mixture."""
    device = torch.device(cfg.device)
    model.to(device)
    t, d, x = t.to(device), d.to(device), x.to(device)

    gen = torch.Generator(device="cpu").manual_seed(cfg.seed)
    n = t.shape[0]
    if val is None:
        n_val = max(1, int(round(0.15 * n)))
        perm = torch.randperm(n, generator=gen)
        idx_val, idx_tr = perm[:n_val].to(device), perm[n_val:].to(device)
    else:
        t_v, d_v, x_v = (v.to(device) for v in val)
        idx_val = torch.arange(t_v.shape[0], device=device)
        idx_tr = torch.arange(n, device=device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.max_epochs, eta_min=1e-5)

    best_state = None
    best_val = math.inf
    stale = 0
    start = time.perf_counter()
    converged = True

    for epoch in range(cfg.max_epochs):
        model.train()
        order = idx_tr[torch.randperm(idx_tr.shape[0], generator=gen)]
        for batch in order.split(cfg.batch_size):
            opt.zero_grad()
            log_f, log_s = model._compute(t[batch], x[batch])
            ll = d[batch] * torch.diagonal(log_f) + (1.0 - d[batch]) * torch.diagonal(log_s)
            loss = -ll.mean()
            loss.backward()
            opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            if val is None:
                log_f, log_s = model._compute(t[idx_val], x[idx_val])
                ll = d[idx_val] * torch.diagonal(log_f) + (1.0 - d[idx_val]) * torch.diagonal(log_s)
            else:
                log_f, log_s = model._compute(t_v, x_v)
                ll = d_v * torch.diagonal(log_f) + (1.0 - d_v) * torch.diagonal(log_s)
            val_loss = float(-ll.mean())

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= cfg.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    wall_time_s = time.perf_counter() - start
    info = {"best_val_loss": best_val}
    return FitResult(wall_time_s=wall_time_s, converged=converged, info=info)


class DSM(SurvivalMethod):
    """Deep Survival Machines — covariate-conditioned Weibull mixture."""

    name: str = "dsm"
    is_deep: bool = True
    supports_density: bool = True
    supports_hazard: bool = True

    def __init__(self) -> None:
        self.model: _WeibullMixture | None = None
        self.tuning_space: dict[str, list] = {
            "n_components": [2, 4, 8],
            "hidden": [64, 128],
            "n_blocks": [1, 2, 3],
            "dropout": [0.1, 0.2],
            "lr": [1e-3, 5e-4],
            "batch_size": [256],
        }

    def fit(
        self,
        t: Tensor,
        d: Tensor,
        x: Tensor,
        *,
        val: tuple[Tensor, Tensor, Tensor] | None = None,
        seed: int = 0,
        **hyper,
    ) -> FitResult:
        t, d, x = (torch.as_tensor(v, dtype=torch.float32) for v in (t, d, x))
        torch.manual_seed(seed)
        self.model = _WeibullMixture(
            int(x.shape[1]),
            n_components=hyper.get("n_components", 4),
            hidden=hyper.get("hidden", 128),
            n_blocks=hyper.get("n_blocks", 2),
            dropout=hyper.get("dropout", 0.1),
        )
        cfg = _DSMConfig(
            n_components=hyper.get("n_components", 4),
            hidden=hyper.get("hidden", 128),
            n_blocks=hyper.get("n_blocks", 2),
            dropout=hyper.get("dropout", 0.1),
            lr=hyper.get("lr", 1e-3),
            weight_decay=hyper.get("weight_decay", 1e-4),
            batch_size=hyper.get("batch_size", 256),
            max_epochs=hyper.get("max_epochs", 500),
            patience=hyper.get("patience", 30),
            device=hyper.get("device", "cpu"),
            seed=seed,
        )
        return _fit_dsm(self.model, t, d, x, cfg, val=val)

    def predict_surv(self, t: Tensor, x: Tensor) -> Tensor:
        if self.model is None:
            raise RuntimeError("DSM has not been fit")
        t, x = (torch.as_tensor(v, dtype=torch.float32) for v in (t, x))
        with torch.no_grad():
            return as_output(self.model.predict_surv(t, x).numpy())

    def predict_density(self, t: Tensor, x: Tensor) -> Tensor:
        if self.model is None:
            raise RuntimeError("DSM has not been fit")
        t, x = (torch.as_tensor(v, dtype=torch.float32) for v in (t, x))
        with torch.no_grad():
            return as_output(self.model.predict_density(t, x).numpy())

    def predict_hazard(self, t: Tensor, x: Tensor) -> Tensor:
        if self.model is None:
            raise RuntimeError("DSM has not been fit")
        t, x = (torch.as_tensor(v, dtype=torch.float32) for v in (t, x))
        with torch.no_grad():
            return as_output(self.model.predict_hazard(t, x).numpy())

    def predict_risk(self, x: Tensor) -> Tensor:
        if self.model is None:
            raise RuntimeError("DSM has not been fit")
        x = torch.as_tensor(x, dtype=torch.float32)
        with torch.no_grad():
            return as_output(-self.model.predict_risk(x).numpy())
