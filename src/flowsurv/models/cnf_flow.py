"""Conditional 1-D continuous normalizing flow (Ausset et al. 2021 style).

Unlike :class:`~flowsurv.models.rqs_flow.ConditionalRQSFlow` (analytic
forward/inverse/log-det, no solver), this flow's forward and inverse are both
*numerical ODE integrations* -- exactly the "numerical solver per
evaluation" property that differentiates FlowSurv-AFT's exact spline flow
from Ausset's continuous-flow approach (AGENTS.md, differentiator 1). It
exists purely as an empirical competitor baseline (``ausset_cnf``), built on
the same AFT skeleton as FlowSurv-AFT so the comparison isolates "spline
flow vs. continuous flow" rather than differing in any other respect.

Model: a shared velocity network f_theta(z, tau, h(x)) with

    dz/dtau = f_theta(z, tau, h(x)),   tau in [0, 1]

maps residual u = z(0) to base z = z(1) (``forward``) or back (``inverse``).
The residual is scalar (matching the RQS flow's per-observation z), so the
Jacobian trace needed by the instantaneous change-of-variables formula is
just the scalar derivative d f_theta/dz -- exact, no Hutchinson estimator
needed. Integrated by fixed-step RK4 (no ``torchdiffeq`` dependency).

Change of variables (Chen et al. 2018; Grathwohl et al. 2018, FFJORD):
d/dtau log|det dz(tau)/dz(0)| = tr(df/dz) = df/dz (1-D), so

    log|dz(1)/dz(0)| = integral_0^1 (df_theta/dz) dtau

which is exactly the ``ladj`` :meth:`forward` returns, on the same footing
as ``ConditionalRQSFlow.forward``'s analytic ``log|g'(u)|``.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class ConditionalCNFFlow(nn.Module):
    """Shared velocity-network CNF conditioned on a per-observation embedding.

    Unlike ``ConditionalRQSFlow`` (stateless; parameters flow in per-call via
    ``params``), this flow owns learnable weights (the velocity net) shared
    across all observations -- per-subject variation enters through the
    conditioning vector ``h(x)`` (the ``params`` argument, sized
    ``n_params = cond_dim``), matching how ``FlowSurvEncoder`` sizes its
    conditioner head off ``flow.n_params`` regardless of what it represents.

    Args:
        cond_dim: dimension of the per-observation conditioning embedding.
        hidden: velocity-net hidden width.
        n_steps: fixed RK4 steps integrating tau over [0, 1].
    """

    def __init__(self, cond_dim: int = 16, hidden: int = 32, n_steps: int = 20) -> None:
        super().__init__()
        if cond_dim < 1:
            raise ValueError("cond_dim must be >= 1")
        if n_steps < 1:
            raise ValueError("n_steps must be >= 1")
        self.cond_dim = cond_dim
        self.n_params = cond_dim  # matches ConditionalRQSFlow's contract
        self.n_steps = n_steps
        self.net = nn.Sequential(
            nn.Linear(2 + cond_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        # Zero-init the final layer: velocity ~= 0 at init, so z(1) ~= z(0)
        # and the flow starts close to the identity map -- the same warm
        # start rationale as the RQS flow's zero-initialized spline head.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def _velocity(self, z: Tensor, tau: float, h: Tensor) -> Tensor:
        tau_b = torch.full_like(z, float(tau)).unsqueeze(-1)
        inp = torch.cat([z.unsqueeze(-1), tau_b, h], dim=-1)
        return self.net(inp).squeeze(-1)

    def _velocity_and_div(self, z: Tensor, tau: float, h: Tensor) -> tuple[Tensor, Tensor]:
        """(velocity, exact scalar divergence df/dz) at one RK4 stage.

        The divergence needs ``torch.autograd.grad`` regardless of the
        ambient grad mode (e.g. prediction under an outer ``no_grad()``), so
        grad tracking is explicitly re-enabled here. ``create_graph`` is tied
        to ``self.training`` (set correctly by ``model.train()``/``.eval()``
        around ``models/training.py``'s fit loop): during training the
        divergence's own dependence on the velocity net's parameters must
        stay differentiable for the NLL's ``ladj`` term to backpropagate
        correctly; at inference only the value is needed.
        """
        with torch.enable_grad():
            z_in = z if z.requires_grad else z.detach().requires_grad_(True)
            v = self._velocity(z_in, tau, h)
            (dv_dz,) = torch.autograd.grad(v.sum(), z_in, create_graph=self.training)
        return v, dv_dz

    def _integrate(self, z0: Tensor, h: Tensor, reverse: bool) -> tuple[Tensor, Tensor]:
        """RK4 integration of the augmented (z, logdet) ODE over tau in [0, 1].

        ``reverse=False``: tau 0 -> 1 (residual -> base, "forward").
        ``reverse=True``:  tau 1 -> 0 (base -> residual, "inverse").
        """
        dtau = (-1.0 if reverse else 1.0) / self.n_steps
        tau = 1.0 if reverse else 0.0
        z = z0
        logdet = torch.zeros_like(z0)
        for _ in range(self.n_steps):
            k1_v, k1_d = self._velocity_and_div(z, tau, h)
            k2_v, k2_d = self._velocity_and_div(z + 0.5 * dtau * k1_v, tau + 0.5 * dtau, h)
            k3_v, k3_d = self._velocity_and_div(z + 0.5 * dtau * k2_v, tau + 0.5 * dtau, h)
            k4_v, k4_d = self._velocity_and_div(z + dtau * k3_v, tau + dtau, h)
            z = z + (dtau / 6.0) * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)
            logdet = logdet + (dtau / 6.0) * (k1_d + 2 * k2_d + 2 * k3_d + k4_d)
            tau = tau + dtau
        return z, logdet

    def forward(self, u: Tensor, params: Tensor) -> tuple[Tensor, Tensor]:
        """Residual ``u`` -> base ``z`` via ODE integration tau: 0 -> 1.

        Returns ``(z, ladj)``, ``ladj = log|dz/du|`` (no closed form -- a
        numerical solve, unlike ``ConditionalRQSFlow.forward``).
        """
        return self._integrate(u, params, reverse=False)

    def inverse(self, z: Tensor, params: Tensor) -> Tensor:
        """Base ``z`` -> residual ``u`` via ODE integration tau: 1 -> 0."""
        u, _ = self._integrate(z, params, reverse=True)
        return u
