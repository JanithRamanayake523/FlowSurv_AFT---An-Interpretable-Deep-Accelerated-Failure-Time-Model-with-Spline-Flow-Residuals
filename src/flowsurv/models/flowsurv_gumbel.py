"""FlowSurv-Gumbel ablation: identity flow with a minimum-Gumbel residual law.

The residual-shape counterpart of :mod:`flowsurv_gauss`. With the flow fixed
to the identity and the base law the standard minimum-Gumbel,

    Y = log T = mu(x) + sigma(x) * eps,   eps ~ Gumbel-min(0, 1)

is exactly a Weibull AFT with a deep location mu(x) and a deep scale sigma(x)
(Weibull shape k(x) = 1/sigma(x)). It answers the supervisor's question on
Deviation 1: a scenario whose covariate effect enters only through mu(x) and
sigma(x) around a fixed residual law (S4: a two-group Weibull mixture-free
regime switch) is representable *without* any flow conditioning, so if this
model also loses to DSM on S4 the cause is optimization or the discontinuity
in x, not the AFT structure.

Closed-form outputs, all in log-space (z = (log t - mu) / sigma):

    log f(t|x) = z - exp(z) - log t - log sigma
    log S(t|x) = -exp(z)
    F(t|x)     = 1 - exp(-exp(z))
    Q(q|x)     = exp(mu + sigma * log(-log(1 - q)))

The API mirrors :class:`~flowsurv.models.flowsurv_gauss.FlowSurvGauss`, whose
encoder plumbing (dummy unused spline head) and grid/paired handling it reuses.
"""

from __future__ import annotations

import torch
from torch import Tensor

from .flowsurv_gauss import FlowSurvGauss


class FlowSurvGumbel(FlowSurvGauss):
    """Identity-flow minimum-Gumbel ablation (deep-mu, deep-sigma Weibull AFT)."""

    def log_density(self, t: Tensor, x: Tensor, paired: bool | None = None) -> Tensor:
        """log f(t|x)."""
        z, log_t, sigma = self._forward_base(t, x, paired)
        return z - torch.exp(z) - log_t - sigma.log()

    def log_survival(self, t: Tensor, x: Tensor, paired: bool | None = None) -> Tensor:
        """log S(t|x) = -exp(z)."""
        z = self._forward_base(t, x, paired)[0]
        return -torch.exp(z)

    def cdf(self, t: Tensor, x: Tensor, paired: bool | None = None) -> Tensor:
        """F(t|x) = 1 - exp(-exp(z))."""
        z = self._forward_base(t, x, paired)[0]
        return -torch.expm1(-torch.exp(z))

    def log_cdf(self, t: Tensor, x: Tensor, paired: bool | None = None) -> Tensor:
        """log F(t|x)."""
        return torch.log(self.cdf(t, x, paired).clamp_min(torch.finfo(torch.float32).tiny))

    def quantile(self, q: Tensor | float, x: Tensor) -> Tensor:
        """Q(q|x) = exp(mu + sigma * log(-log(1 - q)))."""
        x = torch.as_tensor(x)
        p = x.shape[-1]
        mu, sigma, _ = self.encoder(x.reshape(-1, p))
        q = torch.as_tensor(q, dtype=mu.dtype, device=mu.device).flatten()
        n_q = q.numel()
        if mu.numel() % n_q != 0:
            raise ValueError(f"quantile levels {n_q} do not divide covariate batch {mu.numel()}")
        n_sub = mu.numel() // n_q
        z_q = torch.log(-torch.log1p(-q.clamp(1e-7, 1 - 1e-7))).unsqueeze(1).expand(-1, n_sub)
        return torch.exp(mu.reshape(n_q, n_sub) + sigma.reshape(n_q, n_sub) * z_q)

    def sample(self, x: Tensor, n: int = 1, generator: torch.Generator | None = None) -> Tensor:
        """One-pass samples t = exp(mu + sigma * log(-log(1 - U))), U ~ Uniform(0, 1)."""
        mu, sigma, _ = self.encoder(x)
        u = torch.rand((n,) + mu.shape, dtype=mu.dtype, device=mu.device, generator=generator)
        z = torch.log(-torch.log1p(-u.clamp(1e-7, 1 - 1e-7)))
        return torch.exp(mu.unsqueeze(0) + sigma.unsqueeze(0) * z)
