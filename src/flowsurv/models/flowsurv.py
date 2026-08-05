"""FlowSurv-AFT: deep AFT with a spline-flow residual law.

Model (Methodology Sec. 2.2):

    Y = log T = mu(x) + sigma(x) * eps,   eps = g^{-1}(z ; h(x)),   z ~ Logistic(0, 1)

All distributional outputs are exact and evaluated in log-space
(Methodology Sec. 2.2.1, 2.3). Writing u(t) = (log t - mu) / sigma and
g = g(.; h(x)) with base CDF/density F0/f0 of the standard logistic:

    F(t|x) = F0(g(u(t)))                        -> :meth:`cdf`
    S(t|x) = 1 - F0(g(u(t)))                    -> :meth:`survival` / :meth:`log_survival`
    f(t|x) = f0(g(u(t))) * |g'(u(t))| / (t * sigma) -> :meth:`density` / :meth:`log_density`
    h(t|x) = f / S                              -> :meth:`hazard`
    Q(q|x) = exp(mu + sigma * g^{-1}(F0^{-1}(q)))   -> :meth:`quantile`
    sample: z ~ p0, t = exp(mu + sigma * g^{-1}(z)) -> :meth:`sample`

F0 is the logistic sigmoid; log F0 = log sigmoid(z), log S0 = log sigmoid(-z),
log f0 = log sigmoid(z) + log sigmoid(-z) -- all numerically stable on R.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .encoder import FlowSurvEncoder
from .rqs_flow import ConditionalRQSFlow


class FlowSurvAFT(nn.Module):
    """Full FlowSurv-AFT model (encoder + conditional RQS residual flow).

    Args:
        n_features: number of covariates p.
        hidden: encoder trunk width.
        n_blocks: encoder residual blocks; 0 selects the linear encoder.
        dropout: encoder dropout rate.
        bins: RQS bins per block (Methodology Sec. 2.4 tunes 8/16).
        bound: spline domain [-bound, bound] in standardized residual units,
            identity tails outside (Sec. 2.4 fixes 4).
        n_spline_blocks: number K of stacked RQS blocks (Sec. 2.4 tunes 1-3).
    """

    def __init__(
        self,
        n_features: int,
        hidden: int = 128,
        n_blocks: int = 2,
        dropout: float = 0.1,
        bins: int = 8,
        bound: float = 4.0,
        n_spline_blocks: int = 1,
    ) -> None:
        super().__init__()
        self.flow = ConditionalRQSFlow(bins=bins, bound=bound, n_blocks=n_spline_blocks)
        self.encoder = FlowSurvEncoder(
            n_features, hidden=hidden, n_blocks=n_blocks, dropout=dropout, flow=self.flow
        )

    # ------------------------------------------------------------------
    # internal helpers

    def _forward_base(self, t: Tensor, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """(z = g(u(t)), log|g'(u(t))|, log t, sigma) for times ``t`` given ``x``."""
        mu, sigma, params = self.encoder(x)
        log_t = t.to(mu.dtype).clamp_min(torch.finfo(mu.dtype).tiny).log()
        u = (log_t - mu) / sigma
        z, ladj = self.flow.forward(u, params)
        return z, ladj, log_t, sigma

    @staticmethod
    def _log_f0(z: Tensor) -> Tensor:
        """Standard-logistic log-density, stable on the whole real line."""
        return F.logsigmoid(z) + F.logsigmoid(-z)

    # ------------------------------------------------------------------
    # exact closed-form outputs (Methodology Sec. 2.2.1)

    def log_density(self, t: Tensor, x: Tensor) -> Tensor:
        """log f(t|x)."""
        z, ladj, log_t, sigma = self._forward_base(t, x)
        return self._log_f0(z) + ladj - log_t - sigma.log()

    def density(self, t: Tensor, x: Tensor) -> Tensor:
        """f(t|x)."""
        return self.log_density(t, x).exp()

    def log_survival(self, t: Tensor, x: Tensor) -> Tensor:
        """log S(t|x) = log sigmoid(-g(u(t))), stable under heavy censoring."""
        z = self._forward_base(t, x)[0]
        return F.logsigmoid(-z)

    def survival(self, t: Tensor, x: Tensor) -> Tensor:
        """S(t|x)."""
        return self.log_survival(t, x).exp()

    def cdf(self, t: Tensor, x: Tensor) -> Tensor:
        """F(t|x) = sigmoid(g(u(t)))."""
        z = self._forward_base(t, x)[0]
        return torch.sigmoid(z)

    def log_cdf(self, t: Tensor, x: Tensor) -> Tensor:
        """log F(t|x) = log sigmoid(g(u(t)))."""
        z = self._forward_base(t, x)[0]
        return F.logsigmoid(z)

    def hazard(self, t: Tensor, x: Tensor) -> Tensor:
        """h(t|x) = f(t|x) / S(t|x)."""
        return (self.log_density(t, x) - self.log_survival(t, x)).exp()

    def quantile(self, q: Tensor | float, x: Tensor) -> Tensor:
        """Q(q|x) = exp(mu + sigma * g^{-1}(logit(q))). q broadcastable to x's batch."""
        mu, sigma, params = self.encoder(x)
        q = torch.as_tensor(q, dtype=mu.dtype, device=mu.device)
        z_q = torch.logit(q.clamp(1e-6, 1 - 1e-6))
        return torch.exp(mu + sigma * self.flow.inverse(z_q, params))

    def sample(self, x: Tensor, n: int = 1, generator: torch.Generator | None = None) -> Tensor:
        """One-pass samples t = exp(mu + sigma * g^{-1}(z)), z ~ Logistic(0, 1).

        Returns a tensor of shape ``(n, batch)``.
        """
        mu, sigma, params = self.encoder(x)
        shape = (n,) + mu.shape
        # inverse-CDF sampling from Logistic(0, 1); clamp keeps logit finite
        u01 = torch.rand(shape, dtype=mu.dtype, device=mu.device, generator=generator)
        z = torch.logit(u01.clamp(1e-7, 1 - 1e-7))
        params = params.unsqueeze(0).expand(shape + (params.shape[-1],))
        eps = self.flow.inverse(z, params)
        return torch.exp(mu.unsqueeze(0) + sigma.unsqueeze(0) * eps)

    def predict_risk(self, x: Tensor) -> Tensor:
        """Concordance risk score: negative median lifetime -Q(0.5|x) (higher = riskier)."""
        return -self.quantile(0.5, x)

    def forward(self, t: Tensor, d: Tensor, x: Tensor) -> Tensor:
        """Per-observation right-censored log-likelihood contribution."""
        log_f = self.log_density(t, x)
        log_s = self.log_survival(t, x)
        return d * log_f + (1 - d) * log_s
