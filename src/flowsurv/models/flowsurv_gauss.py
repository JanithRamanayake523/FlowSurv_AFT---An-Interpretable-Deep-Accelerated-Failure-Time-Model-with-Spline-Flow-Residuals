"""FlowSurv-Gauss ablation: identity flow with a Gaussian residual law.

Methodology Sec. 2.2.2: fixing the residual flow g to the identity and the
base law p0 to N(0, 1) recovers the Neural-CET class --

    Y = log T = mu(x) + sigma(x) * eps,   eps ~ N(0, 1)

i.e. a heteroscedastic log-normal AFT. This ablation isolates the
contribution of the spline flow: identical encoder and training protocol,
only the residual law changes.

The API mirrors :class:`~flowsurv.models.flowsurv.FlowSurvAFT` exactly, so
the shared training loop (:func:`flowsurv.models.training.fit`) and all
losses work unchanged. All outputs are exact and evaluated in log-space:

    log f(t|x) = -z^2/2 - log sqrt(2 pi) - log t - log sigma,  z = (log t - mu)/sigma
    log S(t|x) = log_ndtr(-z)
    Q(q|x)     = exp(mu + sigma * sqrt(2) * erfinv(2q - 1))

The encoder is built with a minimal dummy conditional RQS flow so the
spline head exists (keeps checkpoint/encoder plumbing identical); its
parameters are never used -- they are zero-initialized, receive no gradient
(no path to the loss), and AdamW weight decay keeps them at zero.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from .encoder import FlowSurvEncoder
from .rqs_flow import ConditionalRQSFlow

_LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)
_SQRT_2 = math.sqrt(2.0)


class FlowSurvGauss(nn.Module):
    """Identity-flow Gaussian-residual ablation of FlowSurv-AFT (Neural CET).

    Args:
        n_features: number of covariates p.
        hidden: encoder trunk width.
        n_blocks: encoder residual blocks; 0 selects the linear encoder.
        dropout: encoder dropout rate.
    """

    def __init__(
        self,
        n_features: int,
        hidden: int = 128,
        n_blocks: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        # dummy flow: keeps the encoder's spline head in place, never used
        self.flow = ConditionalRQSFlow(bins=2, bound=4.0, n_blocks=1)
        self.encoder = FlowSurvEncoder(
            n_features, hidden=hidden, n_blocks=n_blocks, dropout=dropout, flow=self.flow
        )

    # ------------------------------------------------------------------
    # internal helpers

    def _forward_base(
        self, t: Tensor, x: Tensor, paired: bool | None = None
    ) -> tuple[Tensor, Tensor, Tensor]:
        """(z = (log t - mu)/sigma, log t, sigma) -- identity flow, so z = u.

        - Paired: ``t`` has the same leading shape as ``x.shape[:-1]``; returns
          tensors of the same leading shape.
        - Grid: ``t`` (m,) and ``x`` (n, p) returns (m, n).
        """
        x = torch.as_tensor(x)
        orig_shape = x.shape[:-1]
        p = x.shape[-1]
        x_flat = x.reshape(-1, p)
        mu, sigma, _params = self.encoder(x_flat)  # spline params unused by design
        log_t = t.to(mu.dtype).clamp_min(torch.finfo(mu.dtype).tiny).log()
        if log_t.dim() == 0:
            log_t = log_t.unsqueeze(0)

        if paired is None:  # shape heuristic; internal callers pass it explicitly
            paired = x.dim() != 2 or (log_t.dim() == 1 and log_t.numel() == mu.numel())
        if paired:
            t_flat = log_t.reshape(mu.shape)
            z = (t_flat - mu) / sigma
            return z.reshape(orig_shape), t_flat.reshape(orig_shape), sigma.reshape(orig_shape)

        # grid case
        log_t = log_t.unsqueeze(-1)  # (m, 1)
        mu = mu.unsqueeze(0)
        sigma = sigma.unsqueeze(0)
        z = (log_t - mu) / sigma  # (m, n)
        return z, log_t, sigma

    # ------------------------------------------------------------------
    # exact closed-form outputs (Gaussian analogue of Methodology Sec. 2.2.1)

    def log_density(self, t: Tensor, x: Tensor, paired: bool | None = None) -> Tensor:
        """log f(t|x)."""
        z, log_t, sigma = self._forward_base(t, x, paired)
        return -0.5 * z.pow(2) - _LOG_SQRT_2PI - log_t - sigma.log()

    def density(self, t: Tensor, x: Tensor, paired: bool | None = None) -> Tensor:
        """f(t|x)."""
        return self.log_density(t, x, paired).exp()

    def log_survival(self, t: Tensor, x: Tensor, paired: bool | None = None) -> Tensor:
        """log S(t|x) = log Phi(-z), stable under heavy censoring."""
        z = self._forward_base(t, x, paired)[0]
        return torch.special.log_ndtr(-z)

    def survival(self, t: Tensor, x: Tensor, paired: bool | None = None) -> Tensor:
        """S(t|x)."""
        return self.log_survival(t, x, paired).exp()

    def cdf(self, t: Tensor, x: Tensor, paired: bool | None = None) -> Tensor:
        """F(t|x) = Phi(z)."""
        z = self._forward_base(t, x, paired)[0]
        return torch.special.ndtr(z)

    def log_cdf(self, t: Tensor, x: Tensor, paired: bool | None = None) -> Tensor:
        """log F(t|x) = log Phi(z)."""
        z = self._forward_base(t, x, paired)[0]
        return torch.special.log_ndtr(z)

    def hazard(self, t: Tensor, x: Tensor, paired: bool | None = None) -> Tensor:
        """h(t|x) = f(t|x) / S(t|x)."""
        return (self.log_density(t, x, paired) - self.log_survival(t, x, paired)).exp()

    def quantile(self, q: Tensor | float, x: Tensor) -> Tensor:
        """Q(q|x) = exp(mu + sigma * Phi^{-1}(q)), Phi^{-1}(q) = sqrt(2) erfinv(2q - 1).

        ``q`` is a (Q,) tensor of quantile levels; ``x`` can be (n, p) or a
        batched shape (B1, ..., Bk, p). The leading dimension of ``q`` is
        broadcast across the first batch dimension of ``x``; all remaining
        ``x`` dimensions are profile dimensions. The returned tensor has shape
        (Q, *x.shape[1:-1]) when ``x`` has more than two dimensions, and
        (Q, n) when ``x`` is (n, p).
        """
        x = torch.as_tensor(x)
        p = x.shape[-1]
        x_flat = x.reshape(-1, p)
        mu, sigma, _ = self.encoder(x_flat)
        q = torch.as_tensor(q, dtype=mu.dtype, device=mu.device).flatten()
        L = q.numel()
        B = mu.numel()
        if B % L != 0:
            raise ValueError(f"quantile levels {L} do not divide covariate batch {B}")
        N = B // L
        mu = mu.reshape(L, N)
        sigma = sigma.reshape(L, N)
        z_q = _SQRT_2 * torch.special.erfinv(2.0 * q.clamp(1e-7, 1 - 1e-7) - 1.0)
        z_q = z_q.unsqueeze(1).expand(-1, N)
        out = torch.exp(mu + sigma * z_q)
        return out

    def quantiles(self, q: Tensor | list[float], x: Tensor) -> Tensor:
        """Q(q_k|x_i) for every level and subject: shape ``(len(q), n)``."""
        x = torch.as_tensor(x)
        q = torch.as_tensor(q, dtype=x.dtype, device=x.device).flatten()
        x_tiled = x.unsqueeze(0).expand(q.numel(), *x.shape)
        return self.quantile(q, x_tiled).reshape(q.numel(), x.shape[0])

    def sample(self, x: Tensor, n: int = 1, generator: torch.Generator | None = None) -> Tensor:
        """One-pass samples t = exp(mu + sigma * z), z ~ N(0, 1); shape (n, batch)."""
        mu, sigma, _ = self.encoder(x)
        shape = (n,) + mu.shape
        z = torch.randn(shape, dtype=mu.dtype, device=mu.device, generator=generator)
        return torch.exp(mu.unsqueeze(0) + sigma.unsqueeze(0) * z)

    def predict_risk(self, x: Tensor) -> Tensor:
        """Concordance risk score: negative median lifetime -Q(0.5|x) = -exp(mu)."""
        return -self.quantile(0.5, x).squeeze(0)

    def forward(self, t: Tensor, d: Tensor, x: Tensor) -> Tensor:
        """Per-observation right-censored log-likelihood contribution."""
        log_f = self.log_density(t, x, paired=True)
        log_s = self.log_survival(t, x, paired=True)
        return d * log_f + (1 - d) * log_s
