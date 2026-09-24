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
        conditional_flow: False gives the strict-AFT ablation: the spline
            parameters and sigma are global (independent of x), so the
            residual law is one for all subjects and exp(mu(a) - mu(b)) is
            exactly a time ratio at every quantile.
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
        conditional_flow: bool = True,
    ) -> None:
        super().__init__()
        self.flow = ConditionalRQSFlow(bins=bins, bound=bound, n_blocks=n_spline_blocks)
        self.encoder = FlowSurvEncoder(
            n_features,
            hidden=hidden,
            n_blocks=n_blocks,
            dropout=dropout,
            flow=self.flow,
            conditional_flow=conditional_flow,
        )

    # ------------------------------------------------------------------
    # internal helpers

    def _forward_base(
        self, t: Tensor, x: Tensor, paired: bool | None = None
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """(z = g(u(t)), log|g'(u(t))|, log t, sigma) for times ``t`` given ``x``.

        - Paired: ``t`` has the same leading shape as ``x.shape[:-1]`` (e.g.
          ``t`` (n,) and ``x`` (n, p), or ``t`` (Q, n) and ``x`` (Q, n, p));
          returns tensors of the same leading shape.
        - Grid: ``t`` (m,) and ``x`` (n, p) returns (m, n) matrices.

        ``paired`` makes the mode explicit. ``None`` keeps the shape heuristic
        (batched ``x``, or ``t`` with as many elements as subjects, is paired),
        which is ambiguous when a grid has exactly as many points as there are
        subjects; every internal caller therefore passes it explicitly.
        """
        x = torch.as_tensor(x)
        orig_shape = x.shape[:-1]
        p = x.shape[-1]
        x_flat = x.reshape(-1, p)
        mu, sigma, params = self.encoder(x_flat)
        log_t = t.to(mu.dtype).clamp_min(torch.finfo(mu.dtype).tiny).log()
        if log_t.dim() == 0:
            log_t = log_t.unsqueeze(0)

        if paired is None:
            paired = x.dim() != 2 or (log_t.dim() == 1 and log_t.numel() == mu.numel())
        if paired:
            t_flat = log_t.reshape(mu.shape)
            u = (t_flat - mu) / sigma
            z, ladj = self.flow.forward(u, params)
            return z.reshape(orig_shape), ladj.reshape(orig_shape), t_flat.reshape(orig_shape), sigma.reshape(orig_shape)

        # grid case: t (m,) x subjects (n, p) -> (m, n)
        log_t = log_t.unsqueeze(-1)  # (m, 1)
        mu = mu.unsqueeze(0)
        sigma = sigma.unsqueeze(0)
        u = (log_t - mu) / sigma  # (m, n)
        # expand per-subject spline parameters to match the (m, n) grid
        if params.dim() == 2:
            params = params.unsqueeze(0).expand(u.shape[0], -1, -1)
        z, ladj = self.flow.forward(u, params)
        return z, ladj, log_t, sigma

    @staticmethod
    def _log_f0(z: Tensor) -> Tensor:
        """Standard-logistic log-density, stable on the whole real line."""
        return F.logsigmoid(z) + F.logsigmoid(-z)

    # ------------------------------------------------------------------
    # exact closed-form outputs (Methodology Sec. 2.2.1)

    def log_density(self, t: Tensor, x: Tensor, paired: bool | None = None) -> Tensor:
        """log f(t|x)."""
        z, ladj, log_t, sigma = self._forward_base(t, x, paired)
        return self._log_f0(z) + ladj - log_t - sigma.log()

    def density(self, t: Tensor, x: Tensor, paired: bool | None = None) -> Tensor:
        """f(t|x)."""
        return self.log_density(t, x, paired).exp()

    def log_survival(self, t: Tensor, x: Tensor, paired: bool | None = None) -> Tensor:
        """log S(t|x) = log sigmoid(-g(u(t))), stable under heavy censoring."""
        z = self._forward_base(t, x, paired)[0]
        return F.logsigmoid(-z)

    def survival(self, t: Tensor, x: Tensor, paired: bool | None = None) -> Tensor:
        """S(t|x)."""
        return self.log_survival(t, x, paired).exp()

    def cdf(self, t: Tensor, x: Tensor, paired: bool | None = None) -> Tensor:
        """F(t|x) = sigmoid(g(u(t)))."""
        z = self._forward_base(t, x, paired)[0]
        return torch.sigmoid(z)

    def log_cdf(self, t: Tensor, x: Tensor, paired: bool | None = None) -> Tensor:
        """log F(t|x) = log sigmoid(g(u(t)))."""
        z = self._forward_base(t, x, paired)[0]
        return F.logsigmoid(z)

    def hazard(self, t: Tensor, x: Tensor, paired: bool | None = None) -> Tensor:
        """h(t|x) = f(t|x) / S(t|x)."""
        return (self.log_density(t, x, paired) - self.log_survival(t, x, paired)).exp()

    def quantile(self, q: Tensor | float, x: Tensor) -> Tensor:
        """Q(q|x) = exp(mu + sigma * g^{-1}(logit(q))).

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
        mu, sigma, params = self.encoder(x_flat)
        q = torch.as_tensor(q, dtype=mu.dtype, device=mu.device).flatten()
        L = q.numel()
        B = mu.numel()
        if B % L != 0:
            raise ValueError(f"quantile levels {L} do not divide covariate batch {B}")
        N = B // L
        mu = mu.reshape(L, N)
        sigma = sigma.reshape(L, N)
        params = params.reshape(L, N, -1)
        z_q = torch.logit(q.clamp(1e-6, 1 - 1e-6)).unsqueeze(1).expand(-1, N)
        eps = self.flow.inverse(z_q, params)
        out = torch.exp(mu + sigma * eps)
        return out

    def quantiles(self, q: Tensor | list[float], x: Tensor) -> Tensor:
        """Q(q_k|x_i) for every level and subject: shape ``(len(q), n)``.

        Convenience over :meth:`quantile`, whose batched-``x`` convention
        needs the covariates tiled per level; this tiles them for you.
        """
        x = torch.as_tensor(x)
        q = torch.as_tensor(q, dtype=x.dtype, device=x.device).flatten()
        x_tiled = x.unsqueeze(0).expand(q.numel(), *x.shape)
        return self.quantile(q, x_tiled).reshape(q.numel(), x.shape[0])

    def spline_derivative_penalty(self, x: Tensor) -> Tensor:
        """Mean over ``x`` of the squared interior-derivative parameters.

        The pre-registered "L2 penalty on spline derivatives" (prereg Sec. 4):
        zuko derivatives are softplus(raw + shift) with raw = 0 the identity
        slope, so shrinking the raw derivative parameters shrinks the spline
        toward the identity (smooth, no tail wiggle under heavy censoring).
        Returns 0 for flows without RQS parameters (e.g. the CNF baseline).
        """
        x = torch.as_tensor(x)
        if not isinstance(self.flow, ConditionalRQSFlow):
            return torch.zeros((), device=x.device)
        _mu, _sigma, params = self.encoder(x)
        bins = self.flow.bins
        raw_d = [c[..., 2 * bins :] for c in params.split(self.flow.params_per_block, dim=-1)]
        return sum(r.pow(2).sum(dim=-1) for r in raw_d).mean()

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
        return -self.quantile(0.5, x).squeeze(0)

    def location(self, x: Tensor) -> Tensor:
        """mu(x), the AFT acceleration surface (Methodology Sec. 2.2, Sec. 5).

        exp(mu(x_a) - mu(x_b)) is the time-ratio interpretation used by the
        interpretability analysis; this is the shared primitive both that
        analysis and any diagnostic code should read mu(x) through.
        """
        x = torch.as_tensor(x)
        mu, _sigma, _params = self.encoder(x.reshape(-1, x.shape[-1]))
        return mu.reshape(x.shape[:-1])

    def forward(self, t: Tensor, d: Tensor, x: Tensor) -> Tensor:
        """Per-observation right-censored log-likelihood contribution."""
        log_f = self.log_density(t, x, paired=True)
        log_s = self.log_survival(t, x, paired=True)
        return d * log_f + (1 - d) * log_s
