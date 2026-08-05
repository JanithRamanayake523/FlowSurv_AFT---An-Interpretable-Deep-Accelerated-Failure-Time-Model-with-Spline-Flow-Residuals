"""Shared residual-MLP encoder with (mu, raw sigma, spline-parameter) heads.

Implements Methodology Sec. 2.2: a residual MLP (GELU, LayerNorm, dropout)
maps covariates x to an embedding, and three linear heads produce the AFT
location mu(x), the raw scale (softplus -> sigma(x) > 0), and the
unconstrained spline parameters of the conditional residual flow.

All heads are zero-initialized, so at initialization mu = 0, sigma =
softplus(0) and every spline block is the identity: the model starts as a
log-logistic AFT (the warm start of Methodology Sec. 2.2.2, nesting the
classical model inside the deep one).

``n_blocks=0`` gives a linear encoder (mu, log-sigma and spline parameters
linear in x); this is the configuration used by the Weibull-recovery gate
test, where the mu-head weight is directly comparable to the true AFT
coefficients.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .rqs_flow import ConditionalRQSFlow


class ResidualBlock(nn.Module):
    """Pre-LayerNorm residual block: ``x + Dropout(W2 GELU(W1 LN(x)))``."""

    def __init__(self, dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.drop(self.fc2(self.act(self.fc1(self.norm(x)))))


def _zero_linear(in_features: int, out_features: int) -> nn.Linear:
    """Linear layer with zero weight and bias (identity/warm-start init)."""
    layer = nn.Linear(in_features, out_features)
    nn.init.zeros_(layer.weight)
    nn.init.zeros_(layer.bias)
    return layer


class FlowSurvEncoder(nn.Module):
    """Covariates -> (mu, sigma, spline params).

    Args:
        n_features: number of covariates p.
        hidden: hidden width of the residual MLP trunk.
        n_blocks: number of residual blocks; 0 selects the linear encoder.
        dropout: dropout rate inside residual blocks.
        flow: the conditional RQS flow the spline head parameterizes.
    """

    def __init__(
        self,
        n_features: int,
        hidden: int = 128,
        n_blocks: int = 2,
        dropout: float = 0.1,
        flow: ConditionalRQSFlow | None = None,
    ) -> None:
        super().__init__()
        self.flow = flow if flow is not None else ConditionalRQSFlow()

        if n_blocks > 0:
            trunk = [nn.Linear(n_features, hidden)]
            trunk += [ResidualBlock(hidden, dropout) for _ in range(n_blocks)]
            trunk.append(nn.LayerNorm(hidden))
            out_dim = hidden
        else:  # linear encoder: heads act directly on x
            trunk = [nn.Identity()]
            out_dim = n_features
        self.trunk = nn.Sequential(*trunk)

        self.mu_head = _zero_linear(out_dim, 1)
        self.sigma_head = _zero_linear(out_dim, 1)
        self.spline_head = _zero_linear(out_dim, self.flow.n_params)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Return ``(mu, sigma, spline_params)``, each row one observation."""
        h = self.trunk(x)
        mu = self.mu_head(h).squeeze(-1)
        sigma = nn.functional.softplus(self.sigma_head(h).squeeze(-1))
        params = self.spline_head(h)
        return mu, sigma, params
