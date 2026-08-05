"""Conditional monotone rational-quadratic spline (RQS) transform.

Thin wrapper around :class:`zuko.transforms.MonotonicRQSTransform` implementing
the residual flow g of Methodology Sec. 2.2: a strictly monotone map

    z = g(u ; h(x))        (forward, residual -> base)
    u = g^{-1}(z ; h(x))   (inverse, base -> residual)

with analytic forward, inverse, and log|det| in both directions (the
bidirectional-exactness property). Zero parameters yield the exact identity
map, which gives the log-logistic warm start of Methodology Sec. 2.2.2.

Confirmed zuko semantics match Durkan et al. (2019): `bins` bins on the
bounded domain [-bound, bound], identity tails outside the bound (boundary
derivatives fixed to 1), softmax-normalized widths/heights, exponentiated
interior derivatives.
"""

from __future__ import annotations

import torch
from torch import Tensor
from zuko.transforms import MonotonicRQSTransform


class ConditionalRQSFlow:
    """Per-observation composition of ``n_blocks`` monotone RQS transforms.

    The conditioner supplies, for each observation, the unconstrained spline
    parameters of every block. The composed map is

        g = g_K o ... o g_1   (forward),   g^{-1} = g_1^{-1} o ... o g_K^{-1}.

    Stateless: all conditioning enters through the ``params`` argument of
    :meth:`forward` / :meth:`inverse`, so the same object serves any batch.
    """

    def __init__(self, bins: int = 8, bound: float = 4.0, n_blocks: int = 1) -> None:
        if bins < 2:
            raise ValueError("bins must be >= 2")
        self.bins = bins
        self.bound = float(bound)
        self.n_blocks = n_blocks
        #: unconstrained parameters per block: widths + heights + interior derivatives
        self.params_per_block = 3 * bins - 1
        self.n_params = n_blocks * self.params_per_block

    def _transforms(self, params: Tensor) -> list[MonotonicRQSTransform]:
        """Build one transform per block from a ``(*, n_params)`` tensor."""
        if params.shape[-1] != self.n_params:
            raise ValueError(
                f"expected {self.n_params} spline parameters, got {params.shape[-1]}"
            )
        transforms = []
        for chunk in params.split(self.params_per_block, dim=-1):
            w, h, d = chunk.split((self.bins, self.bins, self.bins - 1), dim=-1)
            transforms.append(MonotonicRQSTransform(w, h, d, bound=self.bound))
        return transforms

    def forward(self, u: Tensor, params: Tensor) -> tuple[Tensor, Tensor]:
        """Map residual ``u`` to base ``z = g(u)``; returns ``(z, log|g'(u)|)``."""
        z, ladj = u, torch.zeros_like(u)
        for t in self._transforms(params):
            z, ladj_t = t.call_and_ladj(z)
            ladj = ladj + ladj_t
        return z, ladj

    def inverse(self, z: Tensor, params: Tensor) -> Tensor:
        """Map base ``z`` to residual ``u = g^{-1}(z)`` (analytic)."""
        u = z
        for t in reversed(self._transforms(params)):
            u = t.inv(u)
        return u
