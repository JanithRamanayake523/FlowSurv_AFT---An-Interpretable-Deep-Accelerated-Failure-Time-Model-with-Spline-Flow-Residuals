"""AussetCNF: FlowSurv-AFT's AFT skeleton with the residual flow swapped for
a continuous normalizing flow (Ausset et al. 2021 style competitor).

Same ``log T = mu(x) + sigma(x) * eps`` decomposition, same encoder, same
exact outputs (:meth:`~flowsurv.models.flowsurv.FlowSurvAFT.density`,
``survival``, ``hazard``, ``quantile``, ``sample``) -- those are all generic
in the flow object (only ``flow.forward``/``flow.inverse``/``flow.n_params``
are used). Only the residual law's transform changes: an ODE-integrated
:class:`~flowsurv.models.cnf_flow.ConditionalCNFFlow` instead of the
analytic :class:`~flowsurv.models.rqs_flow.ConditionalRQSFlow`. This keeps
the empirical comparison to exactly the one axis the paper's differentiator
claims (bidirectional exactness / no solver) rather than confounding it with
unrelated architecture differences.
"""

from __future__ import annotations

from torch import nn

from .cnf_flow import ConditionalCNFFlow
from .encoder import FlowSurvEncoder
from .flowsurv import FlowSurvAFT


class AussetCNFModel(FlowSurvAFT):
    """FlowSurv-AFT's AFT skeleton with a continuous-flow residual law."""

    def __init__(
        self,
        n_features: int,
        hidden: int = 128,
        n_blocks: int = 2,
        dropout: float = 0.1,
        cond_dim: int = 16,
        cnf_hidden: int = 32,
        n_steps: int = 20,
    ) -> None:
        nn.Module.__init__(self)  # bypass FlowSurvAFT.__init__: different flow
        self.flow = ConditionalCNFFlow(cond_dim=cond_dim, hidden=cnf_hidden, n_steps=n_steps)
        self.encoder = FlowSurvEncoder(
            n_features, hidden=hidden, n_blocks=n_blocks, dropout=dropout, flow=self.flow
        )
