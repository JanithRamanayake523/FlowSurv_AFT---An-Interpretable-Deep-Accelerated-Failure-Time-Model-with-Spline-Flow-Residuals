"""Ausset et al. (2021)-style continuous normalizing flow baseline.

Native PyTorch reimplementation (Methodology Sec. 3.2): FlowSurv-AFT's AFT
decomposition with the residual law transformed by a continuous normalizing
flow (:class:`~flowsurv.models.cnf_flow.ConditionalCNFFlow`, RK4-integrated,
no ``torchdiffeq`` dependency) instead of the exact RQS spline flow. This is
the empirical stand-in for Ausset et al. 2021 (IEEE DSAA, arXiv:2107.12825):
a numerical solver is required for every forward/inverse evaluation, which is
exactly the property FlowSurv-AFT's bidirectional-exactness claim (AGENTS.md
differentiator 1) contrasts against.

Reuses ``_FlowSurvWrapper`` (``flowsurv_wrappers.py``) unchanged: predict_*
already route through ``torch.no_grad()``, and ``ConditionalCNFFlow``
re-enables grad locally where it needs to (its exact-in-1-D divergence
computation), so it works correctly under that outer no-grad context.
"""

from __future__ import annotations

from ..models import AussetCNFModel
from .flowsurv_wrappers import _FlowSurvWrapper


class AussetCNF(_FlowSurvWrapper):
    """FlowSurv-AFT skeleton with a continuous-flow (ODE) residual law."""

    name: str = "ausset_cnf"
    _cls = AussetCNFModel

    def __init__(self) -> None:
        super().__init__()
        self.tuning_space = {
            "hidden": [64, 128],
            "n_blocks": [1, 2, 3],
            "dropout": [0.1, 0.2],
            "cond_dim": [8, 16],
            "cnf_hidden": [16, 32],
            "n_steps": [10, 20],
        }

    def _make_model(self, n_features: int, **hyper) -> AussetCNFModel:
        return AussetCNFModel(
            n_features,
            hidden=hyper.get("hidden", 128),
            n_blocks=hyper.get("n_blocks", 2),
            dropout=hyper.get("dropout", 0.1),
            cond_dim=hyper.get("cond_dim", 16),
            cnf_hidden=hyper.get("cnf_hidden", 32),
            n_steps=hyper.get("n_steps", 20),
        )
