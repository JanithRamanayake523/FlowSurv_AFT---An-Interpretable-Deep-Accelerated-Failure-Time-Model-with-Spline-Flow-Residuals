"""Shared fixtures for the Phase 1 gate tests (Methodology Sec. 6).

The two large-n S1 fits (flow frozen at identity / full flow) are shared by
gate test 1 (Weibull recovery) and gate test 6 (D-calibration sanity).
Linear encoder (n_blocks=0): the mu-head weight is then directly comparable
to the true AFT coefficients, and the truth is nested in the model class.
Both fits run on CPU so they are reproducible in CI.
"""

import pytest
import torch

from flowsurv.data.dgp import simulate_s1
from flowsurv.models import FlowSurvAFT, TrainConfig, fit

N_LARGE = 20_000

FIT_CONFIG = TrainConfig(batch_size=1024, max_epochs=150, patience=20, seed=0, device="cpu")


@pytest.fixture(scope="session")
def s1_large():
    """Large uncensored S1 sample: (t, x), d == 1 everywhere."""
    t, x = simulate_s1(N_LARGE, seed=2026)
    return t, x


@pytest.fixture(scope="session")
def s1_frozen_fit(s1_large):
    """FlowSurv-AFT with the flow frozen at identity (log-logistic AFT)."""
    t, x = s1_large
    model = FlowSurvAFT(10, n_blocks=0)
    for p in model.encoder.spline_head.parameters():
        p.requires_grad_(False)
    fit(model, t, torch.ones_like(t), x, FIT_CONFIG)
    return model


@pytest.fixture(scope="session")
def s1_full_fit(s1_large):
    """Full FlowSurv-AFT (linear encoder + free spline flow)."""
    t, x = s1_large
    model = FlowSurvAFT(10, n_blocks=0)
    fit(model, t, torch.ones_like(t), x, FIT_CONFIG)
    return model
