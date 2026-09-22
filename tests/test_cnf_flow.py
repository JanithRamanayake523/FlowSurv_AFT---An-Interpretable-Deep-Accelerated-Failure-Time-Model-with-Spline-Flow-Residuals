"""ConditionalCNFFlow: RK4-integrated 1-D continuous normalizing flow
(Ausset-CNF baseline's residual transform).

Checks the two properties that must hold for it to be a valid change-of-
variables flow usable in an exact-likelihood NLL: forward/inverse are (near)
exact inverses of each other, and the accumulated log-det matches an
independent ground truth (autograd's own dz/du through the whole forward
call, which does not go through the flow's internal RK4 log-det bookkeeping
at all).
"""

from __future__ import annotations

import torch

from flowsurv.models.cnf_flow import ConditionalCNFFlow


def _randomized_flow(seed=0, cond_dim=4, hidden=16, n_steps=20):
    torch.manual_seed(seed)
    flow = ConditionalCNFFlow(cond_dim=cond_dim, hidden=hidden, n_steps=n_steps)
    with torch.no_grad():
        for p in flow.net.parameters():
            p.add_(0.3 * torch.randn_like(p))
    return flow


def test_n_params_matches_cond_dim():
    flow = ConditionalCNFFlow(cond_dim=7)
    assert flow.n_params == 7


def test_zero_init_is_near_identity():
    """At init (final layer zeroed) the velocity is ~0, so forward/inverse
    are both close to the identity map -- the CNF analogue of the RQS flow's
    exact identity at zero-initialized spline parameters."""
    flow = ConditionalCNFFlow(cond_dim=4, n_steps=10)
    u = torch.randn(20)
    h = torch.randn(20, 4)
    z, ladj = flow.forward(u, h)
    torch.testing.assert_close(z, u, atol=1e-6, rtol=0)
    torch.testing.assert_close(ladj, torch.zeros_like(ladj), atol=1e-6, rtol=0)


def test_inverse_is_roundtrip_of_forward():
    flow = _randomized_flow()
    u = torch.randn(30) * 0.7
    h = torch.randn(30, 4)
    z, _ = flow.forward(u, h)
    u_rt = flow.inverse(z, h)
    torch.testing.assert_close(u_rt, u, atol=2e-5, rtol=1e-4)


def test_forward_is_roundtrip_of_inverse():
    flow = _randomized_flow()
    z = torch.randn(30) * 0.7
    h = torch.randn(30, 4)
    u = flow.inverse(z, h)
    z_rt, _ = flow.forward(u, h)
    torch.testing.assert_close(z_rt, z, atol=2e-5, rtol=1e-4)


def test_log_det_matches_autograd_ground_truth():
    """ladj from the integrated divergence must equal log|dz/du| computed by
    plain autograd through the whole forward() call -- a check that does not
    depend on the internal RK4 log-det bookkeeping being self-consistent."""
    flow = _randomized_flow()
    n = 10
    u = torch.randn(n).requires_grad_(True)
    h = torch.randn(n, 4)
    z, ladj = flow.forward(u, h)
    (jac,) = torch.autograd.grad(z.sum(), u)  # dz_i/du_i (elementwise map across the batch)
    ladj_true = jac.abs().log()
    torch.testing.assert_close(ladj.detach(), ladj_true, atol=1e-4, rtol=1e-3)


def test_forward_works_under_outer_no_grad():
    """predict_* wrappers call this under torch.no_grad(); the flow must
    still compute the correct (exact) divergence internally."""
    flow = _randomized_flow()
    u = torch.randn(10)
    h = torch.randn(10, 4)
    flow.eval()
    with torch.no_grad():
        z, ladj = flow.forward(u, h)
    assert torch.isfinite(z).all()
    assert torch.isfinite(ladj).all()


def test_divergence_term_is_differentiable_in_training_mode():
    """The log-det term must backpropagate to the velocity net's parameters
    during training (right-censored NLL includes ladj directly) -- a
    regression guard on create_graph=self.training in _velocity_and_div."""
    flow = _randomized_flow()
    flow.train()
    u = torch.randn(10)
    h = torch.randn(10, 4)
    _, ladj = flow.forward(u, h)
    loss = ladj.pow(2).mean()
    loss.backward()
    grad_norm = next(flow.net.parameters()).grad.norm().item()
    assert grad_norm > 0.0


def test_grid_broadcast_shape():
    """FlowSurvAFT's grid case expands params to (m, n, cond_dim); the flow
    must handle that leading-shape broadcast like ConditionalRQSFlow does."""
    flow = _randomized_flow(cond_dim=3, n_steps=5)
    m, n = 4, 6
    u = torch.randn(m, n)
    h = torch.randn(n, 3).unsqueeze(0).expand(m, n, 3)
    z, ladj = flow.forward(u, h)
    assert z.shape == (m, n)
    assert ladj.shape == (m, n)


def test_invalid_constructor_args_raise():
    try:
        ConditionalCNFFlow(cond_dim=0)
        assert False
    except ValueError:
        pass
    try:
        ConditionalCNFFlow(cond_dim=4, n_steps=0)
        assert False
    except ValueError:
        pass
