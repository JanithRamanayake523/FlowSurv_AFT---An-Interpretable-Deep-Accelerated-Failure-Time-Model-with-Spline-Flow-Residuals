"""Gate test 3 (Methodology Sec. 6): inverse audit.

|| g^{-1}(g(u)) - u ||_inf < 1e-6 on the spline domain, for batched
conditional parameters, several bin counts, and stacked blocks.
float64 (the transform itself is exact; the tolerance only bounds float
rounding).
"""

import torch

from flowsurv.models import ConditionalRQSFlow

TOL = 1e-6


def test_inverse_roundtrip_on_domain():
    for bins, n_blocks in [(8, 1), (16, 1), (8, 3)]:
        flow = ConditionalRQSFlow(bins=bins, bound=4.0, n_blocks=n_blocks)
        gen = torch.Generator().manual_seed(bins * 10 + n_blocks)
        params = torch.randn(64, flow.n_params, generator=gen, dtype=torch.float64) * 0.5
        u = torch.linspace(-3.99, 3.99, 257, dtype=torch.float64)
        u_b = u.unsqueeze(1).expand(-1, 64)
        params_b = params.unsqueeze(0).expand(257, -1, -1)
        z, _ = flow.forward(u_b, params_b)
        u_back = flow.inverse(z, params_b)
        err = (u_back - u_b).abs().max()
        assert err < TOL, f"bins={bins} blocks={n_blocks}: roundtrip err {err:.2e}"


def test_inverse_matches_forward_outside_domain_identity_tails():
    flow = ConditionalRQSFlow(bins=8, bound=4.0)
    gen = torch.Generator().manual_seed(0)
    params = torch.randn(16, flow.n_params, generator=gen, dtype=torch.float64) * 0.5
    u = torch.linspace(-12, 12, 481, dtype=torch.float64).unsqueeze(1).expand(-1, 16)
    z, ladj = flow.forward(u, params.unsqueeze(0).expand(481, -1, -1))
    outside = u.abs() > flow.bound
    # identity tails: g(u) = u and log|g'| = 0 outside [-bound, bound]
    assert (z[outside] - u[outside]).abs().max() < TOL
    assert ladj[outside].abs().max() < TOL
