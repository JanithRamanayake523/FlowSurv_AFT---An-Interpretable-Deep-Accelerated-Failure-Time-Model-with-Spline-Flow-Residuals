"""Gate test 2 (Methodology Sec. 6): change-of-variables audit.

For random covariate profiles x (and randomly perturbed heads, so the flow
is non-identity), the exact density must satisfy:
  - integral of f_hat(t|x) over t in (0, inf) equals 1 within 1e-4,
  - S_hat(t|x) is monotone non-increasing in t,
  - S_hat(0|x) ~ 1.

Integration uses the change-of-variables form directly in the standardized
residual u: f(t) dt = f0(g(u)) |g'(u)| du. A fine uniform grid over [-30, 30]
(beyond which the logistic tails carry < 1e-13 mass) exercises arbitrarily
narrow/peaky transformed densities. Exercises the MLP encoder. All in float64,
models in eval mode (no dropout).
"""

import torch

from flowsurv.models import FlowSurvAFT

N_GRID = 100_001
INT_TOL = 1e-4


def _perturbed_model(seed: int) -> FlowSurvAFT:
    torch.manual_seed(seed)
    model = FlowSurvAFT(10, n_blocks=2, hidden=64, n_spline_blocks=2).double()
    with torch.no_grad():
        for head in (model.encoder.mu_head, model.encoder.sigma_head, model.encoder.spline_head):
            head.weight.normal_(0, 0.1)
            head.bias.normal_(0, 0.1)
    model.eval()
    return model


def test_density_integrates_to_one():
    model = _perturbed_model(seed=7)
    torch.manual_seed(11)
    x = torch.randn(8, 10, dtype=torch.float64)
    u = torch.linspace(-30, 30, N_GRID, dtype=torch.float64)
    with torch.no_grad():
        mu, sigma, params = model.encoder(x)
        masses = []
        for j in range(x.shape[0]):
            # integrate in the standardized residual u, where dt = sigma * t du
            # f(t) dt = f0(g(u)) * |g'(u)| du
            z, ladj = model.flow.forward(
                u, params[j].unsqueeze(0).expand(u.numel(), -1)
            )
            integrand = torch.exp(model._log_f0(z) + ladj)
            masses.append(torch.trapezoid(integrand, u).item())
    err = max(abs(m - 1.0) for m in masses)
    assert err < INT_TOL, f"density mass off by {err:.2e}: {masses}"


def test_survival_monotone_and_starts_at_one():
    model = _perturbed_model(seed=8)
    torch.manual_seed(13)
    x = torch.randn(8, 10, dtype=torch.float64)
    t = torch.logspace(-6, 6, 1001, dtype=torch.float64)
    with torch.no_grad():
        s = model.survival(t.unsqueeze(1), x)
        s0 = model.survival(torch.full((8,), torch.exp(torch.tensor(-200.0, dtype=torch.float64))), x)
    increments = (s[1:] - s[:-1]).max()
    assert increments <= 1e-9, f"S not monotone non-increasing: max increment {increments:.2e}"
    assert (s0 - 1).abs().max() < 1e-6, f"S(0) != 1: {s0}"
