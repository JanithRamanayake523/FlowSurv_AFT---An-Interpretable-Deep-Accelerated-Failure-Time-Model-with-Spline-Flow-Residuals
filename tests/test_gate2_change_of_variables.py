"""Gate test 2 (Methodology Sec. 6): change-of-variables audit.

For random covariate profiles x (and randomly perturbed heads, so the flow
is far from identity), the exact density must satisfy:
  - integral of f_hat(t|x) over t in (0, inf) equals 1 within 1e-4,
  - S_hat(t|x) is monotone non-increasing in t,
  - S_hat(0|x) ~ 1.

Integration is a trapezoid rule in t, on a per-profile t-grid adapted to the
profile's own location-scale: t = exp(mu + sigma * u) for u on a fine grid
over [-30, 30] (beyond which the logistic tails carry < 1e-13 mass). This
resolves arbitrarily narrow/peaky densities while still auditing the full
t-space formula f(t|x) = f0(g(u)) |g'(u)| / (t sigma). Exercises the MLP
encoder. All in float64, models in eval mode (no dropout).
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
            head.weight.normal_(0, 0.3)
            head.bias.normal_(0, 0.3)
    model.eval()
    return model


def test_density_integrates_to_one():
    model = _perturbed_model(seed=7)
    torch.manual_seed(11)
    x = torch.randn(8, 10, dtype=torch.float64)
    u = torch.linspace(-30, 30, N_GRID, dtype=torch.float64)
    with torch.no_grad():
        mu, sigma, _ = model.encoder(x)
        masses = []
        for j in range(x.shape[0]):
            t = torch.exp(mu[j] + sigma[j] * u)  # per-profile adapted t-grid
            f = model.density(t, x[j].expand(u.numel(), -1))
            masses.append(torch.trapezoid(f, t).item())
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
