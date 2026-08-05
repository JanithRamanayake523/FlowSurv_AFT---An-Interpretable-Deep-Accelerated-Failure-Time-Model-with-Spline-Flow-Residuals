"""Gate test 5 (Methodology Sec. 6): quantile/sampling audit.

KS test between 1e5 one-pass samples and the analytic CDF (probability
integral transform: F_hat(T|x) ~ Uniform(0,1)) -- p > 0.01. Also checks the
exact inverse consistency F_hat(Q(q|x)|x) = q. Single fixed profile x so the
samples are i.i.d. from one conditional law; float64.
"""

import torch
from scipy import stats

from flowsurv.models import FlowSurvAFT

N_SAMPLES = 100_000
KS_ALPHA = 0.01


def _model():
    torch.manual_seed(5)
    model = FlowSurvAFT(10, n_blocks=2, hidden=64).double()
    with torch.no_grad():
        for head in (model.encoder.mu_head, model.encoder.sigma_head, model.encoder.spline_head):
            head.weight.normal_(0, 0.3)
            head.bias.normal_(0, 0.3)
    model.eval()
    return model


def test_sampling_matches_analytic_cdf_ks():
    model = _model()
    torch.manual_seed(9)
    x = torch.randn(1, 10, dtype=torch.float64)
    gen = torch.Generator().manual_seed(42)
    with torch.no_grad():
        draws = model.sample(x, n=N_SAMPLES, generator=gen).squeeze(1)
        pit = model.cdf(draws, x.expand(N_SAMPLES, -1))
    ks = stats.kstest(pit.numpy(), "uniform")
    assert ks.pvalue > KS_ALPHA, f"KS p-value {ks.pvalue:.4f} <= {KS_ALPHA}"


def test_quantile_is_exact_inverse_of_cdf():
    model = _model()
    torch.manual_seed(15)
    x = torch.randn(4, 10, dtype=torch.float64)
    q = torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95], dtype=torch.float64)
    with torch.no_grad():
        x_b = x.unsqueeze(0).expand(q.numel(), -1, -1)  # (quantiles, profiles, p)
        t_q = model.quantile(q.unsqueeze(1), x_b)
        f_q = model.cdf(t_q, x_b)
    err = (f_q - q.unsqueeze(1)).abs().max()
    assert err < 1e-6, f"F(Q(q)) off by {err:.2e}"
