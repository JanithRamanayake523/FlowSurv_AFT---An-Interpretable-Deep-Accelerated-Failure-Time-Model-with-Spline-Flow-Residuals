"""Gate test 1 (Methodology Sec. 6): Weibull recovery.

Fit FlowSurv-AFT on large-n S1 data with the flow frozen near identity ->
recover the AFT location coefficients -beta/k within Monte-Carlo error;
the full model's likelihood must reach the true-model likelihood - eps.

The intercept is not checked: under the misspecified logistic residual law
it absorbs the extreme-value location shift, while the slopes remain
consistent (classical AFT robustness).
"""

import torch

from flowsurv.data.dgp import S1_BETA, S1_K, s1_true_log_density
from flowsurv.models import right_censored_nll

SLOPE_TOL = 0.05  # ~10 Monte-Carlo standard errors at n = 20_000
LL_TOL = 0.05  # nats per observation


def test_weibull_beta_recovery(s1_frozen_fit):
    w = s1_frozen_fit.encoder.mu_head.weight.detach().squeeze(0)
    true_slopes = -torch.tensor(S1_BETA) / S1_K  # AFT location = -beta/k
    err_active = (w[:5] - true_slopes).abs().max()
    err_inactive = w[5:].abs().max()
    assert err_active < SLOPE_TOL, f"active slopes off by {err_active:.4f}: {w[:5]} vs {true_slopes}"
    assert err_inactive < SLOPE_TOL, f"inactive slopes off by {err_inactive:.4f}: {w[5:]}"


def test_weibull_likelihood_reaches_truth(s1_full_fit, s1_large):
    t, x = s1_large
    d = torch.ones_like(t)
    with torch.no_grad():
        model_nll = right_censored_nll(s1_full_fit, t, d, x).item()
        true_nll = -s1_true_log_density(t, x).mean().item()
    assert model_nll <= true_nll + LL_TOL, (
        f"model NLL {model_nll:.4f} exceeds true NLL {true_nll:.4f} by more than {LL_TOL}"
    )
