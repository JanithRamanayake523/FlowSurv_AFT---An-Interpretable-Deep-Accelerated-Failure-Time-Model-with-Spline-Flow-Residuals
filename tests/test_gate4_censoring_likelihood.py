"""Gate test 4 (Methodology Sec. 6): censoring-likelihood gradient audit.

On data with delta = 0 everywhere the likelihood reduces to the survival
term: the loss must equal -mean(log S) and its gradient must equal the
gradient of -mean(log S) only (finite-difference verified). float64 for
the finite-difference check.
"""

import torch

from flowsurv.models import FlowSurvAFT, right_censored_nll

FD_EPS = 1e-5
FD_TOL = 1e-4


def _model_and_batch():
    torch.manual_seed(3)
    model = FlowSurvAFT(10, n_blocks=0).double()
    with torch.no_grad():
        for head in (model.encoder.mu_head, model.encoder.sigma_head, model.encoder.spline_head):
            head.weight.normal_(0, 0.3)
            head.bias.normal_(0, 0.3)
    x = torch.randn(64, 10, dtype=torch.float64)
    t = torch.rand(64, dtype=torch.float64) * 5 + 0.05
    d = torch.zeros(64, dtype=torch.float64)  # all censored
    model.eval()
    return model, t, d, x


def test_all_censored_loss_is_survival_term_only():
    model, t, d, x = _model_and_batch()
    loss = right_censored_nll(model, t, d, x)
    pure = -model.log_survival(t, x).mean()
    assert torch.allclose(loss, pure, atol=1e-12), f"{loss.item()} vs {pure.item()}"


def test_all_censored_gradient_matches_survival_score():
    model, t, d, x = _model_and_batch()
    loss = right_censored_nll(model, t, d, x)
    loss.backward()

    # finite-difference check of autograd on a slice of mu-head weights
    w = model.encoder.mu_head.weight
    idx = [(0, j) for j in range(4)]
    for i, j in idx:
        orig = w[i, j].item()
        with torch.no_grad():
            w[i, j] = orig + FD_EPS
            plus = right_censored_nll(model, t, d, x).item()
            w[i, j] = orig - FD_EPS
            minus = right_censored_nll(model, t, d, x).item()
            w[i, j] = orig
        fd = (plus - minus) / (2 * FD_EPS)
        ad = w.grad[i, j].item()
        assert abs(fd - ad) < FD_TOL, f"param ({i},{j}): autograd {ad:.6f} vs fd {fd:.6f}"

    # and the autograd gradient equals the gradient of the pure survival term
    model.zero_grad()
    pure = -model.log_survival(t, x).mean()
    pure.backward()
    # recompute censored-loss gradient on a fresh pass and compare
    model2, t2, d2, x2 = _model_and_batch()
    right_censored_nll(model2, t2, d2, x2).backward()
    for p1, p2 in zip(model.parameters(), model2.parameters()):
        assert torch.allclose(p1.grad, p2.grad, atol=1e-12)
