"""Simulation data-generating processes (Methodology Sec. 3.1).

Phase 2 scope: S1-S6 generators (inversion method from the stated conditional
hazards) plus analytic ground-truth functions used by the recovery metrics
(Methodology Sec. 3.3) and the gate tests.

Broadcasting convention for all ``sX_true_*`` functions: ``x`` is always
(n, p). ``t`` of shape (n,) is evaluated per subject, returning (n,); ``t``
of shape (m, 1) is treated as a time grid evaluated against every subject,
returning (m, n). The ``"hazard"`` callable attached by
``cells.generate_dataset`` wraps this so a 1-D grid (m,) yields (m, n).
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

#: active coefficients on the hazard (Methodology Sec. 3.1)
S1_BETA = (0.8, -0.6, 0.5, -0.4, 0.3)
S1_K = 1.5
S1_LAMBDA = 1.0

#: S2 bathtub baseline-hazard constants: h0(t) = A * exp(-B t) + C t
S2_A = 0.8
S2_B = 1.5
S2_C = 0.15

#: S3 lognormal-mixture constants: weight, (mu, sigma) per component
S3_W1 = 0.6
S3_MU1 = 0.5
S3_SIG1 = 0.4
S3_MU2 = 2.5
S3_SIG2 = 0.3

#: S4 crossing-hazard Weibull shapes/scales for group A (x1 > 0) / B (x1 <= 0)
S4_K_A = 0.7
S4_LAM_A = 1.0
S4_K_B = 2.2
S4_LAM_B = 1.3

#: S5 nonlinear-mu Weibull baseline constants (same hazard form as S1)
S5_K = 1.5
S5_LAMBDA = 1.0

#: S6 semi-synthetic Weibull--gamma-frailty constants (see simulate_s6)
S6_K = 1.5
S6_FRAILTY_SHAPE = 4.0  # gamma shape; rate = shape, so E[omega] = 1, Var = 1/4
#: fixed seed for the S6 coefficient draw (drawn once, cached, reproducible)
S6_BETA_SEED = 20260805

_TINY = torch.finfo(torch.float32).tiny


def _make_generator(seed: int | None) -> torch.Generator | None:
    return torch.Generator().manual_seed(seed) if seed is not None else None


def _lin(x: Tensor, beta: tuple[float, ...] | Tensor) -> Tensor:
    """Linear predictor on the first ``len(beta)`` covariates -> shape (n,)."""
    b = torch.as_tensor(beta, dtype=x.dtype, device=x.device)
    return x[:, : len(b)] @ b


def _norm_cdf(z: Tensor) -> Tensor:
    """Standard normal CDF via torch.erf (no scipy dependency)."""
    return 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))


def _norm_pdf(z: Tensor) -> Tensor:
    return torch.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


def _bisect_invert(fn, y: Tensor, iters: int = 60) -> Tensor:
    """Vectorized bisection inverse of a monotone increasing scalar function.

    Solves ``fn(t) = y`` elementwise. The upper bracket is doubled until it
    covers every target, then ``iters`` bisection steps are run. 60 iterations
    give ~1e-18 relative resolution in float64 and saturate float32 precision
    (~35 iterations to reach eps), so 60 is comfortably plenty.
    """
    lo = torch.zeros_like(y)
    hi = torch.ones_like(y)
    # expand the bracket until fn(hi) >= y everywhere
    for _ in range(200):
        need = fn(hi) < y
        if not bool(need.any()):
            break
        hi = torch.where(need, hi * 2.0, hi)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        below = fn(mid) < y
        lo = torch.where(below, mid, lo)
        hi = torch.where(below, hi, mid)
    return 0.5 * (lo + hi)


def simulate_s1(
    n: int,
    p: int = 10,
    beta: tuple[float, ...] = S1_BETA,
    k: float = S1_K,
    lam: float = S1_LAMBDA,
    seed: int | None = None,
) -> tuple[Tensor, Tensor]:
    """S1 Weibull AFT: h(t|x) = (k/lam) (t/lam)^(k-1) exp(beta'x), x ~ N(0, I_p).

    The first ``len(beta)`` covariates are active. Inversion method:
    T = lam * (-log U)^(1/k) * exp(-beta'x / k), U ~ Uniform(0, 1).

    In AFT form: log T = log lam - (beta'x)/k + W/k with W a standard
    minimum-extreme-value variate, so the true AFT location coefficients are
    ``-beta / k`` and the true AFT scale is ``1 / k``.

    Returns ``(t, x)`` -- event times (n,) and covariates (n, p).
    """
    if len(beta) > p:
        raise ValueError("more active coefficients than covariates")
    gen = torch.Generator().manual_seed(seed) if seed is not None else None
    x = torch.randn(n, p, generator=gen)
    u = torch.rand(n, generator=gen).clamp_min(torch.finfo(torch.float32).tiny)
    b = torch.tensor(beta)
    lin = x[:, : len(beta)] @ b
    t = lam * (-u.log()) ** (1.0 / k) * torch.exp(-lin / k)
    return t, x


def s1_true_log_density(t: Tensor, x: Tensor, beta=S1_BETA, k: float = S1_K, lam: float = S1_LAMBDA) -> Tensor:
    """True S1 log-density, for the likelihood comparison of gate test 1."""
    b = torch.tensor(beta, dtype=t.dtype)
    lin = x[:, : len(beta)] @ b
    return (
        torch.log(torch.tensor(k / lam, dtype=t.dtype))
        + (k - 1) * (t.log() - torch.log(torch.tensor(lam, dtype=t.dtype)))
        + lin
        - (t / lam) ** k * lin.exp()
    )


def s1_true_hazard(t: Tensor, x: Tensor, beta=S1_BETA, k: float = S1_K, lam: float = S1_LAMBDA) -> Tensor:
    """True S1 hazard (k/lam)(t/lam)^(k-1) exp(beta'x); see module docstring for shapes."""
    lin = _lin(x, beta)
    return (k / lam) * (t / lam) ** (k - 1.0) * torch.exp(lin)


def s1_true_cumhazard(t: Tensor, x: Tensor, beta=S1_BETA, k: float = S1_K, lam: float = S1_LAMBDA) -> Tensor:
    """True S1 cumulative hazard (t/lam)^k exp(beta'x)."""
    lin = _lin(x, beta)
    return (t / lam) ** k * torch.exp(lin)


def s1_true_cdf(t: Tensor, x: Tensor, beta=S1_BETA, k: float = S1_K, lam: float = S1_LAMBDA) -> Tensor:
    """True S1 conditional CDF 1 - exp(-H(t|x))."""
    return 1.0 - torch.exp(-s1_true_cumhazard(t, x, beta, k, lam))


def s1_true_density(t: Tensor, x: Tensor, beta=S1_BETA, k: float = S1_K, lam: float = S1_LAMBDA) -> Tensor:
    """True S1 conditional density h(t|x) * exp(-H(t|x))."""
    h = s1_true_hazard(t, x, beta, k, lam)
    return h * torch.exp(-s1_true_cumhazard(t, x, beta, k, lam))


# ---------------------------------------------------------------------------
# S2 -- bathtub hazard (Methodology Sec. 3.1)
# ---------------------------------------------------------------------------


def s2_base_cumhazard(t: Tensor) -> Tensor:
    """Baseline cumulative hazard H0(t) = (A/B)(1 - exp(-B t)) + (C/2) t^2."""
    return (S2_A / S2_B) * (1.0 - torch.exp(-S2_B * t)) + 0.5 * S2_C * t * t


def simulate_s2(
    n: int,
    p: int = 10,
    beta: tuple[float, ...] = S1_BETA,
    seed: int | None = None,
) -> tuple[Tensor, Tensor]:
    """S2 bathtub: h(t|x) = (0.8 exp(-1.5 t) + 0.15 t) exp(beta'x).

    No closed-form inverse of the cumulative hazard exists, so event times
    are drawn by inversion with vectorized bisection on the monotone baseline
    cumulative hazard (``_bisect_invert``, 60 iterations): solve
    H0(t) = -log(U) * exp(-beta'x).

    Returns ``(t, x)`` -- event times (n,) and covariates (n, p).
    """
    gen = _make_generator(seed)
    x = torch.randn(n, p, generator=gen)
    u = torch.rand(n, generator=gen).clamp_min(_TINY)
    lin = _lin(x, beta)
    target = -u.log() * torch.exp(-lin)
    t = _bisect_invert(s2_base_cumhazard, target)
    return t, x


def s2_true_hazard(t: Tensor, x: Tensor, beta=S1_BETA) -> Tensor:
    """True S2 hazard (0.8 exp(-1.5 t) + 0.15 t) exp(beta'x)."""
    lin = _lin(x, beta)
    return (S2_A * torch.exp(-S2_B * t) + S2_C * t) * torch.exp(lin)


def s2_true_cumhazard(t: Tensor, x: Tensor, beta=S1_BETA) -> Tensor:
    """True S2 cumulative hazard H0(t) exp(beta'x)."""
    return s2_base_cumhazard(t) * torch.exp(_lin(x, beta))


def s2_true_cdf(t: Tensor, x: Tensor, beta=S1_BETA) -> Tensor:
    """True S2 conditional CDF 1 - exp(-H(t|x))."""
    return 1.0 - torch.exp(-s2_true_cumhazard(t, x, beta))


def s2_true_density(t: Tensor, x: Tensor, beta=S1_BETA) -> Tensor:
    """True S2 conditional density h(t|x) * exp(-H(t|x))."""
    h = s2_true_hazard(t, x, beta)
    return h * torch.exp(-s2_true_cumhazard(t, x, beta))


# ---------------------------------------------------------------------------
# S3 -- multimodal lognormal mixture (Methodology Sec. 3.1)
# ---------------------------------------------------------------------------


def simulate_s3(
    n: int,
    p: int = 10,
    beta: tuple[float, ...] = S1_BETA,
    seed: int | None = None,
) -> tuple[Tensor, Tensor]:
    """S3 multimodal: T|x ~ 0.6 LogNormal(0.5 + beta'x, 0.4) + 0.4 LogNormal(2.5, 0.3).

    The mixture component is sampled with probability 0.6, then the lognormal
    is sampled directly (closed-form inversion of the normal CDF is absorbed
    into randn). Returns ``(t, x)``.
    """
    gen = _make_generator(seed)
    x = torch.randn(n, p, generator=gen)
    lin = _lin(x, beta)
    comp1 = torch.rand(n, generator=gen) < S3_W1
    z = torch.randn(n, generator=gen)
    mu = torch.where(comp1, S3_MU1 + lin, torch.full_like(lin, S3_MU2))
    sig = torch.where(comp1, torch.full_like(lin, S3_SIG1), torch.full_like(lin, S3_SIG2))
    t = torch.exp(mu + sig * z)
    return t, x


def s3_true_cdf(t: Tensor, x: Tensor, beta=S1_BETA) -> Tensor:
    """True S3 conditional CDF (analytic lognormal mixture, torch.erf based)."""
    lin = _lin(x, beta)
    log_t = t.clamp_min(_TINY).log()
    f1 = _norm_cdf((log_t - S3_MU1 - lin) / S3_SIG1)
    f2 = _norm_cdf((log_t - S3_MU2) / S3_SIG2)
    return S3_W1 * f1 + (1.0 - S3_W1) * f2


def s3_true_density(t: Tensor, x: Tensor, beta=S1_BETA) -> Tensor:
    """True S3 conditional density (analytic lognormal mixture)."""
    lin = _lin(x, beta)
    t_safe = t.clamp_min(_TINY)
    log_t = t_safe.log()
    f1 = _norm_pdf((log_t - S3_MU1 - lin) / S3_SIG1) / (S3_SIG1 * t_safe)
    f2 = _norm_pdf((log_t - S3_MU2) / S3_SIG2) / (S3_SIG2 * t_safe)
    return S3_W1 * f1 + (1.0 - S3_W1) * f2


def s3_true_hazard(t: Tensor, x: Tensor, beta=S1_BETA) -> Tensor:
    """True S3 conditional hazard f / (1 - F)."""
    f = s3_true_density(t, x, beta)
    s = 1.0 - s3_true_cdf(t, x, beta)
    return f / s.clamp_min(_TINY)


# ---------------------------------------------------------------------------
# S4 -- crossing hazards (Methodology Sec. 3.1)
# ---------------------------------------------------------------------------


def _s4_group_params(x: Tensor) -> tuple[Tensor, Tensor]:
    """Per-subject Weibull (k, lam): group A if x1 > 0 else group B."""
    group_a = x[:, 0] > 0
    k = torch.where(group_a, torch.full_like(x[:, 0], S4_K_A), torch.full_like(x[:, 0], S4_K_B))
    lam = torch.where(group_a, torch.full_like(x[:, 0], S4_LAM_A), torch.full_like(x[:, 0], S4_LAM_B))
    return k, lam


def simulate_s4(n: int, p: int = 10, seed: int | None = None) -> tuple[Tensor, Tensor]:
    """S4 crossing hazards: group A (x1 > 0) Weibull(k=0.7, lam=1);
    group B (x1 <= 0) Weibull(k=2.2, lam=1.3).

    Per-group closed-form inversion, as in S1: T = lam (-log U)^(1/k).
    Returns ``(t, x)``.
    """
    gen = _make_generator(seed)
    x = torch.randn(n, p, generator=gen)
    u = torch.rand(n, generator=gen).clamp_min(_TINY)
    k, lam = _s4_group_params(x)
    t = lam * (-u.log()) ** (1.0 / k)
    return t, x


def s4_true_hazard(t: Tensor, x: Tensor) -> Tensor:
    """True S4 hazard (k_g/lam_g)(t/lam_g)^(k_g - 1), group g from x1."""
    k, lam = _s4_group_params(x)
    return (k / lam) * (t / lam) ** (k - 1.0)


def s4_true_cumhazard(t: Tensor, x: Tensor) -> Tensor:
    """True S4 cumulative hazard (t/lam_g)^k_g."""
    k, lam = _s4_group_params(x)
    return (t / lam) ** k


def s4_true_cdf(t: Tensor, x: Tensor) -> Tensor:
    """True S4 conditional CDF 1 - exp(-H(t|x))."""
    return 1.0 - torch.exp(-s4_true_cumhazard(t, x))


def s4_true_density(t: Tensor, x: Tensor) -> Tensor:
    """True S4 conditional density h(t|x) * exp(-H(t|x))."""
    return s4_true_hazard(t, x) * torch.exp(-s4_true_cumhazard(t, x))


# ---------------------------------------------------------------------------
# S5 -- nonlinear acceleration surface (Methodology Sec. 3.1)
# ---------------------------------------------------------------------------


def s5_mu(x: Tensor) -> Tensor:
    """True S5 acceleration surface mu(x) = 0.8 sin(2 x1) - 0.6 x2 x3 + 0.5 x4^2."""
    return 0.8 * torch.sin(2.0 * x[:, 0]) - 0.6 * x[:, 1] * x[:, 2] + 0.5 * x[:, 3] ** 2


def simulate_s5(
    n: int,
    p: int = 10,
    k: float = S5_K,
    lam: float = S5_LAMBDA,
    seed: int | None = None,
) -> tuple[Tensor, Tensor]:
    """S5 nonlinear mu: S1 Weibull hazard form with the linear predictor
    replaced by mu(x) = 0.8 sin(2 x1) - 0.6 x2 x3 + 0.5 x4^2.

    Closed-form inversion as in S1: T = lam (-log U)^(1/k) exp(-mu(x)/k).
    Returns ``(t, x)``.
    """
    gen = _make_generator(seed)
    x = torch.randn(n, p, generator=gen)
    u = torch.rand(n, generator=gen).clamp_min(_TINY)
    mu = s5_mu(x)
    t = lam * (-u.log()) ** (1.0 / k) * torch.exp(-mu / k)
    return t, x


def s5_true_hazard(t: Tensor, x: Tensor, k: float = S5_K, lam: float = S5_LAMBDA) -> Tensor:
    """True S5 hazard (k/lam)(t/lam)^(k-1) exp(mu(x))."""
    return (k / lam) * (t / lam) ** (k - 1.0) * torch.exp(s5_mu(x))


def s5_true_cumhazard(t: Tensor, x: Tensor, k: float = S5_K, lam: float = S5_LAMBDA) -> Tensor:
    """True S5 cumulative hazard (t/lam)^k exp(mu(x))."""
    return (t / lam) ** k * torch.exp(s5_mu(x))


def s5_true_cdf(t: Tensor, x: Tensor, k: float = S5_K, lam: float = S5_LAMBDA) -> Tensor:
    """True S5 conditional CDF 1 - exp(-H(t|x))."""
    return 1.0 - torch.exp(-s5_true_cumhazard(t, x, k, lam))


def s5_true_density(t: Tensor, x: Tensor, k: float = S5_K, lam: float = S5_LAMBDA) -> Tensor:
    """True S5 conditional density h(t|x) * exp(-H(t|x))."""
    return s5_true_hazard(t, x, k, lam) * torch.exp(-s5_true_cumhazard(t, x, k, lam))


# ---------------------------------------------------------------------------
# S6 -- semi-synthetic Weibull--gamma-frailty on real covariates (Sec. 3.1)
# ---------------------------------------------------------------------------

_S6_BETA_CACHE: dict[int, Tensor] = {}


def _s6_beta(p: int) -> Tensor:
    """S6 coefficients, drawn once from S6_BETA_SEED and cached.

    beta ~ 0.5 * N(0, I_p): moderate effect sizes on standardized real
    covariates. Fixed seed makes the DGP fully reproducible.
    """
    if p not in _S6_BETA_CACHE:
        gen = torch.Generator().manual_seed(S6_BETA_SEED)
        _S6_BETA_CACHE[p] = 0.5 * torch.randn(p, generator=gen)
    return _S6_BETA_CACHE[p]


def simulate_s6(
    n: int,
    source: str = "support",
    seed: int | None = None,
) -> tuple[Tensor, Tensor]:
    """S6 semi-synthetic: real covariates + Weibull--gamma-frailty times.

    Covariates come from ``flowsurv.data.real`` (lazy import -- pycox is an
    optional dependency), resampled with replacement to size n and
    re-standardized. Event times follow T | x, omega ~ Weibull with
    conditional survival S(t|x,omega) = exp(-omega * t^k * exp(beta'x)),
    k = 1.5, and a gamma frailty omega ~ Gamma(shape=4, rate=4) (E[omega]=1,
    Var=1/4). The frailty is drawn as -log(U1 U2 U3 U4) / 4 (sum of four
    exponentials), so no scipy/distributions dependency is needed.

    Gamma (not lognormal) frailty is a deliberate, deviation-free choice: the
    Laplace transform of the gamma law gives the *exact* closed-form marginal
    survival S(t|x) = (1 + t^k exp(beta'x) / 4)^(-4) (Burr-XII type), which
    the ``s6_true_*`` functions implement. A lognormal frailty has no such
    closed form and would force numerical marginalization of the truth.

    Returns ``(t, x)`` with x the standardized real covariates (n, p_real).
    """
    from .real import REAL_DATASETS  # lazy: avoids a hard pycox dependency

    gen = _make_generator(seed)
    data = REAL_DATASETS[source]()
    x_real = data["x"]
    idx = torch.randint(0, x_real.shape[0], (n,), generator=gen)
    x = x_real[idx]
    x = (x - x.mean(dim=0)) / x.std(dim=0).clamp_min(_TINY)
    lin = x @ _s6_beta(x.shape[1]).to(x.dtype)
    u_f = torch.rand(n, int(S6_FRAILTY_SHAPE), generator=gen).clamp_min(_TINY)
    omega = -u_f.log().sum(dim=1) / S6_FRAILTY_SHAPE
    u = torch.rand(n, generator=gen).clamp_min(_TINY)
    t = (-u.log() / (omega * torch.exp(lin))) ** (1.0 / S6_K)
    return t, x


def s6_true_survival(t: Tensor, x: Tensor, k: float = S6_K) -> Tensor:
    """Frailty-marginal S6 survival (1 + t^k exp(beta'x) / shape)^(-shape).

    ``x`` must be the standardized covariates as returned by ``simulate_s6``.
    """
    lin = x @ _s6_beta(x.shape[1]).to(x.dtype)
    h0 = t**k * torch.exp(lin)
    return (1.0 + h0 / S6_FRAILTY_SHAPE) ** (-S6_FRAILTY_SHAPE)


def s6_true_cdf(t: Tensor, x: Tensor, k: float = S6_K) -> Tensor:
    """Frailty-marginal S6 conditional CDF 1 - S(t|x)."""
    return 1.0 - s6_true_survival(t, x, k)


def s6_true_density(t: Tensor, x: Tensor, k: float = S6_K) -> Tensor:
    """Frailty-marginal S6 density k t^(k-1) exp(beta'x) (1 + H0/shape)^(-(shape+1))."""
    lin = x @ _s6_beta(x.shape[1]).to(x.dtype)
    h0 = t**k * torch.exp(lin)
    return k * t ** (k - 1.0) * torch.exp(lin) * (1.0 + h0 / S6_FRAILTY_SHAPE) ** (-(S6_FRAILTY_SHAPE + 1.0))


def s6_true_hazard(t: Tensor, x: Tensor, k: float = S6_K) -> Tensor:
    """Frailty-marginal S6 hazard k t^(k-1) exp(beta'x) / (1 + H0/shape)."""
    lin = x @ _s6_beta(x.shape[1]).to(x.dtype)
    h0 = t**k * torch.exp(lin)
    return k * t ** (k - 1.0) * torch.exp(lin) / (1.0 + h0 / S6_FRAILTY_SHAPE)


#: scenario registry used by ``cells.generate_dataset`` (Methodology Sec. 3.1)
SCENARIOS: dict[str, dict] = {
    "S1": {"simulate": simulate_s1, "hazard": s1_true_hazard, "cdf": s1_true_cdf, "density": s1_true_density},
    "S2": {"simulate": simulate_s2, "hazard": s2_true_hazard, "cdf": s2_true_cdf, "density": s2_true_density},
    "S3": {"simulate": simulate_s3, "hazard": s3_true_hazard, "cdf": s3_true_cdf, "density": s3_true_density},
    "S4": {"simulate": simulate_s4, "hazard": s4_true_hazard, "cdf": s4_true_cdf, "density": s4_true_density},
    "S5": {"simulate": simulate_s5, "hazard": s5_true_hazard, "cdf": s5_true_cdf, "density": s5_true_density},
    "S6": {"simulate": simulate_s6, "hazard": s6_true_hazard, "cdf": s6_true_cdf, "density": s6_true_density},
}
