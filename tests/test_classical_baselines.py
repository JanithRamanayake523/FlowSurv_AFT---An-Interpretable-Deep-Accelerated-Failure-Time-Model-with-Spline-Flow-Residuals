"""Default-penalizer regression test for the lifelines-backed baselines.

Deviation 2 (prereg Sec. 8, with its 2026-09-24 correction): Cox-PH /
Weibull-AFT / log-normal AFT fit with a fixed nonzero default penalizer (a
mild ridge that lowers small-n HRE; unregularized fits do not actually fail).
This test only checks the default is actually wired through to lifelines and
can still be overridden by the tuner-style ``penalizer`` kwarg.
"""

from __future__ import annotations

from flowsurv.baselines.classical import _DEFAULT_PENALIZER, CoxPH, LogNormalAFT, WeibullAFT
from flowsurv.data.censoring import apply_censoring
from flowsurv.data.dgp import simulate_s1

N = 500


def _s1_censored(seed=1, censoring=0.3):
    t, x = simulate_s1(N, seed=seed)
    t_obs, d = apply_censoring(t, censoring, "I", seed=seed + 1)
    return t_obs, d, x


def test_default_penalizer_is_nonzero_and_applied():
    assert _DEFAULT_PENALIZER > 0.0
    t, d, x = _s1_censored()
    for cls in (CoxPH, WeibullAFT, LogNormalAFT):
        method = cls()
        res = method.fit(t, d, x, seed=0)
        assert res.converged, res.info
        assert method.model.penalizer == _DEFAULT_PENALIZER


def test_penalizer_kwarg_still_overrides_default():
    t, d, x = _s1_censored()
    method = CoxPH()
    res = method.fit(t, d, x, seed=0, penalizer=0.1)
    assert res.converged
    assert method.model.penalizer == 0.1
