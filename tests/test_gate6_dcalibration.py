"""Gate test 6 (Methodology Sec. 6): calibration sanity.

On a large fresh uncensored S1 sample, the fitted model must pass
D-calibration (Haider et al. 2020): deciles of S_hat(t_i|x_i) are uniform
(probability integral transform), chi-square uniformity test at alpha=0.05.
"""

import torch
from scipy import stats

from flowsurv.data.dgp import simulate_s1

N_TEST = 20_000
DCAL_ALPHA = 0.05


def test_dcalibration_passes_on_uncensored_s1(s1_full_fit):
    t, x = simulate_s1(N_TEST, seed=99)  # fresh sample, same DGP
    with torch.no_grad():
        s = s1_full_fit.survival(t, x)
    bins = torch.clamp((s * 10).long(), max=9)  # decile of S in [0, .1), ..., [.9, 1]
    counts = torch.bincount(bins, minlength=10).numpy()
    chi2 = stats.chisquare(counts)
    assert chi2.pvalue > DCAL_ALPHA, (
        f"D-calibration chi2 p-value {chi2.pvalue:.4f} <= {DCAL_ALPHA}; counts {counts}"
    )
