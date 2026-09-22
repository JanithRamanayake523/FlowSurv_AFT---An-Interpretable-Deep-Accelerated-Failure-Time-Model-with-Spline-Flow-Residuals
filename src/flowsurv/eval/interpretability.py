"""Pillar C -- interpretability analysis (Methodology Sec. 5, Implementation Plan Phase 7).

Three deliverables, each a standalone function operating on an already-fitted
:class:`~flowsurv.models.flowsurv.FlowSurvAFT` (and, for time-ratio
concordance, an already-fitted ``lifelines.WeibullAFTFitter``):

1. :func:`time_ratio_table` -- per-covariate, per-subject time ratios
   TR = exp(mu(x_a) - mu(x_b)) against the classical Weibull-AFT time ratio
   exp(beta_j * (x_a - x_b)); feeds ``eval.analyze.time_ratio_concordance``
   / contrast C4 (H4). Output columns match ``analyze.TIME_RATIOS_PATH``:
   ``dataset, covariate, tr_flow, tr_weibull``.
2. :func:`mu_pdp_ice` -- partial-dependence and individual-conditional-
   expectation curves of mu(x) over one covariate (Methodology Sec. 5, item 2).
3. :func:`case_study_curves` -- per-subject density/survival/hazard curves for
   a small set of profiles (Methodology Sec. 5, item 3: qualitative figure).

Each function takes model objects directly rather than method names, so it
works the same on simulation or real-data fits; nothing here depends on the
grid driver or the tuning protocol.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from ..models import FlowSurvAFT

# ---------------------------------------------------------------------------
# 1. time-ratio concordance (Methodology Sec. 5, item 1; prereg contrast C4)
# ---------------------------------------------------------------------------


def weibull_lambda_coefs(weibull_model) -> pd.Series:
    """Extract the AFT log-time-ratio coefficients beta_j from a fitted
    ``lifelines.WeibullAFTFitter``.

    lifelines parameterizes ``lambda_(x) = exp(beta . x)``, so a one-unit
    increase in covariate j multiplies the whole event-time distribution by
    ``exp(beta_j)`` -- exactly the classical AFT time-ratio semantics this
    module compares FlowSurv-AFT against. Returns the ``lambda_`` block of
    ``params_`` indexed by covariate name, Intercept excluded.
    """
    coefs = weibull_model.params_["lambda_"]
    return coefs.drop(index="Intercept", errors="ignore")


def time_ratio_table(
    flow_model: FlowSurvAFT,
    weibull_coefs: pd.Series,
    x: Tensor,
    covariate_names: list[str],
    dataset: str,
    active_covariates: list[str] | None = None,
    reference: str = "mean",
) -> pd.DataFrame:
    """Per-subject, per-covariate time ratios: FlowSurv-AFT vs Weibull-AFT.

    For each active covariate j, every subject's own value of x_j is
    contrasted against a fixed reference (population mean or median) with all
    other covariates held at the subject's own profile:

        x_a = subject's full profile
        x_b = x_a with covariate j replaced by the reference value

        tr_flow_i    = exp(mu_flow(x_a) - mu_flow(x_b))
        tr_weibull_i = exp(beta_j * (x_a[j] - x_b[j]))

    This keeps both sides genuinely varying across subjects (unlike a fixed
    global TR_j), which is what a scatter / concordance-correlation
    comparison (H4) needs. ``reference`` is ``"mean"`` or ``"median"``.

    Returns one row per (subject, active covariate): columns ``dataset,
    covariate, subject, tr_flow, tr_weibull`` -- a superset of the
    ``dataset, covariate, tr_flow, tr_weibull`` schema
    ``eval.analyze.time_ratio_concordance`` / C4 consume.
    """
    x = torch.as_tensor(x, dtype=torch.float32)
    if x.shape[1] != len(covariate_names):
        raise ValueError(f"x has {x.shape[1]} columns but {len(covariate_names)} names given")

    if reference == "mean":
        x_ref = x.mean(dim=0)
    elif reference == "median":
        x_ref = x.median(dim=0).values
    else:
        raise ValueError(f"unknown reference {reference!r}; use 'mean' or 'median'")

    active = active_covariates or [c for c in covariate_names if c in weibull_coefs.index]
    unknown = [c for c in active if c not in covariate_names]
    if unknown:
        raise ValueError(f"unknown covariates {unknown}; known: {covariate_names}")

    flow_model.eval()
    rows: list[pd.DataFrame] = []
    with torch.no_grad():
        mu_a_full = flow_model.location(x)  # mu at each subject's own full profile
        for cov in active:
            j = covariate_names.index(cov)
            x_b = x.clone()
            x_b[:, j] = x_ref[j]
            mu_b = flow_model.location(x_b)
            tr_flow = (mu_a_full - mu_b).exp().cpu().numpy()

            beta_j = float(weibull_coefs.get(cov, np.nan))
            tr_weibull = np.exp(beta_j * (x[:, j] - x_ref[j]).cpu().numpy())

            rows.append(
                pd.DataFrame(
                    {
                        "dataset": dataset,
                        "covariate": cov,
                        "subject": np.arange(x.shape[0]),
                        "tr_flow": tr_flow,
                        "tr_weibull": tr_weibull,
                    }
                )
            )
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["dataset", "covariate", "subject", "tr_flow", "tr_weibull"]
    )


def save_time_ratios(
    tables: list[pd.DataFrame],
    path: str | Path = "experiments/interpretability/time_ratios.parquet",
) -> Path:
    """Concatenate per-dataset :func:`time_ratio_table` outputs and write the
    parquet ``eval.analyze.TIME_RATIOS_PATH`` / contrast C4 read."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(tables, ignore_index=True).to_parquet(path, index=False)
    return path


# ---------------------------------------------------------------------------
# 2. PDP / ICE of mu(x) (Methodology Sec. 5, item 2)
# ---------------------------------------------------------------------------


def mu_pdp_ice(
    flow_model: FlowSurvAFT,
    x: Tensor,
    covariate_idx: int,
    grid: Tensor,
    covariate_name: str = "x",
    dataset: str = "dataset",
    max_subjects: int | None = None,
) -> pd.DataFrame:
    """Partial-dependence and individual-conditional-expectation curves of
    mu(x) over one covariate, holding all others at each subject's own value.

    ICE row for subject i, grid point v: mu(x_i with covariate_idx set to v).
    PDP at v is the mean of the ICE curves at v (the standard PDP-from-ICE
    definition). ``max_subjects`` subsamples subjects (deterministically,
    first-N) for readable ICE plots on large datasets; PDP always uses the
    full ``x``.

    Returns long-format rows: ``dataset, covariate, subject, grid_value,
    mu`` (``subject`` is ``-1`` for the PDP row, so PDP and ICE plot off the
    same table without a second query).
    """
    x = torch.as_tensor(x, dtype=torch.float32)
    grid = torch.as_tensor(grid, dtype=torch.float32)
    flow_model.eval()

    with torch.no_grad():
        # (G, n): mu for every subject at every grid value of the covariate
        x_rep = x.unsqueeze(0).expand(grid.shape[0], -1, -1).clone()
        x_rep[:, :, covariate_idx] = grid.unsqueeze(1)
        mu_grid = flow_model.location(x_rep)  # (G, n)

    pdp = mu_grid.mean(dim=1).cpu().numpy()
    rows = [
        pd.DataFrame(
            {
                "dataset": dataset,
                "covariate": covariate_name,
                "subject": -1,
                "grid_value": grid.cpu().numpy(),
                "mu": pdp,
            }
        )
    ]

    n_ice = x.shape[0] if max_subjects is None else min(max_subjects, x.shape[0])
    ice = mu_grid[:, :n_ice].cpu().numpy()  # (G, n_ice)
    for i in range(n_ice):
        rows.append(
            pd.DataFrame(
                {
                    "dataset": dataset,
                    "covariate": covariate_name,
                    "subject": i,
                    "grid_value": grid.cpu().numpy(),
                    "mu": ice[:, i],
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------------------
# 3. case-study curves (Methodology Sec. 5, item 3)
# ---------------------------------------------------------------------------


def case_study_curves(
    flow_model: FlowSurvAFT,
    x_subjects: Tensor,
    t_grid: Tensor,
    dataset: str = "dataset",
    subject_ids: list | None = None,
) -> pd.DataFrame:
    """Per-subject density / survival / hazard curves on a fixed time grid.

    For a small hand-picked set of profiles (Methodology Sec. 5, item 3: "3-4
    individual conditional density/hazard curves ... where FlowSurv-AFT
    infers a non-standard shape that DSM/parametric models smooth away").
    Long format: ``dataset, subject, t, density, survival, hazard``.
    """
    x_subjects = torch.as_tensor(x_subjects, dtype=torch.float32)
    t_grid = torch.as_tensor(t_grid, dtype=torch.float32)
    n = x_subjects.shape[0]
    ids = subject_ids if subject_ids is not None else list(range(n))
    if len(ids) != n:
        raise ValueError(f"{len(ids)} subject_ids for {n} subjects")

    flow_model.eval()
    with torch.no_grad():
        # grid case: t (m,), x (n, p) -> (m, n)
        density = flow_model.density(t_grid, x_subjects).cpu().numpy()
        survival = flow_model.survival(t_grid, x_subjects).cpu().numpy()
        hazard = flow_model.hazard(t_grid, x_subjects).cpu().numpy()

    t_np = t_grid.cpu().numpy()
    rows = []
    for i, sid in enumerate(ids):
        rows.append(
            pd.DataFrame(
                {
                    "dataset": dataset,
                    "subject": sid,
                    "t": t_np,
                    "density": density[:, i],
                    "survival": survival[:, i],
                    "hazard": hazard[:, i],
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def save_case_studies(
    tables: list[pd.DataFrame],
    path: str | Path = "experiments/interpretability/case_studies.parquet",
) -> Path:
    """Concatenate per-dataset :func:`case_study_curves` outputs to parquet."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(tables, ignore_index=True).to_parquet(path, index=False)
    return path
