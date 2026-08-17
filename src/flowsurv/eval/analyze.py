"""Phase 6/7/8 analysis: bootstrap CIs, mixed-effects models, contrasts, figures.

Implements the pre-registered analysis plan (prereg Sec. 5):

- Per (method, cell): median metric over replications with 10,000-replicate
  bootstrap percentile CIs.
- Across cells: linear mixed-effects models per metric (fixed effects method,
  scenario, n, censoring, censoring type + two-way method x design-factor
  interactions; random intercept for replication batch ``rep // 10``).
- Primary contrasts C1-C4 (prereg Sec. 2); Holm correction elsewhere.
- Real data: median [IQR] over the 10 splits; paired Wilcoxon + Holm.
- Figures F1-F5: every PNG is accompanied by the aggregated CSV it was drawn
  from, so every paper number is traceable to metrics.parquet (Phase 8).
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

BOOTSTRAP_REPS = 10_000
ANALYSIS_SEED = 20260804  # prereg master seed

METRIC_COLUMNS = [
    "unos_c",
    "ibs",
    "dcal_stat",
    "dcal_p",
    "dcal_pass",
    "ici",
    "hre",
    "ks",
    "w1",
    "time_fit_s",
    "time_eval_s",
    "time_sample_s",
]
SUMMARY_METRICS = ["unos_c", "ibs", "dcal_pass", "ici", "hre", "ks", "w1"]
REAL_METRICS = ["unos_c", "ibs", "dcal_pass", "ici"]

FLOW_METHOD = "flowsurv_aft"
DISCRIMINATIVE_DL = ["deepsurv", "deephit"]

# Pre-registered non-inferiority margins (prereg Sec. 6).
DELTA_IBS = 0.01
DELTA_ICI = 0.02

# Expected schema of the Pillar C interpretability output consumed by C4 / F2.
TIME_RATIOS_PATH = Path("experiments/interpretability/time_ratios.parquet")
#   columns: dataset (cell id or real dataset name), covariate, tr_flow, tr_weibull
HAZARD_PREDICTIONS_PATH = Path("experiments/predictions/hazards.parquet")
#   long format, columns: cell_id, rep, method, subject, t, h_pred, h_true


# ---------------------------------------------------------------------------
# loading + per-cell summaries
# ---------------------------------------------------------------------------


def load_results(path: str | Path = "experiments/metrics") -> pd.DataFrame:
    """Concatenate all parquet part-files under ``path`` into one DataFrame."""
    files = sorted(Path(path).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet files under {path}")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def bootstrap_median_ci(
    values,
    n_boot: int = BOOTSTRAP_REPS,
    alpha: float = 0.05,
    seed: int = ANALYSIS_SEED,
) -> tuple[float, float, float]:
    """Median and bootstrap percentile CI (vectorized; prereg Sec. 5).

    NaNs are dropped. Returns ``(median, lo, hi)``; NaN triplet if empty.
    """
    v = np.asarray(values, dtype=float).ravel()
    v = v[~np.isnan(v)]
    if v.size == 0:
        return (np.nan, np.nan, np.nan)
    med = float(np.median(v))
    if v.size == 1:
        return (med, med, med)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    meds = np.median(v[idx], axis=1)
    lo, hi = np.percentile(meds, [100.0 * alpha / 2, 100.0 * (1 - alpha / 2)])
    return (med, float(lo), float(hi))


def cell_summary(
    df: pd.DataFrame,
    metrics: list[str] | None = None,
    n_boot: int = BOOTSTRAP_REPS,
    seed: int = ANALYSIS_SEED,
) -> pd.DataFrame:
    """Per (method, cell_id): median + bootstrap CI over reps for each metric."""
    metrics = metrics or SUMMARY_METRICS
    rows = []
    for (method, cell_id), g in df.groupby(["method", "cell_id"], observed=True):
        row: dict = {"method": method, "cell_id": cell_id, "n_reps": int(g["rep"].nunique())}
        for col in ("scenario", "n", "censoring", "censoring_type"):
            if col in g.columns:
                row[col] = g[col].iloc[0]
        for m in metrics:
            med, lo, hi = bootstrap_median_ci(g[m].to_numpy(), n_boot=n_boot, seed=seed)
            row[f"{m}_median"], row[f"{m}_lo"], row[f"{m}_hi"] = med, lo, hi
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# mixed-effects models (prereg Sec. 5)
# ---------------------------------------------------------------------------


def mixed_effects(df: pd.DataFrame, metric: str):
    """Linear mixed-effects model for ``metric`` (prereg Sec. 5).

    Fixed effects: method + scenario + C(n) + C(censoring) + censoring_type +
    two-way method x design-factor interactions. Random intercept for
    replication batch (``batch = rep // 10``). Returns the statsmodels result,
    or ``None`` (with a warning) on convergence/fit failure.
    """
    import statsmodels.formula.api as smf

    d = df.dropna(subset=[metric]).copy()
    d["batch"] = (d["rep"].astype(int) // 10).astype(int)
    formula = (
        f"{metric} ~ method + scenario + C(n) + C(censoring) + censoring_type"
        " + method:scenario + method:C(n) + method:C(censoring) + method:censoring_type"
    )
    try:
        model = smf.mixedlm(formula, d, groups=d["batch"])
        result = model.fit(method="lbfgs", maxiter=500)
    except Exception as e:  # noqa: BLE001 -- guard convergence failures
        warnings.warn(f"mixedlm({metric}) failed: {e!r}")
        return None
    if not getattr(result, "converged", True):
        warnings.warn(f"mixedlm({metric}) did not converge; interpret with care")
    return result


# ---------------------------------------------------------------------------
# primary contrasts (prereg Sec. 2)
# ---------------------------------------------------------------------------


def _paired_by(g: pd.DataFrame, method_a: str, method_b: str, metric: str, keys: list[str]):
    """Paired (a, b) rep-level values of ``metric`` for two methods."""
    a = g[g["method"] == method_a].set_index(keys)[metric]
    b = g[g["method"] == method_b].set_index(keys)[metric]
    both = pd.concat([a, b], axis=1, keys=["a", "b"]).dropna()
    return both["a"].to_numpy(), both["b"].to_numpy()


def _wilcoxon_pvalue(a, b) -> float:
    from scipy.stats import wilcoxon

    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    diff = diff[~np.isnan(diff)]
    if diff.size < 5 or np.all(diff == 0):
        return np.nan
    try:
        return float(wilcoxon(diff).pvalue)
    except ValueError:
        return np.nan


def _contrast_c1(df: pd.DataFrame, seed: int) -> list[dict]:
    """C1 (H1): FlowSurv-AFT vs DSM, HRE, scenarios S2-S4."""
    sub = df[df["scenario"].isin(["S2", "S3", "S4"]) & df["method"].isin([FLOW_METHOD, "dsm"])]
    rows = []
    for cell_id, g in sub.groupby("cell_id", observed=True):
        a, b = _paired_by(g, FLOW_METHOD, "dsm", "hre", ["rep"])
        if a.size < 5:
            continue
        med, lo, hi = bootstrap_median_ci(a - b, seed=seed)
        rows.append(
            {
                "contrast": "C1",
                "cell_id": cell_id,
                "metric": "hre",
                "other_method": "dsm",
                "n_pairs": a.size,
                "median_flow": float(np.median(a)),
                "median_other": float(np.median(b)),
                "median_diff": med,
                "ci_lo": lo,
                "ci_hi": hi,
                "pvalue": _wilcoxon_pvalue(a, b),
            }
        )
    return rows


def _contrast_c2(df: pd.DataFrame, seed: int) -> list[dict]:
    """C2 (H2): FlowSurv-AFT vs DeepHit/DeepSurv calibration trend across censoring.

    Per censoring level: paired Wilcoxon on ICI (pairs keyed by cell_id x rep)
    plus D-calibration pass rates; per method: slope of the pass rate across
    censoring levels (least-squares line) -- H2 predicts a flatter/steeper
    disadvantage slope for the discriminative DL baselines.
    """
    rows = []
    sim = df[df["censoring_type"] != "real"]
    for other in ["deephit", "deepsurv"]:
        sub = sim[sim["method"].isin([FLOW_METHOD, other])]
        for cens, g in sub.groupby("censoring", observed=True):
            a, b = _paired_by(g, FLOW_METHOD, other, "ici", ["cell_id", "rep"])
            rate = g.groupby("method")["dcal_pass"].mean()
            rows.append(
                {
                    "contrast": "C2",
                    "censoring": float(cens),
                    "metric": "ici",
                    "other_method": other,
                    "n_pairs": a.size,
                    "median_flow": float(np.median(a)) if a.size else np.nan,
                    "median_other": float(np.median(b)) if b.size else np.nan,
                    "dcal_pass_flow": float(rate.get(FLOW_METHOD, np.nan)),
                    "dcal_pass_other": float(rate.get(other, np.nan)),
                    "pvalue": _wilcoxon_pvalue(a, b),
                }
            )
        for method in (FLOW_METHOD, other):
            rates = sub[sub["method"] == method].groupby("censoring")["dcal_pass"].mean()
            if len(rates) >= 2:
                slope = float(np.polyfit(rates.index.to_numpy(float), rates.to_numpy(float), 1)[0])
            else:
                slope = np.nan
            rows.append(
                {
                    "contrast": "C2_slope",
                    "metric": "dcal_pass",
                    "other_method": method,
                    "median_diff": slope,  # pass-rate slope across censoring levels
                }
            )
    return rows


def _contrast_c3(df: pd.DataFrame, margin_ibs: float, margin_hre: float | None, seed: int) -> list[dict]:
    """C3 (H3): FlowSurv-AFT vs Weibull AFT on S1, non-inferiority.

    Non-inferiority holds when the upper bootstrap CI bound of the paired
    difference (flow - weibull) is below the margin. The IBS margin is
    pre-registered (0.01); no numeric HRE margin was frozen in prereg-v1, so
    the HRE CI is reported and the NI decision left out unless a margin is
    passed explicitly (log any choice as a deviation, prereg Sec. 8).
    """
    sub = df[(df["scenario"] == "S1") & df["method"].isin([FLOW_METHOD, "weibull_aft"])]
    rows = []
    for cell_id, g in sub.groupby("cell_id", observed=True):
        for metric, margin in (("hre", margin_hre), ("ibs", margin_ibs)):
            a, b = _paired_by(g, FLOW_METHOD, "weibull_aft", metric, ["rep"])
            if a.size < 5:
                continue
            med, lo, hi = bootstrap_median_ci(a - b, seed=seed)
            rows.append(
                {
                    "contrast": "C3",
                    "cell_id": cell_id,
                    "metric": metric,
                    "other_method": "weibull_aft",
                    "n_pairs": a.size,
                    "median_flow": float(np.median(a)),
                    "median_other": float(np.median(b)),
                    "median_diff": med,
                    "ci_lo": lo,
                    "ci_hi": hi,
                    "ni_margin": margin if margin is not None else np.nan,
                    "non_inferior": bool(hi < margin) if margin is not None else None,
                }
            )
    return rows


def time_ratio_concordance(tr_flow, tr_weibull) -> dict:
    """C4 (H4, descriptive): Pearson r + Lin's concordance correlation coefficient.

    CCC = 2 * cov(x, y) / (var(x) + var(y) + (mean(x) - mean(y))^2) with
    sample (ddof=1) moments; equals Pearson r only when means/variances agree.
    """
    x = np.asarray(tr_flow, dtype=float).ravel()
    y = np.asarray(tr_weibull, dtype=float).ravel()
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    n = int(x.size)
    if n < 2:
        return {"n": n, "pearson_r": np.nan, "ccc": np.nan}
    cov = float(np.cov(x, y, ddof=1)[0, 1])
    vx, vy = float(np.var(x, ddof=1)), float(np.var(y, ddof=1))
    r = cov / np.sqrt(vx * vy) if vx > 0 and vy > 0 else np.nan
    ccc = 2.0 * cov / (vx + vy + (float(x.mean()) - float(y.mean())) ** 2)
    return {"n": n, "pearson_r": float(r), "ccc": float(ccc)}


def _contrast_c4(path: str | Path = TIME_RATIOS_PATH) -> list[dict]:
    """C4 placeholder: consumes Pillar C time-ratio outputs.

    Expects ``experiments/interpretability/time_ratios.parquet`` (produced by
    the interpretability analysis, Methodology Sec. 5) with columns
    ``dataset, covariate, tr_flow, tr_weibull``. Returns an empty list with a
    warning if the file does not exist yet.
    """
    path = Path(path)
    if not path.exists():
        warnings.warn(f"C4 skipped: {path} not found (produced by Pillar C analysis)")
        return []
    tbl = pd.read_parquet(path)
    rows = []
    for dataset, g in tbl.groupby("dataset", observed=True):
        stats = time_ratio_concordance(g["tr_flow"], g["tr_weibull"])
        rows.append({"contrast": "C4", "dataset": dataset, **stats})
    return rows


def primary_contrasts(
    df: pd.DataFrame,
    margin_ibs: float = DELTA_IBS,
    margin_hre: float | None = None,
    time_ratios_path: str | Path = TIME_RATIOS_PATH,
    seed: int = ANALYSIS_SEED,
) -> pd.DataFrame:
    """Evaluate pre-registered primary contrasts C1-C4 (prereg Sec. 2)."""
    rows = (
        _contrast_c1(df, seed)
        + _contrast_c2(df, seed)
        + _contrast_c3(df, margin_ibs, margin_hre, seed)
        + _contrast_c4(time_ratios_path)
    )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# real-data summaries and tests (Methodology Sec. 4.2, prereg Sec. 5)
# ---------------------------------------------------------------------------


def real_summary(df: pd.DataFrame, metrics: list[str] | None = None) -> pd.DataFrame:
    """Median [IQR] per dataset x method over the 10 splits."""
    metrics = metrics or REAL_METRICS
    d = df[df["censoring_type"] == "real"]
    rows = []
    for (dataset, method), g in d.groupby(["scenario", "method"], observed=True):
        row: dict = {"dataset": dataset, "method": method, "n_splits": int(g["rep"].nunique())}
        for m in metrics:
            v = g[m].to_numpy(dtype=float)
            v = v[~np.isnan(v)]
            if v.size:
                q25, med, q75 = np.percentile(v, [25, 50, 75])
                row[f"{m}_median"], row[f"{m}_q25"], row[f"{m}_q75"] = med, q25, q75
                row[f"{m}_median_iqr"] = f"{med:.3f} [{q25:.3f}, {q75:.3f}]"
            else:
                row[f"{m}_median"] = row[f"{m}_q25"] = row[f"{m}_q75"] = np.nan
                row[f"{m}_median_iqr"] = ""
        rows.append(row)
    return pd.DataFrame(rows)


def holm_adjust(pvalues) -> np.ndarray:
    """Holm step-down adjusted p-values (implemented from sorted raw p-values)."""
    p = np.asarray(pvalues, dtype=float)
    m = p.size
    adjusted = np.full(m, np.nan)
    finite = np.isfinite(p)
    order = np.argsort(p[finite], kind="stable")
    running = 0.0
    for rank, idx in enumerate(order):
        value = (finite.sum() - rank) * p[finite][idx]
        running = max(running, value)
        adjusted[np.flatnonzero(finite)[idx]] = min(1.0, running)
    return adjusted


def real_tests(
    df: pd.DataFrame,
    metrics: list[str] | None = None,
    reference: str = FLOW_METHOD,
) -> pd.DataFrame:
    """Paired Wilcoxon signed-rank across the 10 splits + Holm correction.

    One row per (dataset, metric, method vs ``reference``); Holm applied within
    each (dataset, metric) family.
    """
    metrics = metrics or REAL_METRICS
    d = df[df["censoring_type"] == "real"]
    rows = []
    for dataset, gd in d.groupby("scenario", observed=True):
        for metric in metrics:
            family = []
            for method, gm in gd.groupby("method", observed=True):
                if method == reference:
                    continue
                a, b = _paired_by(gd, reference, method, metric, ["rep"])
                family.append(
                    {
                        "dataset": dataset,
                        "metric": metric,
                        "method": method,
                        "reference": reference,
                        "n_pairs": a.size,
                        "median_diff": float(np.median(a - b)) if a.size else np.nan,
                        "p_raw": _wilcoxon_pvalue(a, b),
                    }
                )
            if family:
                p_holm = holm_adjust([r["p_raw"] for r in family])
                for r, ph in zip(family, p_holm):
                    r["p_holm"] = ph
                rows.extend(family)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# success criteria (prereg Sec. 6)
# ---------------------------------------------------------------------------


def success_criteria(
    df_real: pd.DataFrame,
    reference: str = FLOW_METHOD,
    discriminative_dl: list[str] | None = None,
    delta_ibs: float = DELTA_IBS,
    delta_ici: float = DELTA_ICI,
) -> pd.DataFrame:
    """Evaluate the pre-registered real-data success criteria (prereg Sec. 6).

    1. D-calibration pass rate of FlowSurv-AFT >= discriminative DL baselines.
    2. IBS non-inferior to the best baseline within Delta_IBS = 0.01.
    3. ICI non-inferior to the best baseline within Delta_ICI = 0.02.
    """
    discriminative_dl = discriminative_dl or DISCRIMINATIVE_DL
    rows = []
    for dataset, g in df_real.groupby("scenario", observed=True):
        med = g.groupby("method")[["ibs", "ici"]].median()
        pass_rate = g.groupby("method")["dcal_pass"].mean()
        if reference not in med.index:
            continue
        others = med.drop(index=reference, errors="ignore")
        dl_present = [m for m in discriminative_dl if m in pass_rate.index]
        row = {
            "dataset": dataset,
            "dcal_pass_flow": float(pass_rate.get(reference, np.nan)),
            "dcal_pass_best_dl": float(pass_rate[dl_present].max()) if dl_present else np.nan,
            "ibs_flow": float(med.loc[reference, "ibs"]),
            "ibs_best_baseline": float(others["ibs"].min()) if len(others) else np.nan,
            "ici_flow": float(med.loc[reference, "ici"]),
            "ici_best_baseline": float(others["ici"].min()) if len(others) else np.nan,
        }
        row["crit1_dcal_pass"] = bool(row["dcal_pass_flow"] >= row["dcal_pass_best_dl"])
        row["crit2_ibs_noninferior"] = bool(row["ibs_flow"] <= row["ibs_best_baseline"] + delta_ibs)
        row["crit3_ici_noninferior"] = bool(row["ici_flow"] <= row["ici_best_baseline"] + delta_ici)
        row["all_pass"] = bool(
            row["crit1_dcal_pass"] and row["crit2_ibs_noninferior"] and row["crit3_ici_noninferior"]
        )
        rows.append(row)
    out = pd.DataFrame(rows)
    if len(out):
        print(out.to_string(index=False))
    return out


# ---------------------------------------------------------------------------
# figures F1-F5 (Phase 6/8; PNG + aggregated CSV for traceability)
# ---------------------------------------------------------------------------


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def figure_f1(df: pd.DataFrame, out_dir: str | Path) -> list[Path]:
    """F1 performance maps: method x censoring heat-tables per scenario (HRE, IBS)."""
    plt = _plt()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    sim = df[df["censoring_type"] != "real"]
    for metric in ("hre", "ibs"):
        for scenario, g in sim.groupby("scenario", observed=True):
            cell = (
                g.groupby(["method", "cell_id", "censoring"], observed=True)[metric]
                .median()
                .reset_index()
            )
            pivot = cell.groupby(["method", "censoring"], observed=True)[metric].median().unstack()
            if pivot.dropna(how="all").empty:
                continue
            csv_path = out_dir / f"f1_{metric}_{scenario}.csv"
            pivot.to_csv(csv_path)
            fig, ax = plt.subplots(figsize=(1.2 + 1.1 * pivot.shape[1], 0.5 + 0.45 * pivot.shape[0]))
            im = ax.imshow(pivot.to_numpy(float), aspect="auto", cmap="viridis_r")
            ax.set_xticks(range(pivot.shape[1]), [str(c) for c in pivot.columns])
            ax.set_yticks(range(pivot.shape[0]), list(pivot.index))
            ax.set_xlabel("censoring")
            ax.set_title(f"{metric.upper()} median over reps — {scenario}")
            for i in range(pivot.shape[0]):
                for j in range(pivot.shape[1]):
                    v = pivot.iloc[i, j]
                    if np.isfinite(v):
                        ax.text(j, i, f"{v:.3g}", ha="center", va="center", fontsize=7, color="w")
            fig.colorbar(im, ax=ax, label=metric)
            fig.tight_layout()
            png_path = out_dir / f"f1_{metric}_{scenario}.png"
            fig.savefig(png_path, dpi=150)
            plt.close(fig)
            written.extend([png_path, csv_path])
    return written


def figure_f2(
    out_dir: str | Path,
    pred_path: str | Path = HAZARD_PREDICTIONS_PATH,
    n_panels: int = 4,
    n_subjects: int = 3,
) -> list[Path]:
    """F2 hazard-recovery curves: predicted vs true hazard example panels.

    Consumes cached predictions, expected at
    ``experiments/predictions/hazards.parquet`` (long format; columns
    ``cell_id, rep, method, subject, t, h_pred, h_true``) — written by the
    experiment driver for a small set of example cells. Skips with a warning
    if the cache does not exist.
    """
    pred_path = Path(pred_path)
    if not pred_path.exists():
        warnings.warn(f"F2 skipped: {pred_path} not found (cached predictions missing)")
        return []
    plt = _plt()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred = pd.read_parquet(pred_path)
    combos = list(pred.groupby(["cell_id", "method"], observed=True).groups)[:n_panels]
    fig, axes = plt.subplots(1, len(combos), figsize=(4 * len(combos), 3.4), squeeze=False)
    for ax, (cell_id, method) in zip(axes[0], combos):
        g = pred[(pred["cell_id"] == cell_id) & (pred["method"] == method)]
        g = g[g["rep"] == g["rep"].min()]
        for subject, gs in list(g.groupby("subject"))[:n_subjects]:
            gs = gs.sort_values("t")
            ax.plot(gs["t"], gs["h_true"], color=f"C{int(subject) % 10}", lw=2)
            ax.plot(gs["t"], gs["h_pred"], color=f"C{int(subject) % 10}", ls="--")
        ax.set_title(f"{cell_id}\n{method}", fontsize=9)
        ax.set_xlabel("t")
        ax.set_ylabel("h(t|x)")
    axes[0][0].plot([], [], "k-", label="true")
    axes[0][0].plot([], [], "k--", label="estimated")
    axes[0][0].legend(fontsize=8)
    fig.tight_layout()
    png_path = out_dir / "f2_hazard_recovery.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    csv_path = out_dir / "f2_hazard_recovery.csv"
    pred.to_csv(csv_path, index=False)
    return [png_path, csv_path]


def _wilson_ci(k: float, n: float, z: float = 1.959964) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n <= 0:
        return (np.nan, np.nan)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (center - half, center + half)


def figure_f3(df: pd.DataFrame, out_dir: str | Path) -> list[Path]:
    """F3 calibration-vs-censoring lines: D-cal pass rate +/- Wilson CI per method."""
    plt = _plt()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sim = df[df["censoring_type"] != "real"].dropna(subset=["dcal_pass"])
    agg = (
        sim.groupby(["method", "censoring"], observed=True)["dcal_pass"]
        .agg(["mean", "sum", "size"])
        .reset_index()
    )
    ci = agg.apply(lambda r: _wilson_ci(r["sum"], r["size"]), axis=1, result_type="expand")
    agg["lo"], agg["hi"] = ci[0], ci[1]
    csv_path = out_dir / "f3_dcal_pass_vs_censoring.csv"
    agg.to_csv(csv_path, index=False)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for i, (method, g) in enumerate(agg.groupby("method", observed=True)):
        g = g.sort_values("censoring")
        ax.plot(g["censoring"], g["mean"], marker="o", ms=3, lw=1.2, color=f"C{i % 10}", label=method)
        ax.fill_between(g["censoring"], g["lo"], g["hi"], color=f"C{i % 10}", alpha=0.12)
    ax.set_xlabel("censoring proportion")
    ax.set_ylabel("D-calibration pass rate")
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=8)
    fig.tight_layout()
    png_path = out_dir / "f3_dcal_pass_vs_censoring.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    return [png_path, csv_path]


def figure_f4(df: pd.DataFrame, out_dir: str | Path) -> list[Path]:
    """F4 wall-clock bars: median fit/eval/sample seconds per method (log scale)."""
    plt = _plt()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = ["time_fit_s", "time_eval_s", "time_sample_s"]
    med = df.groupby("method", observed=True)[cols].median()
    csv_path = out_dir / "f4_wallclock.csv"
    med.to_csv(csv_path)
    fig, ax = plt.subplots(figsize=(max(6, 0.9 * len(med)), 4))
    x = np.arange(len(med))
    width = 0.27
    for j, col in enumerate(cols):
        ax.bar(x + (j - 1) * width, med[col].to_numpy(float), width, label=col.replace("time_", "").replace("_s", ""))
    ax.set_xticks(x, list(med.index), rotation=45, ha="right")
    ax.set_yscale("log")
    ax.set_ylabel("seconds (median, log scale)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    png_path = out_dir / "f4_wallclock.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    return [png_path, csv_path]


def figure_f5(df: pd.DataFrame, out_dir: str | Path, metrics: list[str] | None = None) -> list[Path]:
    """F5 real-data metric boxplots per dataset (over the 10 splits)."""
    plt = _plt()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = metrics or ["unos_c", "ibs", "ici"]
    d = df[df["censoring_type"] == "real"]
    written = []
    for metric in metrics:
        datasets = sorted(d["scenario"].unique())
        if not datasets:
            continue
        fig, axes = plt.subplots(1, len(datasets), figsize=(3.2 * len(datasets), 4), squeeze=False)
        agg_rows = []
        for ax, dataset in zip(axes[0], datasets):
            g = d[d["scenario"] == dataset]
            methods = sorted(g["method"].unique())
            data = [g[g["method"] == m][metric].dropna().to_numpy(float) for m in methods]
            ax.boxplot(data, tick_labels=[m[:12] for m in methods])
            ax.set_title(dataset, fontsize=9)
            ax.set_ylabel(metric)
            ax.tick_params(axis="x", rotation=45)
            for m, v in zip(methods, data):
                if v.size:
                    agg_rows.append(
                        {
                            "dataset": dataset,
                            "method": m,
                            "median": float(np.median(v)),
                            "q25": float(np.percentile(v, 25)),
                            "q75": float(np.percentile(v, 75)),
                            "n": v.size,
                        }
                    )
        fig.tight_layout()
        png_path = out_dir / f"f5_real_{metric}.png"
        fig.savefig(png_path, dpi=150)
        plt.close(fig)
        csv_path = out_dir / f"f5_real_{metric}.csv"
        pd.DataFrame(agg_rows).to_csv(csv_path, index=False)
        written.extend([png_path, csv_path])
    return written
