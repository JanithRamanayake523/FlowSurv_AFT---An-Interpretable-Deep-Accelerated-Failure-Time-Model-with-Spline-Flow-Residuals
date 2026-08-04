# FlowSurv-AFT

**An Interpretable Deep Accelerated Failure Time Model with Spline-Flow Residuals.**

`log T = μ(x) + σ(x)·ε`, where the residual law of ε is a conditional **rational-quadratic spline (RQS) normalizing flow** (Logistic(0,1) base), trained by the **exact right-censored full likelihood** — density term for events, survival term for censored observations.

Three differentiators over prior work:

1. **Bidirectional exactness** — closed-form density `f(t|x)`, survival `S(t|x)`, hazard, quantiles, and one-pass sampling (analytic inverse in both directions). No ODE solver (Ausset et al. 2021), no bisection (Li & Cai 2026).
2. **AFT interpretability** — `μ(x)` yields time ratios `exp(μ(xₐ) − μ(x_b))`, the classical acceleration-factor semantics.
3. **First systematic hazard-shape-recovery + calibration study across censoring regimes** — 120 simulation cells (scenarios S1–S6 × n × 0–80% censoring × Type I/III) × 100 replications, evaluated with D-calibration/ICI/IBS and true-vs-estimated hazard L2 error (HRE).

Claims are made on **calibration and hazard recovery, not concordance** (Burk et al. 2024); Uno's C is the only reported concordance variant (Sonabend 2022).

## Project documents

- `docs/01_Introduction_Literature_Aims.md` — intro, literature review, aim, objectives O1–O5, hypotheses H1–H4
- `docs/02_Methodology.md` — model math, likelihood, training protocol, simulation design, metrics, gate tests
- `docs/03_Implementation_Plan.md` — 9 phases (22 weeks) with exit checklists
- `preregistration.md` — **frozen 2026-08-04 (prereg-v1)**, includes the final novelty-sweep appendix

## Status

**Phase 0 complete** (environment, novelty sweep, pre-registration, repo skeleton + CI). Next: Phase 1 — core model (`src/flowsurv/models/`) and the six gate tests (Methodology §6), all of which must pass before any experiment.

## Setup

Two routes:

- **Clean machine (canonical spec):** `conda env create -f environment.yml && conda activate flowsurv`
- **Existing env (local dev):** the project is developed in the `torch_gpu` conda env (Python 3.10, torch 2.5.1+CUDA); install missing pieces with
  `pip install -e . pytest` (plus `pycox scikit-survival lifelines` from Phase 3 onward).

## Tests

```bash
pytest -q
```

CI (`.github/workflows/ci.yml`) runs the same on every push. The six pre-experiment gate tests (Methodology §6) land in `tests/` during Phase 1 and gate all experiments.

## Repository layout

Per `docs/03_Implementation_Plan.md`: `configs/` (one YAML per experiment cell), `src/flowsurv/{models,data,baselines,metrics,eval}/`, `tests/`, `experiments/` (outputs, git-ignored), `notebooks/` (exploration only — nothing reported from here). Design documents live in `docs/`; reference PDFs (Ausset thesis, Li & Cai manuscript) live in `papers/` and are git-ignored — never committed.

## Reproduction

A `make reproduce` walkthrough (one full simulation cell end-to-end) is added in Phase 8, together with license and citation files.
