# FlowSurv-AFT — Project Context for AI Assistants

## What this project is

Development of **FlowSurv-AFT**, a novel deep-learning survival analysis method targeting a Q2/Q3 statistics journal (Lifetime Data Analysis / Computational Statistics / Journal of Applied Statistics / Statistics in Biosciences).

**Core idea:** a deep accelerated failure time (AFT) model, `log T = μ(x) + σ(x)·ε`, where the residual law of ε is a conditional **rational-quadratic spline (RQS) normalizing flow**, trained by the **exact right-censored full likelihood** (density term for events, survival term for censored). This gives: closed-form exact density f(t|x), survival S(t|x), hazard, quantiles, and one-pass sampling (analytic inverse in both directions) + interpretable AFT time ratios from μ(x).

**Origin:** extension of Opatha & Jayasinghe (2024), ISC 2024 Proceedings p. 55 (Cox vs Weibull AFT across censoring proportions/hazard shapes).

## The three differentiators (verified by full literature scan + full-text checks)

1. **Bidirectional exactness** — RQS flows are analytically invertible both ways. Closest neighbors each get half: Ausset et al. 2021 (IEEE DSAA, arXiv:2107.12825) used continuous ODE flows (numerical solver per evaluation); Li & Cai 2026 (STAI-X submission, concurrent work, `papers/101_A_Deep_Generative_Framewor.pdf`) used discrete monotone tanh-sum flows with exact forward likelihood but 40-step bisection for any quantile/sample.
2. **Explicit interpretable AFT decomposition** — μ(x) gives time ratios; Ausset's "generalization of AFT" is a one-sentence framing remark only. Never claim "first to connect flows to AFT".
3. **First systematic hazard-shape-recovery + calibration study across censoring regimes** — bathtub/multimodal/crossing hazards × 0–80% censoring × Type I/III, with D-calibration/ICI/IBS + true-vs-estimated hazard L2 error. Verified absent from all prior work.

## Hard rules (from evaluation literature — do not violate)

- Never sell C-index wins (Burk et al. 2024: nothing beats Cox on standard tabular data). Sell calibration + hazard recovery + interpretability.
- Report only Uno's C as concordance (avoid "C-hacking"). D-calibration + ICI mandatory for calibration claims.
- Baselines must include Royston–Parmar flexible parametric (M-splines) + DSM + DeepSurv + DeepHit + RSF + Cox + Weibull/log-normal AFT + FlowSurv-Gauss ablation.
- Identical 30-config tuning budget for all DL methods (frozen-then-audit protocol).
- Pre-submission: sweep for post-2026 spline-flow-survival preprints; verify Lifetime Data Analysis quartile.

## Key files

- `docs/01_Introduction_Literature_Aims.md` — intro, lit review, aim, objectives O1–O5, RQ1–4, hypotheses H1–H4, full reference list
- `docs/02_Methodology.md` — complete methodology: model math, closed-form outputs, likelihood, training protocol, simulation design (DGPs S1–S6 with explicit formulas, 120 cells × 100 reps), metrics with formulas, real-data plan (SUPPORT/METABRIC/GBSG/FLCHAIN/WHAS), 6 implementation gate tests, threats to validity
- `docs/03_Implementation_Plan.md` — repo layout, 9 phases (22 weeks), exit checklists, compute budget, risk checkpoints
- `papers/101_A_Deep_Generative_Framewor.pdf` — Li & Cai 2026 (concurrent competitor; cite and differentiate) — git-ignored, never commit
- `papers/98243_AUSSET_2021_archivage.pdf` — Ausset PhD thesis (closest prior work; cite and differentiate) — git-ignored, never commit

## Current status

**Phase 0 complete** (2026-08-04): `torch_gpu` conda env verified + `environment.yml` written; novelty sweep documented in `preregistration.md` Appendix A (no new threat; GRAFT/LT-ICL logged as adjacent); pre-registration frozen as prereg-v1; repo skeleton + CI created, pytest green. Design docs live in `docs/`, reference PDFs in `papers/` (git-ignored).

**Next action = Phase 1 of `docs/03_Implementation_Plan.md`:** implement the core model (`src/flowsurv/models/`: `rqs_flow.py`, `encoder.py`, `flowsurv.py`, `losses.py`) and the 6 gate tests from `docs/02_Methodology.md` §6. All 6 tests must pass before any experiment.

## Working agreements

- Ask the user before any git mutations (commit/push/etc.).
- Keep code PyTorch-based; RQS flows via `zuko` (fallback `nflows`).
- Follow the repo layout in `docs/03_Implementation_Plan.md` exactly.
- Do not create extra documentation files unless the user asks.
- PDFs under `papers/` are git-ignored (`*.pdf`) — never commit or push them.
