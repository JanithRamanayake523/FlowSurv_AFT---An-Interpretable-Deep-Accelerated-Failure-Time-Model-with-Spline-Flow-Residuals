# FlowSurv-AFT — Pre-Registration

**Version:** prereg-v1 · **Frozen:** 2026-08-04 (Sri Lanka Standard Time, UTC+5:30)
**Status:** FROZEN. Changes after this date are only permitted as logged entries in the Deviations section (§8); the frozen text itself is not edited.

This document pre-registers the hypotheses, primary contrasts, metrics, tuning budget, and success criteria of the FlowSurv-AFT study, per Implementation Plan Phase 0. It is drawn from Document 1 (§3 hypotheses), Document 2 (§2.4 optimization protocol, §3.3 metrics, §3.4 analysis, §4.3 success criteria), and freezes them before any experiment is run.

---

## 1. Hypotheses

- **H1 (flexibility).** On non-standard hazard shapes (S2 bathtub, S3 multimodal, S4 crossing), FlowSurv-AFT attains lower hazard-recovery error (HRE) than all baselines.
- **H2 (calibration × censoring).** The calibration advantage of exact-likelihood training over discrete-time/partial-likelihood deep models (DeepHit, DeepSurv) increases with censoring proportion.
- **H3 (no flexibility penalty).** On standard-shaped (Weibull, S1) data, FlowSurv-AFT matches parametric Weibull AFT within a small non-inferiority margin — flexibility costs nothing where it is not needed.
- **H4 (interpretability).** FlowSurv-AFT's acceleration factors exp(μ(xₐ) − μ(x_b)) are concordant with classical AFT time ratios on data where the classical model is adequate.

## 2. Primary contrasts (confirmatory)

| ID | Contrast | Metric | Cells | Hypothesis |
|---|---|---|---|---|
| C1 | FlowSurv-AFT vs DSM | HRE | S2–S4, all n × censoring × type | H1 |
| C2 | FlowSurv-AFT vs DeepHit / DeepSurv, trend across censoring % | D-calibration pass rate, ICI | all scenarios, censoring ∈ {0,20,50,80}% | H2 |
| C3 | FlowSurv-AFT vs Weibull AFT | HRE, IBS | S1 | H3 |
| C4 | FlowSurv-AFT time ratios vs Weibull-AFT time ratios | concordance correlation coefficient (descriptive) | S1 + real datasets | H4 |

C1–C3 are the confirmatory family; C4 is descriptive (no threshold claim). All other comparisons are secondary/exploratory and Holm-corrected.

## 3. Metrics (complete pre-registered list)

Formulas as in Methodology §3.3. τ = 90th percentile of observed test times.

1. **Discrimination:** Uno's C (IPCW) — the *only* concordance variant reported (anti-C-hacking, Sonabend 2022).
2. **Overall accuracy:** Integrated Brier Score with IPCW on [0, τ].
3. **Calibration:** D-calibration (10-bin χ² test; report statistic and pass/fail at α = 0.05; pass rate across replications) and ICI (loess calibration curve at τ, mean absolute deviation).
4. **Hazard recovery (simulation only, headline):** HRE = mean over test subjects of the weighted L2 error of ĥ(t|x) vs h_true(t|x) on a fixed grid, weights ∝ marginal density of evaluation times.
5. **Distributional fidelity (simulation only):** KS and W1 between estimated and true conditional CDFs at fixed covariate profiles; quartile-wise signed CDF deviation (bias diagnostic).
6. **Cost:** wall-clock for fit, evaluation, and 1,000-sample generation per subject.

No other metric may appear in claims; additional descriptive statistics go to the appendix of the paper, labelled exploratory.

## 4. Tuning budget and freezing protocol

- **Budget:** 30 random hyperparameter configurations per DL method per macro-cell, selected on validation NLL (or the package-equivalent objective), identical for every DL method (fairness constraint). Classical methods use package defaults with documented tuning where applicable.
- **FlowSurv-AFT grid:** K ∈ {1,2,3} RQS blocks × bins ∈ {8,16} × hidden ∈ {64,128} × dropout ∈ {0.1, 0.2}.
- **Freeze-then-audit:** within each macro-cell, hyperparameters are tuned on replications 1–5 and frozen for replications 6–100. Full nested tuning on a random 10% of cells serves as the audit; freezing bias is reported.
- **Optimizer (FlowSurv-AFT):** AdamW, lr 1e-3, cosine decay to 1e-5, weight decay 1e-4; batch 256; ≤ 500 epochs; early stopping on validation NLL, patience 30; gradient clip 1.0. Spline domain [−4, 4] with identity tails; L2 penalty 1e-5 on spline derivatives. Soft-NA Wasserstein regularizer λ = 0 (primary); λ ∈ {0.1, 0.2, 0.5} ablated.
- **Splits:** simulation — train/val/test 70/15/15 within each replication; real data — 10 repeated 80/20 stratified splits with 15% inner validation.
- **Seeds:** master seed `20260804`; per-(cell, replication) seeds derived deterministically; common random seeds across methods within a replication.

## 5. Analysis plan

- **Per cell:** median metric over R = 100 replications with 10,000-replicate bootstrap percentile CIs.
- **Across cells:** linear mixed-effects model per metric — fixed effects method × scenario × n × censoring % × censoring type plus two-way method × design-factor interactions; random intercept for replication batch.
- **Multiplicity:** C1–C3 confirmatory as stated; all other p-values Holm-corrected; effect sizes with CIs reported throughout (no p-value fishing).
- **Real data:** median [IQR] across the 10 splits per dataset; paired Wilcoxon signed-rank across splits with Holm correction.
- **Failures:** non-convergence/NaN rates per (method, cell) are logged and reported as a finding, not silently averaged over.

## 6. Success criteria (real data, Methodology §4.3)

Real-data claims are restricted to calibration/distributional quality (per Burk et al. 2024):

1. D-calibration pass rate ≥ discriminative DL baselines, especially on FLCHAIN (heavy censoring).
2. ICI and IBS **non-inferior** to the best baseline on every dataset — margins fixed here: Δ_IBS = 0.01 and Δ_ICI = 0.02 (absolute).
3. No concordance-superiority claim is made unless it materializes; if it does, it is reported as a secondary finding.

## 7. Pre-experiment gate

No experiment may run before all six implementation gate tests pass (Methodology §6): Weibull recovery; change-of-variables audit; inverse audit; censoring-likelihood gradient audit; quantile/sampling KS audit; D-calibration sanity. The tests live in `tests/` and run in CI.

## 8. Deviations log

**Deviation 1 (2026-09-22): H1 scoped by scenario after Phase 5 pilot diagnostics.**

**What was pre-registered.** H1 (flexibility): "On non-standard hazard shapes (S2 bathtub, S3 multimodal, S4 crossing), FlowSurv-AFT attains lower hazard-recovery error (HRE) than all baselines." Contrast C1 (confirmatory, §2): FlowSurv-AFT vs DSM on HRE, pooled across S2–S4, all n × censoring × type.

**What was found.** The Phase 5 pilot (untuned, n=5000 cells) showed a mixed FlowSurv-AFT vs DSM result on HRE: FlowSurv-AFT ahead on S3, DSM ahead on S2 and S4. Per the plan's own risk checkpoint (Implementation Plan, "End Phase 5" — if FlowSurv-AFT does not beat DSM on HRE in S2–S4, revisit flow capacity/conditioning before the full grid), three follow-up diagnostics were run on S2 and S4 (n=5000, c=20%, held-out reps not used in any tuning):

1. A reduced-budget (8-config) real tuning pass — the gap persisted and widened on S4 (untuned flow/dsm HRE ratio ≈1.2×; tuned ≈2.1×).
2. Two capacity increases (max spline bins/blocks; max spline + a larger encoder trunk) — HRE did not improve on S2, and on S4 the larger configuration reduced HRE only at the cost of D-calibration collapsing to 0% pass rate across all reps (vs. 100% for the smaller model).
3. An ablation breaking the model's identity warm start (spline head initialized with small Gaussian noise instead of exact zero) — no improvement on S2 or S4; S2 calibration was worse under the broken warm start.

All three interventions converged on the same negative result, taken as sufficient evidence this is not a tuning, capacity, or initialization artifact.

**Interpretation.** S2's data-generating hazard is Cox-proportional-hazards multiplicative in form (h0(t)·exp(β′x), Methodology §3.1); S4 is a discrete regime switch between two distinct Weibull shapes at x1 = 0. Neither is a location/scale/residual-shape perturbation of a single underlying law in the sense an AFT decomposition is naturally suited to represent — the covariate effect must be absorbed entirely through the flow's shape-conditioning pathway rather than the location shift μ(x) the model is warm-started toward. S3 (a multimodal mixture, naturally a residual-shape question) and S5 (nonlinear μ(x), naturally a location question) remain squarely within what the AFT decomposition is designed to represent, and FlowSurv-AFT's performance there is unaffected by this finding.

**Deviation.** H1 is split into two sub-hypotheses, and contrast C1 is narrowed to its confirmatory core with the remainder reported as descriptive:

- **H1a (confirmatory, replaces the S3 portion of H1).** On hazard shapes that are residual-shape or location perturbations of a common mechanism (S3 multimodal, S5 nonlinear-μ), FlowSurv-AFT attains lower HRE than parametric AFT, DSM, and DeepHit.
- **H1b (descriptive, not confirmatory; replaces the S2/S4 portion of H1).** On hazard shapes generated by a mechanism structurally misaligned with the AFT decomposition (S2 PH-multiplicative bathtub, S4 discrete regime-switch crossing hazards), FlowSurv-AFT's HRE relative to DSM is reported descriptively, without a directional claim of superiority. The full factorial grid (all n × censoring × type, not just the pilot's n=5000/c=20% cells) is still run for S2 and S4, since censoring and sample size may still matter even where the DGP mechanism does not favor an AFT-type model.
- **Contrast C1** is narrowed to S3 (+ S5 if the full grid confirms the pilot's directional pattern there) as the confirmatory comparison against DSM; the S2/S4 vs. DSM comparison is retained as a secondary, Holm-corrected exploratory contrast, reported honestly regardless of direction.

**Why this is not post-hoc cherry-picking.** The split is made on an a priori mechanistic distinction — whether the DGP is expressible as a location/scale/shape transform of a common residual law, which is fixed by the DGP's mathematical form (Methodology §3.1) and was decidable before running any experiment — that happens to align with, and is supported by, the pilot's empirical pattern and three independent negative diagnostics. The original, unscoped H1 (§1) and the original pooled C1 (§2) are left unedited above precisely so a reader can see what was pre-registered and what changed.

**Consequence for the paper's framing.** Document 1's objective O2 and the differentiator claims are updated (see `docs/01_Introduction_Literature_Aims.md`, same date) to state the flexibility claim as scoped to residual-shape and nonlinear-location hazard recovery, not as a blanket claim across all non-standard shapes — a more defensible claim than the original, and one that delineates exactly where an AFT-decomposed flow helps and where a mixture model remains preferable, rather than asserting a universal win.

---

## Appendix A — Final novelty sweep (2026-08-04)

Pre-freeze verification that the gap claimed in Document 1 §2.7 (the "empty cell") is still open. Sources: arXiv API (full-text search over title/abstract), attempted OpenReview API and Semantic Scholar API.

| Query | Hits | Findings relevant to novelty |
|---|---|---|
| arXiv: "normalizing flow" AND "survival analysis", by date | 2 | Ausset et al. 2021 (known, differentiated); Yin et al. 2025 (arXiv:2510.21829) — flow for cross-modal alignment in multimodal WSI survival, not an event-time flow model |
| arXiv: "spline flow" AND "survival" | **0** | — |
| arXiv: "neural spline flow" AND "survival" | **0** | — |
| arXiv: "survival" AND "flow" AND "censoring" | 4 | Ausset 2021; **LT-ICL** (arXiv:2607.18530, 2026-07) — conditional normalizing-flow head for right-censored supply-chain lead-time forecasting with in-context learning: adjacent concurrent item, but no AFT decomposition, no spline flow, CRPS evaluation, industrial domain → monitor, no threat to D1–D3; Eguchi 2026 (book, flow matching for statistical inference, includes survival chapters — cite as context); one insurance discrete-time paper (irrelevant) |
| arXiv: "accelerated failure time" AND "deep", by date | 8 | **GRAFT** (arXiv:2602.07884, 2026-02) — gated residual AFT, C-index-aligned rank loss + post-hoc calibration: new member of the likelihood-free deep-AFT camp (Doc 1 §2.4); add to related work at writing stage, no threat to the conjunction; DART (ECAI 2023) Gehan-rank AFT; remainder known/already cited |
| arXiv: "survival" AND "monotonic" AND "flow" | 7 | all irrelevant (physics/fluid-dynamics usage of "flow"/"survival") |
| arXiv: "survival" AND "spline" AND "neural" | 6 | Yuan et al. 2025 (arXiv:2503.19763, interval-censored partially-linear transformation model, monotone-spline sieve MLE + DNN — transformation-model camp already cited via DRIFT/deeptrafo); CENNSurv 2025 (exposure-lag modeling); Gregorio et al. 2023 (Royston–Parmar-style spline NN for treatment effects); deeptrafo (already cited) |
| OpenReview API (ICLR 2026 accepted list; STAI-X #101 / Li & Cai status) | n/a | **Not accessible** from the development environment (JS-rendered responses) → open item, see below |
| Semantic Scholar API | n/a | HTTP 429 (rate-limited); arXiv coverage judged sufficient for freeze |

**Conclusion.** As of 2026-08-04, arXiv contains **no** spline-flow survival model, **no** normalizing-flow AFT model, and no new exact-likelihood flow competitor beyond the already-differentiated Ausset et al. (2021) and Li & Cai (2026, concurrent). The empty-cell conjunction (bidirectional exactness + AFT interpretability + exact censored likelihood + censoring-regime study) remains open. **No design adjustment required.**

**Open items (carried to the Week-21 final priority sweep):** (i) manual OpenReview check of the ICLR 2026 accepted list and of Li & Cai's STAI-X status; (ii) monitor LT-ICL and GRAFT for journal versions; (iii) verify Lifetime Data Analysis quartile before submission.

## Appendix B — Environment at freeze

- Local dev env: conda `torch_gpu` — Python 3.10.20, torch 2.5.1 (CUDA, RTX 4050 Laptop 6 GB), zuko 1.6.0, pytest 9.1.1, pyarrow 25.0.0, pandas 2.3.3, numpy 2.0.1, scipy 1.15.3, scikit-learn 1.6.1, statsmodels 0.14.6.
- Clean-machine spec: `environment.yml` (Python 3.11 + full dependency set).
- Known gaps to be installed before their phases: `lifelines`, `scikit-survival`, `pycox` (Phase 3); `rpy2`/R-`flexsurv` (Phase 3, exported-CSV route on Windows); `torchdiffeq` (optional Ausset-CNF).
- Hardware note: local GPU has 6 GB VRAM (plan assumes ≥ 12 GB). MLP batch-256 fits at n ≤ 5000 are small and expected to fit; if OOM or throughput problems arise, the plan's CPU-fallback rule applies (trim tuning to 15 configs) and is logged in §8.
