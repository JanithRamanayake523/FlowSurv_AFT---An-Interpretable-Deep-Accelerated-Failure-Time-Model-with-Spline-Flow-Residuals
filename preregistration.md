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

**Correction to Deviation 1 (2026-09-24): the mechanistic rationale was wrong for S3 and S4, and the S4 loss did not reproduce.** The paragraphs above are left as originally written so the record shows what was claimed.

*What was wrong.* The "Interpretation" and "why this is not cherry-picking" paragraphs say S4 is not a location/scale/residual-shape perturbation and S3 is. Working through the DGPs (Methodology §3.1) shows the opposite. **S4 is exactly an AFT model:** log T = log lam_g + W/k_g with W a standard minimum-Gumbel variable independent of x, so x enters only through mu(x) and sigma(x) and FlowSurv can represent S4 without any flow conditioning. **S3 is not an AFT model:** its second mixture component (LN(2.5, 0.3)) does not move with x, so the standardized residual's shape changes with x and only the conditional flow can capture it. The split was therefore not decidable a priori on the stated ground.

*Evidence (diagnostic run 2026-09-24, corrected pipeline; script `scripts/diag_s4.py`, saved output `experiments/diagnostics/diag_s4.parquet`, 4 methods x 4 cells x 5 replications = 80 rows; the run was made before this entry was written and is reproducible with `python scripts/diag_s4.py 5 cuda`).* FlowSurv-AFT, FlowSurv-Gumbel (flow frozen at identity, minimum-Gumbel base, deep mu(x) and sigma(x): a Weibull AFT), FlowSurv-Gauss and DSM, all at class-default configurations (no tuning), 5 replications per cell, Type I, 20% censoring, median HRE ratio to DSM: **S4 n=5000:** FlowSurv-AFT 0.71, FlowSurv-Gumbel 0.40, FlowSurv-Gauss 1.15. **S4 n=1000:** 1.15, 1.21, 0.96. **S3 n=5000** (non-AFT): 0.20, 1.01, 0.87 (the flow wins; the identity-flow AFT models do not). **S1 n=5000** (H3 sanity): FlowSurv-AFT 6.4, FlowSurv-Gumbel 1.02, FlowSurv-Gauss 4.96. So on S4 at n=5000 the pilot's DSM advantage is absent (FlowSurv-AFT is ahead, the Weibull-AFT-with-deep-parameters model further ahead), S4 at n=1000 is a tie, S3 behaves as an x-dependent residual shape predicts, and on S1 the flow itself adds hazard error where the DGP needs none (bearing on H3). Limits: untuned, 5 replications, one censoring level and type, S2 not re-run, and the pilot-versus-now difference bundles several protocol changes (Deviation 5: validation split, spline bound and penalty, log-time centring; Deviation 4 does not touch HRE), so the pilot's S4 gap cannot be attributed to any one of them.

*Consequence.* The mechanistic argument for splitting H1 is withdrawn, and the pilot's S4 loss is not evidence that the AFT decomposition is misaligned with S4. The H1a/H1b split is not relied on: the original H1 and the pooled contrast C1 (S2-S4 vs DSM) will be evaluated and reported as pre-registered on the tuned full grid, with the S3-only and S3+S5 reading reported alongside as descriptive. The framing edits to `docs/01_Introduction_Literature_Aims.md` made under Deviation 1 (flexibility claim scoped to residual-shape and nonlinear-location recovery) should be revisited once the full-grid results are in.

**Deviation 2 (2026-09-22): fixed nonzero default penalizer for the lifelines classical baselines.**

**What was pre-registered.** Methodology §4/§6 specifies Cox-PH, Weibull-AFT, and log-normal AFT as baselines fit by maximum likelihood, with `penalizer` listed as a tunable hyperparameter (candidate grid {0.0, 0.01, 0.1}) but with no default value fixed for the untuned (`is_deep=False`) code path that all three classical methods run through.

**What was found.** In the Phase 5 pilot, the classical baselines' fit used `penalizer=0.0` (unregularized MLE) unconditionally, since the frozen-tuning gate (Implementation Plan Phase 4) only fires for `is_deep=True` methods. At small n and low censoring, several Weibull-AFT and Cox-PH fits produced extreme, numerically unstable hazard estimates (traced to the fitted survival curve underflowing to float zero across consecutive points on the evaluation grid, driving the predicted hazard to the `np.clip(h, 0, 1e3)` ceiling while the true hazard was on the order of 30). This is a known small-sample behavior of unregularized parametric/semiparametric MLE, not a metric or implementation bug.

**Deviation.** `src/flowsurv/baselines/classical.py` now applies a fixed default `penalizer=0.01` (module constant `_DEFAULT_PENALIZER`) to `CoxPH`, `WeibullAFT`, and `LogNormalAFT` whenever no explicit `penalizer` is supplied. The value is fixed a priori (the smallest nonzero value already present in the pre-registered tuning grid), applied identically across every cell, and is not selected or adjusted based on any Phase 5 or Phase 6 outcome. It remains overridable by an explicit `penalizer` kwarg, so nothing about the tuning-space contract for a future tuned run changes.

**Why this is not post-hoc cherry-picking.** The change is motivated purely by a numerical-stability failure mode identified in the pilot before any Phase 6 results existed, applies uniformly to all cells and all three affected methods regardless of scenario or direction of effect, and does not touch the deep-method frozen-tuning protocol (Deviation 1) or any evaluation metric. It is applied prospectively to Phase 6, not retroactively reinterpreted from Phase 6 results.

**Consequence for the paper's framing.** Classical-baseline HRE/C-index/IBS values reported from Phase 6 onward reflect a mildly regularized fit; this is noted in the Methods section as a deviation from unregularized textbook MLE, with the rationale given above. The Phase 5 pilot's classical-baseline numbers (already reported/archived) reflect the unregularized fit and are not retroactively recomputed; any pilot-vs-Phase-6 comparison for classical baselines should account for this.

**Correction to Deviation 2 (2026-09-24): the stated cause was wrong; the decision is kept for a different reason.** The paragraphs above are left as originally written so the record shows what was claimed.

*What was wrong.* "What was found" attributes the pilot's extreme hazard estimates to unregularized MLE being unstable at small n and low censoring, and says this was "not a metric or implementation bug". That diagnosis was incorrect. The classical baselines' `predict_hazard` differentiated a survival curve that had first been cast to float32 at the method interface. For subjects with very high hazard, late-time survival (~1e-30) is not representable in float32; the corruption produced spikes at the `1e3` clip ceiling that dominated HRE (Weibull-AFT, S1, n=5000, no censoring, rep 1: HRE 801.7 through the float32 path vs 0.225 in float64, same fitted model; Cox-PH 825.7 vs 0.947). This was an implementation bug, fixed in commit `d0ff297` (survival kept in float64, hazard clip removed). Separately, a few subjects in S5 have true hazards up to ~1e5 that every linear baseline misses equally, which dominates HRE in those cells whatever the penalty.

*Evidence (re-test on the fixed code, 2026-09-24).* 1,395 paired fits (Cox-PH, Weibull-AFT, log-normal AFT; penalizer 0 vs 0.01; S1-S5; Type I censoring; n=200 with 20 reps and 0/20/80% censoring, n=1000 with 10 reps and the same censoring, n=5000 with 3 reps at 0% censoring): zero fit failures at penalizer 0; IBS, Uno's C and ICI differ by at most 0.002 and the D-calibration pass rate by at most 0.004; the share of fits with HRE > 100 is identical with and without the penalty (5% / 11% / 20% at n=200 / 1000 / 5000, all in S5); median HRE at n=200 is 10-20% lower with the ridge (Cox-PH 4.04 to 3.49, Weibull-AFT 1.40 to 1.12, log-normal AFT 1.61 to 1.43) and identical at n=5000. Limits: simulated data only, Type I censoring only, few n=5000 replications.

*Decision.* The fixed default `penalizer=0.01` is retained unchanged (uniform across cells and the three methods, never tuned). Its justification is now: a mild ridge that modestly lowers small-sample HRE at n=200 at no cost elsewhere, and conservative with respect to the paper's claims because it gives the classical baselines their better small-n result. It is not a fix for fit instability, which was not observed.

*Consequence.* The claim above that the change was "motivated purely by a numerical-stability failure mode" no longer applies. The retention decision was made from this re-test before any of the affected methods were refit, applies uniformly, and is not selected per cell or per outcome. The Phase 6 rows of the six affected baselines (Cox-PH, Weibull-AFT, log-normal AFT, RSF, DeepSurv, DeepHit) were archived and are being refit with the fixed code and this default. The Phase 5 pilot numbers for these baselines (archived in `experiments/archive_pre_fix/`) reflect penalizer 0 and the float32 hazard and must not be used for comparison.

**Deviation 3 (2026-09-22): reduced replication count for Ausset-CNF (R=20, not R=100).**

**What was pre-registered.** §4 "Seeds" specifies R = 100 replications per cell, uniformly across methods, with per-cell medians and 10,000-replicate bootstrap CIs computed from those 100 replications (§5).

**What was found.** A GPU benchmark run after fixing the device-wiring gap (Ausset-CNF was previously fitting on CPU regardless of the requested device; see `run_cell.py::_fit_config`, commit `52a269b`) showed only a ~1.0-2.2x speedup over the CPU numbers observed in the Phase 5 backfill pilot, far short of what the fix was expected to deliver. The cause is architectural, not a further wiring bug: `ConditionalCNFFlow._integrate` (src/flowsurv/models/cnf_flow.py) is correctly batched across all subjects in a cell, but each training epoch's forward pass issues `n_steps=20` RK4 steps x 4 stages x (a velocity evaluation plus a separate `torch.autograd.grad` divergence call) = 160 sequential small kernel launches through a tiny 3-layer velocity network. This cost is dominated by per-launch latency, not matmul throughput, so it does not shrink much with a larger batch or a faster device. At the benchmarked per-fit cost, the full grid's 12,000 (cell x replication) combinations for this one method alone (120 cells x 100 reps) project to roughly 265 GPU-hours (~11 days) of continuous compute -- infeasible within the study's timeline alongside the other 10 methods.

**Deviation.** Ausset-CNF alone is run with **R = 20** replications per cell (reps 1-20 of the same deterministic `cell_seed` sequence used for every other method, so its subjects/censoring draws for those 20 reps are byte-identical to the corresponding reps of every other method — no separate randomization scheme). All other 10 methods (`flowsurv_aft`, `flowsurv_gauss`, `cox_ph`, `weibull_aft`, `log_normal_aft`, `rsf`, `deepsurv`, `deephit`, `dsm`, `royston_parmar`) keep R = 100 as pre-registered. Per-cell medians and bootstrap CIs for Ausset-CNF are computed from its 20 replications; this is noted wherever Ausset-CNF appears in a table or figure. Ausset-CNF is a secondary empirical-competitor baseline (the "numerical-solver" contrast point for FlowSurv-AFT's bidirectional-exactness claim, AGENTS.md differentiator 1) and is not a party to any confirmatory contrast (C1-C3, §2) or to H1-H4 directly — it appears only in descriptive/exploratory comparisons, where the reduced replication count widens its reported CIs but does not invalidate a confirmatory claim.

**Why this is not post-hoc cherry-picking.** The reduction is a compute-feasibility decision made from a wall-clock benchmark, before any Phase 6 result exists for Ausset-CNF at any cell, and it is fixed at R=20 for every cell uniformly -- not adjusted per scenario, per direction of effect, or in response to any interim finding. It follows the same reps 1-20-of-100 subsequence already used for the freeze-then-audit tuning protocol (§4), so no new seed logic is introduced.

**Consequence for the paper's framing.** Any table or figure reporting Ausset-CNF is annotated with "R=20" (vs. "R=100" for every other method), and its CIs are visibly wider as an honest consequence, not smoothed over. The Methods section documents the architectural cause (sequential-kernel-launch-bound CNF integration) as a limitation of the numerical-CNF baseline itself, which is, if anything, corroborating evidence for the paper's differentiator claim that FlowSurv-AFT's closed-form spline flow avoids exactly this per-evaluation solver cost.

**Deviation 4 (2026-09-24): metric estimators corrected after supervisor review (ICI, calibration slope, Uno's C, IBS, D-calibration).**

**What was pre-registered.** §3 lists ICI (Austin et al. 2020), calibration slope, Uno's C, IBS and D-calibration (Haider et al. 2020), with the observed outcome for ICI/slope defined as the event indicator at tau and Uno's C untruncated.

**What was found.** Evaluating each metric on the *true* survival function (S1, n = 1000, 20 reps; an oracle should score ICI ~ 0, slope ~ 1) showed the implementations, not the models, were biased:

| Censoring | ICI old | ICI new | Slope old | Slope new |
|---|---|---|---|---|
| 0% | 0.101 | 0.012 | 1.03 | 1.03 |
| 50%, Type III | 0.182 | 0.023 | 0.39 | 1.00 |
| 80%, Type III | 0.348 | 0.029 | 0.64 | 1.04 |

Two causes: (i) statsmodels `lowess` was called with its default robust iterations (`it=3`), which down-weight the minority class of a 0/1 outcome; (ii) the observed indicator `1{t <= tau, d = 1}` counts subjects censored before tau as event-free, a bias that grows with censoring. Separately, Uno's C had no tau truncation with G clamped at 1e-7 (weights up to 1e14 for events near the end of follow-up) and used G(t) rather than G(t-); IBS used G(t) for the event term; and D-calibration split censored mass 1/(b+1) per bin instead of proportionally to where S falls (Haider et al. Algorithm 1).

**Deviation.** (a) ICI uses plain loess (`it=0`) on Kaplan-Meier jackknife pseudo-observations of `1{T <= tau}`; (b) calibration slope uses an IPCW-weighted logistic regression; (c) Uno's C is truncated at tau (the same 90th percentile of observed test times used for IBS/ICI) and uses G(t-); IBS's event term uses G(t-); (d) D-calibration uses Haider's proportional censored mass. Hypotheses, contrasts, tau and the non-inferiority margins are unchanged. Regression tests: `tests/test_metrics_calibration.py`.

**Why this is not post-hoc cherry-picking.** The fixes are estimator-correctness changes verified against the true survival function and against scikit-survival's truncated Uno estimator, apply identically to every method, and were made before any Phase 6 result that used them is analysed. They are not selected on any method's performance.

**Consequence.** Every metrics row written before this change (all methods, pilot and Phase 6) carries the old ICI, calibration slope, Uno's C and IBS values and must be recomputed by refitting before analysis. Old D-calibration pass rates change little (oracle pass rates were unchanged in the check), but the chi-square statistic is stored and is the recommended summary under heavy censoring, where the oracle passes about 99-100% of the time under either version.

**Deviation 5 (2026-09-24): FlowSurv-AFT training protocol aligned with the pre-registration; ablations and diagnostics added after supervisor review.**

**What was pre-registered.** §4: spline domain [-4, 4]; L2 penalty 1e-5 on spline derivatives; a 24-point FlowSurv-AFT grid (K x bins x hidden x dropout); early stopping on validation NLL; simulation splits 70/15/15 identical across methods.

**What was found (implementation vs. §4).**
1. The FlowSurv wrapper ignored the validation split passed to `fit` and carved a further 15% out of the training data: FlowSurv trained on ~59.5% of n while DeepSurv/DeepHit/DSM trained on 70%, and at n = 200 it early-stopped on ~21 subjects. (Unfair to FlowSurv, and a fairness issue in either direction.)
2. The wrapper's spline bound was 6.0, not 4.
3. The L2 penalty on spline derivatives was not implemented.
4. The tuning grid additionally included the number of encoder residual blocks `n_blocks` in {1, 2, 3}. The pre-registered 24-point grid is smaller than the 30-configuration budget, so a 30-configuration random search over it was not possible (the search degenerated to the exhaustive grid); with `n_blocks` (72 points) a 30-of-72 random search is meaningful, and Methodology §2.1 already specifies a 2-3 block encoder.
5. Real-data times are in days (log t of about 7) while the identity warm start sits at mu = 0 (t = 1).

**Deviation / fix.** (1) The protocol validation split is passed through and used for early stopping (no second carve-out). (2) Bound set to 4 as pre-registered. (3) The penalty is implemented (1e-5 on the squared raw interior-derivative parameters, mean over the batch; identity-flow ablations have none). (4) `n_blocks` in {1, 2, 3} is **kept in the grid and logged here** (applies identically to FlowSurv-Gauss and Ausset-CNF, which tune the same encoder blocks; DSM/DeepHit/DeepSurv use their own space). (5) A fixed offset on mu equal to the mean training log time centres the warm start on the data scale; this leaves densities, quantiles and time ratios exact. Also added, none of which touch a pre-registered hypothesis or contrast: quantile-specific time ratios, an AFT-ness diagnostic, a strict-AFT ablation (`flowsurv_strict_aft`, unconditional flow and scale) for H4, a FlowSurv-Gumbel diagnostic (`flowsurv_gumbel`), an exploratory cumulative-hazard companion to HRE (`hre_cum`), and an explicit `paired=` mode replacing a shape heuristic that was ambiguous when a grid had exactly as many points as test subjects.

**Consequence.** All FlowSurv-AFT rows written before this change (the tuned S1 rows and the untuned pilot rows, which also carry the old metrics of Deviation 4) were produced under items 1-3 and 5 above and must not be mixed with post-change rows; they need to be regenerated before analysis. The new ablation methods are excluded from the default full-grid method list and run only when requested with `--methods`.

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
