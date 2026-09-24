# FlowSurv-AFT: An Interpretable Deep Accelerated Failure Time Model with Spline-Flow Residuals

## Document 2 — Complete Methodology

---

# 1. Research Design Overview

A **methods-development study** with three empirical pillars: (A) ground-truth **simulation study** establishing hazard-shape recovery, calibration, and censoring robustness across a full-factorial design grid; (B) **real-data validation** on five public clinical benchmarks; (C) **interpretability analysis** of the learned acceleration factors. The design follows the Pearce et al. (2022) benchmarking protocol (synthetic-with-synthetic-censoring plus real-with-real-censoring, repeated splits), reports metrics aligned with each claim per Lillelund et al. (2025), and pre-registers one concordance variant per Sonabend (2022). The simulation grid deliberately extends the classical design of Opatha & Jayasinghe (2024) so results are directly comparable with theirs.

---

# 2. Statistical Framework

## 2.1 Notation and assumptions

Observed data per subject i = 1…n: covariates xᵢ ∈ Rᵖ, observed time tᵢ = min(Tᵢ, Cᵢ), event indicator δᵢ = 1{Tᵢ ≤ Cᵢ}, where Tᵢ is the event time and Cᵢ the censoring time.

**Assumptions** (stated explicitly in the paper per Lillelund et al. 2025): (A1) conditional independent censoring, T ⊥ C | x; (A2) positivity — non-zero event and censoring probability across the support; (A3) i.i.d. observations given the covariate law. Dependent censoring is out of scope (see Document 1, §1.8).

**Estimand:** the full conditional law of T | x — equivalently the conditional density f(t|x), survival function S(t|x) = P(T > t | x), hazard h(t|x) = f(t|x)/S(t|x), and quantile function Q(q|x).

## 2.2 The FlowSurv-AFT model

Let Y = log T. The model decomposes Y into an interpretable location-scale form with a learned, shape-free residual law:

```
Y = μ(x) + σ(x) · ε,     ε = g⁻¹_θ(z ; h(x)),   z ~ p₀ = Logistic(0, 1)
```

- **μ(x)** — acceleration surface (scalar). **Time-ratio interpretation:** for two covariate profiles xₐ, x_b, TR = exp(μ(xₐ) − μ(x_b)) multiplies the entire event-time distribution (all quantiles) **only if the residual law does not depend on x** — the exact analogue of the classical AFT acceleration factor holds for the strict-AFT ablation (unconditional flow and scale, §2.2.2). For the conditional flow the exact, always-valid object is the quantile-specific ratio TR_q(a, b) = Q(q|xₐ)/Q(q|x_b), and exp(μ(xₐ) − μ(x_b)) is interpreted as a time ratio only to the extent TR_q is flat in q (AFT-ness diagnostic, §5).
- **σ(x) > 0** — learned scale, via softplus output.
- **g_θ(·; h(x))** — conditional **monotone rational-quadratic spline (RQS) transform** (Durkan et al. 2019), the forward map of the residual flow: B = 8–16 bins on a bounded domain [−L, L] (L = 4·σ₀ standardization, tails handled by linear extension outside the spline range, standard in NSF), with knot abscissae, ordinates, and interior derivatives output by the conditioner network from the embedding **h(x)**. Optionally K = 1–3 stacked RQS blocks.
- **μ, log σ, h(x)** — heads of a shared encoder: residual MLP (2–3 hidden layers of 64–128 units, GELU, LayerNorm, dropout 0.1–0.2).

**Why RQS specifically (the bidirectional-exactness argument):** g_θ is piecewise rational-quadratic and strictly monotone, so both g and g⁻¹ are **analytic** — as are g′ and all quantities below. Neither Ausset et al.'s continuous ODE flow (numerical IVP per evaluation) nor Li & Cai's tanh-sum monotone flow (no closed-form inverse; 40-step bisection per quantile/sample) has this property.

### 2.2.1 Exact closed-form outputs

Writing u(t) = (log t − μ(x))/σ(x) and g = g_θ(·; h(x)):

```
F(t|x) = F₀(g(u(t)))                                   exact CDF
S(t|x) = 1 − F₀(g(u(t)))                               exact survival
f(t|x) = f₀(g(u(t))) · |g′(u(t))| / (t · σ(x))         exact density (change of variables + exp Jacobian)
h(t|x) = f(t|x) / S(t|x)                               exact hazard
Q(q|x) = exp( μ(x) + σ(x) · g⁻¹(F₀⁻¹(q); h(x)) )       exact quantiles
sample:  z ~ p₀,  t = exp( μ(x) + σ(x) · g⁻¹(z; h(x)) )  one pass
```

where F₀, f₀ are the Logistic(0,1) CDF and density, themselves closed-form. **Concordance risk scores** (the risk score used throughout is the negative median lifetime −Q(0.5|x); the mean E[T|x] may be infinite or unstable under heavy right tails) come out analytically — Ausset's thesis required Monte-Carlo sampling for the same quantity.

### 2.2.2 Identifiability and the role of the decomposition

The AFT form separates *where* the distribution lives (μ, σ — covariate-driven, interpretable) from *what shape* it has (the flow — a standardized residual law, weakly conditioned). To prevent the conditioner h(x) from silently absorbing μ: (i) the base law p₀ has fixed location/scale (Logistic(0,1)); (ii) spline ordinates are parameterized as offsets from the identity map (identity at initialization, and shrunk toward it during training by the pre-registered L2 penalty on the spline derivative parameters; this is a soft regularizer, not a hard constraint); (iii) the μ-head is initialized so that at initialization g ≈ identity and the model starts as a **log-logistic AFT** — a principled warm start guaranteeing the classical model is nested inside the deep one. Ablation FlowSurv-Gauss (g fixed to identity, p₀ Gaussian) recovers the Neural-CET class and isolates the flow's contribution.

**Caveat (supervisor review, 2026-09-24).** Because the spline parameters depend on x, a conditional RQS can absorb part of any location shift inside [−L, L], so μ(x) is identified only softly (through the identity tails and the penalty), and exp(Δμ) is not in general a time ratio for the whole distribution. Three safeguards are therefore reported: (a) **quantile-specific time ratios** TR_q = Q(q|xₐ)/Q(q|x_b), exact and cheap because the quantile function is analytic (differentiator 1); (b) an **AFT-ness diagnostic**, the spread of log TR_q across q (≈ 0 where the DGP is a strict AFT, e.g. S1/S5; clearly positive on S3, whose second mixture component does not move with x); (c) a **strict-AFT ablation** (`flowsurv_strict_aft`: spline parameters *and* σ are global, so the residual law is one for all subjects and exp(Δμ) is exactly a time ratio), used for the time-ratio concordance of H4. A **FlowSurv-Gumbel** diagnostic (identity flow, minimum-Gumbel base, deep μ(x), σ(x): a Weibull AFT with deep parameters) separates AFT structure from optimization difficulty on S4 (Deviation 1, correction).

## 2.3 Estimation: censoring-aware full likelihood

Parameters θ (encoder + heads + spline conditioner) maximize the exact right-censored log-likelihood:

```
ℓ(θ) = Σᵢ [ δᵢ · log f(tᵢ|xᵢ ; θ) + (1 − δᵢ) · log S(tᵢ|xᵢ ; θ) ]
```

All terms are evaluated in log-space (log-density directly; log S via log1p/exp formulations) — essential for stability at 80% censoring, where surviving terms dominate.

**Interval censoring (secondary experiment):** subject known only to fail in (lᵢ, uᵢ] contributes log[ S(lᵢ|xᵢ) − S(uᵢ|xᵢ) ] — one extra exact CDF call; included to widen the contribution (ODE flows and bisection flows pay heavily for this term; discrete-time models cannot express it continuously).

**Optional distribution-matching regularizer (ablation):** the Soft–Nelson–Aalen Wasserstein term of Li & Cai (2026), R(θ) = W₁(Ĥ_model, Ĥ_NA) between the cohort cumulative hazard implied by model samples (one-pass, so cheap) and the empirical Nelson–Aalen curve; total loss L = −ℓ + λR with λ ∈ {0, 0.1, 0.2, 0.5} ablated, λ = 0 pre-registered as primary.

**No IPCW anywhere in training** — the exact S term removes the weighting that destabilizes likelihood-free deep AFTs under heavy censoring.

## 2.4 Optimization protocol (pre-registered)

- Optimizer: AdamW, lr 1e-3, cosine decay to 1e-5, weight decay 1e-4; batch 256; ≤ 500 epochs; early stopping on validation NLL, patience 30; gradient clip 1.0.
- Splits: train/val/test = 70/15/15 within each replication (simulation); outer repeated splits for real data (§4.3).
- Hyperparameter selection: validation NLL over the grid {K ∈ 1–3 RQS blocks} × {bins ∈ 8, 16} × {hidden ∈ 64, 128} × {dropout ∈ 0.1, 0.2} — 30 random configurations, identical budget for every DL baseline (fairness constraint).
- Tuning economy: within each simulation macro-cell (scenario × n × censoring), hyperparameters are tuned on the first 5 replications, then frozen for the remaining 95 — stated in the pre-registration; full nested tuning on a 10% audit subset confirms the freezing bias is negligible.
- Numerical: float32 with log-space likelihood; spline domain [−4, 4] in standardized residual units with identity tails; L2 penalty 1e-5 on the raw spline interior-derivative parameters (smoothness toward the identity, discourages tail wiggle under heavy censoring). Log observed times are centred inside the wrapper (a fixed offset on μ equal to the mean log time of the training data, so the identity warm start sits on the data scale; densities/quantiles stay exact in real time and time ratios are unaffected). Early stopping uses the validation split supplied by the protocol (70/15/15 in simulation), identical to every other deep method.

---

# 3. Pillar A — Simulation Study

## 3.1 Data-generating processes (explicit forms)

All scenarios use p = 10 covariates, x ~ N(0, I), 5 active with β = (0.8, −0.6, 0.5, −0.4, 0.3) on the location unless stated. Event times generated via the inversion method from the stated conditional hazard (equivalent to `simsurv` user-defined hazard; validated against it).

| ID | Name | Data-generating mechanism | Purpose |
|---|---|---|---|
| S1 | Weibull sanity | h(t\|x) = (k/λ)(t/λ)^(k−1)·exp(β′x), k = 1.5, λ = 1 | Model must match parametric AFT — flexibility penalty check (H3) |
| S2 | Bathtub | h₀(t) = 0.8·exp(−1.5t) + 0.15·t, times exp(β′x) | Early high risk → dip → late rise; Weibull AFT structurally incapable (H1) |
| S3 | Multimodal | T \| x ~ 0.6·LogNormal(0.5 + β′x, 0.4) + 0.4·LogNormal(2.5, 0.3) | Two-hump density/hazard (cf. Ausset thesis synthetic); unimodal mixtures struggle (H1) |
| S4 | Crossing hazards | Group A (x₁ > 0): Weibull k = 0.7; Group B: Weibull k = 2.2, λ = 1.3 — hazards cross near the median lifetime | PH violation; Cox/DeepSurv structurally fail; tests shape + interaction recovery (H1) |
| S5 | Nonlinear μ | S1 hazard with μ(x) = 0.8·sin(2x₁) − 0.6·x₂x₃ + 0.5·x₄² | Separates encoder flexibility (μ) from flow flexibility (shape); linear-AFT fails, FlowSurv-Gauss partially succeeds |
| S6 | Semi-synthetic | Real covariates (SUPPORT, GBSG) + synthetic event times from a Weibull-frailty DGP (borrowed from Li & Cai's protocol) | Known ground truth with realistic x — bridge to real data |

**Censoring mechanisms** (per cell, calibrated numerically to hit the target proportion):

- **Type I (administrative):** Cᵢ = c fixed, c = the (1−cens%) quantile of the marginal event-time distribution.
- **Type III (random):** Cᵢ ~ Uniform(0, u), u solved per cell by bisection so E[P(T > C)] = cens%.

**Design factors:** n ∈ {200, 1000, 5000} × censoring ∈ {0%, 20%, 50%, 80%} × type ∈ {I, III} × scenario {S1–S5} (S6 follows Li & Cai's published settings) → 5 × 3 × 4 × 2 = **120 cells**, R = **100 replicated datasets per cell** (12,000 datasets), common random seeds across methods within a replication.

## 3.2 Competing methods (pre-registered)

| Class | Methods |
|---|---|
| Classical | Cox PH; Weibull AFT; log-normal AFT; Royston–Parmar flexible parametric (M-splines, df = 3–5 tuned) |
| ML | Random Survival Forest (500 trees, `min_samples_split`/`min_samples_leaf` at package defaults, not tuned — see rationale below) |
| DL — discriminative | DeepSurv; DeepHit |
| DL — distributional | DSM (Weibull mixture, K ∈ {2,4,8} tuned) |
| Ablations (ours) | **FlowSurv-Gauss** (identity flow, Gaussian residual ≈ Neural CET); **FlowSurv-AFT** (full); optional reimplemented **Ausset-CNF** (continuous FFJORD flow) on a subset of cells to quantify the discrete-flow advantage in accuracy and wall-clock |

Identical tuning budget (30 random configs, inner validation) for every DL method; classical methods per package defaults with documented tuning where applicable.

**Why classical methods are not put through the same 30-config search.** This is a deliberate asymmetry, not an oversight, and follows the convention of the methods whose original papers we compare against (DeepSurv, DeepHit, DSM all tune their own network but report classical baselines at standard/default settings). Three reasons: (1) the fairness constraint that matters for the confirmatory hypotheses (H1–H3) is *identical tuning budget across DL methods*, since FlowSurv-AFT is compared against DSM/DeepHit/DeepSurv, not against Cox or RSF, under H1–H3; (2) a 30-config search selected on only 5 replications' validation folds is already a noisy signal (small-sample max-over-30-noisy-estimates) — extending that same noisy selection to classical methods with few genuinely impactful hyperparameters would not meaningfully improve them and would not change the confirmatory comparisons; (3) it keeps the classical baselines representative of how they are used in standard practice, which is the fairer comparison point for a methods paper (a reader asking "how does this compare to what practitioners actually run" is better served by package defaults than by an unusually well-tuned Cox model). The one classical exception is Royston–Parmar, whose df ∈ {3,4,5} spline-flexibility choice is a built-in model-selection step internal to the method itself (via validation NLL), not part of the external grid-tuning system — analogous to how a package's own cross-validated regularization path would be used as-is rather than re-implemented. Cox-PH/Weibull-AFT/log-normal-AFT additionally use a small fixed (not tuned, not searched) regularization default — see preregistration.md Deviation 2 — a mild ridge that slightly lowers small-sample HRE (n=200) at no cost elsewhere, retained after a re-test on the fixed code showed that unregularized fits do not fail (the instability originally cited was a float32 hazard-computation bug, since fixed; see the correction under Deviation 2); the value was chosen once, not selected by validation score, and applies uniformly.

## 3.3 Evaluation metrics (pre-registered formulas)

Let the test set be {(xⱼ, tⱼ, δⱼ)} and τ the 90th percentile of observed times.

- **Discrimination — Uno's C** (only concordance reported): C_U = [Σ_{i≠j} δᵢ Ĝ(tᵢ−)⁻² 1{tᵢ < tⱼ} 1{tᵢ ≤ τ} 1{rᵢ > rⱼ}] / [Σ δᵢ Ĝ(tᵢ−)⁻² 1{tᵢ < tⱼ} 1{tᵢ ≤ τ}], with Ĝ the IPCW censoring KM estimate (Ĝ(t−) its left limit), truncated at τ as in Uno et al. (2011) so that events near the end of follow-up cannot dominate, and r the model risk score (negative median lifetime from the analytic quantile function).
- **Overall — Integrated Brier Score:** IBS = τ⁻¹ ∫₀^τ BS(t) dt, BS(t) = mean over subjects of IPCW-weighted squared error of Ŝ(t|x) vs observed status (Gerds & Schumacher 2006; Ĝ(tᵢ−) for the event term).
- **Calibration — D-calibration** (Haider et al. 2020): assign each subject to the decile of Ŝ(tᵢ|xᵢ) containing its outcome, a censored subject spreading its unit mass over the bins below Ŝ(tᵢ|xᵢ) in proportion to their width and its own bin by the remainder (Haider et al. Algorithm 1); χ² uniformity test on the 10 bins. Report the pass rate at α = 0.05 **and the χ² statistic**: under heavy censoring the fractional mass flattens the histogram and even the true model passes ≈ 99–100% of the time, so the pass rate alone is compressed toward 1 — plus **ICI** (Austin et al. 2020): mean absolute deviation of the (non-robust, `it = 0`) loess calibration curve of predicted event probability F̂(τ|x) against the Kaplan–Meier jackknife pseudo-observation of 1{T ≤ τ} (the naive indicator 1{t ≤ τ, δ = 1} is biased under censoring). The calibration slope is the IPCW-weighted logistic slope. Oracle check (true survival, S1, n = 1000): ICI ≈ 0.01–0.03 at 0–80% censoring.
- **Hazard recovery (simulation only):** HRE = mean over test subjects of ∫₀^τ [ĥ(t|x) − h_true(t|x)]² w(t) dt on a fixed grid, w(t) ∝ marginal density of evaluation times. **Headline metric — no prior flexible-density paper reports it.** Because HRE is on the squared-hazard scale a handful of extreme-hazard subjects can dominate a cell (S5 reaches ~1e5; S4 with k = 0.7 diverges as t → 0), an exploratory scale-robust companion is stored alongside it (`hre_cum` = mean over subjects of ∫|Ĥ − H_true| w dt on the cumulative-hazard scale); it is never used for a confirmatory claim.
- **Distributional fidelity (simulation only):** KS and W₁ distances between estimated and true conditional CDFs at fixed test covariate profiles; quartile-wise signed CDF deviation as bias diagnostic.
- **Cost:** wall-clock for training, evaluation, and 1,000-sample generation per subject (the bidirectional-exactness claim quantified).

## 3.4 Statistical analysis of simulation results

Per cell: median metric over replications with 10,000-replicate bootstrap percentile CIs. Across cells: linear mixed-effects models per metric with fixed effects (method × scenario × n × censoring% × censoring type, two-way interactions of method × each design factor) and a random intercept for replication batch — directly paralleling, and extending, the analysis structure of Opatha & Jayasinghe (2024). Primary contrasts (pre-registered): FlowSurv-AFT vs DSM on S2–S4 (H1); FlowSurv-AFT calibration slope across censoring% vs DeepHit/DeepSurv (H2); FlowSurv-AFT vs Weibull AFT on S1 (H3).

---

# 4. Pillar B — Real-Data Validation

## 4.1 Datasets

| Dataset | n | Censoring | Domain | Source/preprocessing |
|---|---|---|---|---|
| SUPPORT | 8,873 | ~32% | hospitalized seriously ill | `pycox` standard |
| METABRIC | 1,904 | ~42% | breast cancer | `pycox` standard |
| GBSG (Rotterdam & GBSG) | 2,232 | ~43% | breast cancer | `pycox` standard |
| FLCHAIN | 7,874 | ~72% | free light chain assay | `pycox` standard — the heavy-censoring test |
| WHAS | 1,638 | ~57% | myocardial infarction | DeepSurv-repo version |

## 4.2 Protocol

10 repeated 80/20 train/test splits, stratified on δ; within each training fold, 15% validation for early stopping/tuning (same 30-config budget). Covariates are standardized on the full dataset before splitting (common pycox practice; a mild information leak about the test-fold mean/scale, applied identically to every method). Methods as in §3.2 (Ausset-CNF on 2 datasets for cost). Metrics: Uno's C, IBS, D-calibration, ICI (no hazard recovery — no ground truth). Reporting: median [IQR] over splits per dataset; paired comparisons via Wilcoxon signed-rank across splits with Holm correction. Optional: SEER (data-use agreement required — only if timeline permits).

## 4.3 What counts as success (pre-registered)

Real-data claims are restricted to calibration/distributional quality (per Burk et al. 2024): (i) D-calibration pass rate ≥ discriminative DL baselines, especially FLCHAIN; (ii) ICI and IBS non-inferior (within a pre-set margin) to the best baseline on every dataset; (iii) no claim of concordance superiority is made unless it materializes — in which case it is reported as a secondary finding.

---

# 5. Pillar C — Interpretability Analysis

1. **Time-ratio concordance:** on S1 (Weibull truth) and on real datasets, estimated TR = exp(μ(xₐ) − μ(x_b)) for each active covariate vs Weibull-AFT time ratios — scatter + concordance correlation coefficient; agreement where the classical model is adequate is the trust evidence for the deep one (H4).
1b. **Quantile time ratios and AFT-ness:** TR_q(a, b) = Q(q|xₐ)/Q(q|x_b) for q ∈ {0.1, 0.25, 0.5, 0.75, 0.9}, and the spread of log TR_q across q per covariate (flat on S1/S5, varying on S3); H4 concordance is also reported for the strict-AFT ablation, where exp(Δμ) is exactly a time ratio.
2. **Nonlinear acceleration effects:** partial-dependence/ICE curves of μ(x) on S5 and on continuous clinical covariates (e.g., age in SUPPORT) — shows what a linear AFT misses.
3. **Case-study densities:** 3–4 individual conditional density/hazard curves from METABRIC and SUPPORT where FlowSurv-AFT infers a non-standard shape (early-risk hump, bimodality) that DSM/parametric models smooth away — the qualitative figure for the paper.

---

# 6. Implementation Verification (pre-experiment gate)

Unit tests that must pass before any experiment runs:

1. **Weibull recovery:** fit FlowSurv-AFT on large-n S1 data with the flow frozen near identity → recover β within Monte-Carlo error of the true values; likelihood ≥ true-model likelihood − ε.
2. **Change-of-variables audit:** numerically integrate f̂(t|x) over t — equals 1 within 1e-4 for random x; Ŝ(t) monotone non-increasing; Ŝ(0) ≈ 1.
3. **Inverse audit:** ‖g⁻¹(g(u)) − u‖∞ < 1e-6 on the spline domain.
4. **Censoring likelihood audit:** on data with δ = 0 everywhere, gradient equals the score of S only (finite-difference check).
5. **Quantile/sampling audit:** KS test between 10⁵ one-pass samples and the analytic CDF — p > 0.01.
6. **Calibration sanity:** on large uncensored S1 samples, D-calibration passes.

---

# 7. Software, Compute, Reproducibility

- **Stack:** Python 3.11, PyTorch 2.x, `zuko` (RQS flows; `nflows` fallback), `pycox` (datasets, DeepSurv/DeepHit/DSM baselines), `scikit-survival` (RSF, metrics), `lifelines`. Royston–Parmar is a native Python reimplementation (restricted cubic spline basis + L-BFGS MLE) rather than R's `flexsurv` via `rpy2` — avoids the R/rpy2 dependency entirely; same restricted-cubic-spline model. Config in YAML per cell; seeds per (cell, replication) derived from a master seed.
- **Compute:** ~12,000 simulation datasets × ~10 methods. DL fits with frozen-then-audit tuning (§2.4): ≈ 120 cells × 5 tuning-reps × 30 configs + 12,000 × final fits. Estimated 3,000–6,000 GPU-hours on a single consumer GPU — feasible over 4–6 weeks; classical/ML fits are CPU and trivial. Checkpointing per cell; grid executable in embarrassingly parallel shards.
- **Reproducibility:** public GitHub repo (code + configs + pre-registration + simulation meta-dataset), `conda` lock file, arXiv preprint timestamping priority before journal submission.

---

# 8. Threats to Validity and Mitigations

| Threat | Mitigation |
|---|---|
| Construct: hazard L2 error rewards the true family | Baselines include the true parametric family per scenario (oracle check); conclusions drawn from relative ordering, not absolute values |
| Internal: tuning-budget asymmetry | Identical 30-config budget for all DL methods; audit nested tuning on 10% of cells |
| External: DGPs are stylized | S6 semi-synthetic bridge + 5 real datasets; claims worded per regime, not globally |
| Statistical: multiple comparisons across 120 cells | Pre-registered primary contrasts (§3.4); Holm correction elsewhere; effect sizes with CIs, not p-value fishing |
| Priority: concurrent work (Li & Cai; possible ICLR 2026 item) | Pre-submission sweep (final week); arXiv timestamp; our D2/D3 axes verified absent from all known concurrent work |

---

# 9. Deliverables Mapped to Objectives

| Objective | Methodology section | Deliverable |
|---|---|---|
| O1 Method | §2, §6 | Model + verified implementation (tests 1–6) |
| O2 Flexibility | §3.1–3.3 | HRE/KS/W1 tables & figures for S1–S6 |
| O3 Censoring map | §3.2–3.4 | Performance maps + mixed-effects analysis across 120 cells |
| O4 Calibration | §3.3, §4 | D-cal/ICI comparison across censoring levels, FLCHAIN headline |
| O5 Interpretation | §5 | Time-ratio concordance, PDP/ICE, case-study densities |
