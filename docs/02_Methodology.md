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

- **μ(x)** — acceleration surface (scalar). **Time-ratio interpretation:** for two covariate profiles xₐ, x_b, TR = exp(μ(xₐ) − μ(x_b)) multiplies the entire event-time distribution (all quantiles) — the exact analogue of the classical AFT acceleration factor.
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

where F₀, f₀ are the Logistic(0,1) CDF and density, themselves closed-form. **Concordance risk scores** (e.g., median lifetime Q(0.5|x), or −E[T|x] estimated from analytic quantiles) come out analytically — Ausset's thesis required Monte-Carlo sampling for the same quantity.

### 2.2.2 Identifiability and the role of the decomposition

The AFT form separates *where* the distribution lives (μ, σ — covariate-driven, interpretable) from *what shape* it has (the flow — a standardized residual law, weakly conditioned). To prevent the conditioner h(x) from silently absorbing μ: (i) the base law p₀ has fixed location/scale (Logistic(0,1)); (ii) spline ordinates are parameterized as offsets from the identity map; (iii) the μ-head is initialized so that at initialization g ≈ identity and the model starts as a **log-logistic AFT** — a principled warm start guaranteeing the classical model is nested inside the deep one. Ablation FlowSurv-Gauss (g fixed to identity, p₀ Gaussian) recovers the Neural-CET class and isolates the flow's contribution.

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
- Numerical: float32 with log-space likelihood; spline domain [−4, 4] in standardized residual units with identity tails; L2 penalty 1e-5 on spline derivatives (smoothness, discourages tail wiggle under heavy censoring).

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
| ML | Random Survival Forest (500 trees, tuned nodesize/mtry) |
| DL — discriminative | DeepSurv; DeepHit |
| DL — distributional | DSM (Weibull mixture, K ∈ {2,4,8} tuned) |
| Ablations (ours) | **FlowSurv-Gauss** (identity flow, Gaussian residual ≈ Neural CET); **FlowSurv-AFT** (full); optional reimplemented **Ausset-CNF** (continuous FFJORD flow) on a subset of cells to quantify the discrete-flow advantage in accuracy and wall-clock |

Identical tuning budget (30 random configs, inner validation) for every DL method; classical methods per package defaults with documented tuning where applicable.

## 3.3 Evaluation metrics (pre-registered formulas)

Let the test set be {(xⱼ, tⱼ, δⱼ)} and τ the 90th percentile of observed times.

- **Discrimination — Uno's C** (only concordance reported): C_U = [Σ_{i≠j} δᵢ Ĝ(tᵢ)⁻² 1{tᵢ < tⱼ} 1{rᵢ > rⱼ}] / [Σ δᵢ Ĝ(tᵢ)⁻² 1{tᵢ < tⱼ}], with Ĝ the IPCW censoring KM estimate and r the model risk score (negative expected lifetime from analytic quantiles).
- **Overall — Integrated Brier Score:** IBS = τ⁻¹ ∫₀^τ BS(t) dt, BS(t) = mean over subjects of IPCW-weighted squared error of Ŝ(t|x) vs observed status (Gerds & Schumacher 2006).
- **Calibration — D-calibration** (Haider et al. 2020): assign each subject to the decile of Ŝ(tᵢ|xᵢ) containing its outcome; χ² uniformity test on the 10 bins (report pass rate at α = 0.05 across replications) — plus **ICI** (Austin et al. 2020): mean absolute deviation of the loess calibration curve of observed vs predicted event probabilities at τ.
- **Hazard recovery (simulation only):** HRE = mean over test subjects of ∫₀^τ [ĥ(t|x) − h_true(t|x)]² w(t) dt on a fixed grid, w(t) ∝ marginal density of evaluation times. **Headline metric — no prior flexible-density paper reports it.**
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

10 repeated 80/20 train/test splits, stratified on δ; within each training fold, 15% validation for early stopping/tuning (same 30-config budget). Methods as in §3.2 (Ausset-CNF on 2 datasets for cost). Metrics: Uno's C, IBS, D-calibration, ICI (no hazard recovery — no ground truth). Reporting: median [IQR] over splits per dataset; paired comparisons via Wilcoxon signed-rank across splits with Holm correction. Optional: SEER (data-use agreement required — only if timeline permits).

## 4.3 What counts as success (pre-registered)

Real-data claims are restricted to calibration/distributional quality (per Burk et al. 2024): (i) D-calibration pass rate ≥ discriminative DL baselines, especially FLCHAIN; (ii) ICI and IBS non-inferior (within a pre-set margin) to the best baseline on every dataset; (iii) no claim of concordance superiority is made unless it materializes — in which case it is reported as a secondary finding.

---

# 5. Pillar C — Interpretability Analysis

1. **Time-ratio concordance:** on S1 (Weibull truth) and on real datasets, estimated TR = exp(μ(xₐ) − μ(x_b)) for each active covariate vs Weibull-AFT time ratios — scatter + concordance correlation coefficient; agreement where the classical model is adequate is the trust evidence for the deep one (H4).
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

- **Stack:** Python 3.11, PyTorch 2.x, `zuko` (RQS flows; `nflows` fallback), `pycox` (datasets, DeepSurv/DeepHit/DSM baselines), `scikit-survival` (RSF, metrics), `lifelines`; R (`flexsurv` for Royston–Parmar) via `rpy2` or exported CSVs. Config in YAML per cell; seeds per (cell, replication) derived from a master seed.
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
