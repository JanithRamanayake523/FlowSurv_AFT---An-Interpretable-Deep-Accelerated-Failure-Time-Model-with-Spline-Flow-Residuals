# FlowSurv-AFT: An Interpretable Deep Accelerated Failure Time Model with Spline-Flow Residuals

## Document 1 — Introduction, Literature Review, Research Aim and Objectives

---

# 1. Introduction

## 1.1 Background and motivation

Time-to-event (survival) analysis concerns data where the outcome is the time until an event of interest — death, relapse, machine failure, loan default — and its defining complication is **censoring**: for a substantial fraction of subjects the event has not occurred by the end of follow-up, so only a lower bound on their event time is known. Any useful survival model must therefore do two things at once: (i) handle censored observations correctly in estimation, and (ii) capture the shape of the event-time distribution, whose **hazard function** may increase (aging), decrease (infant mortality), follow a **bathtub** shape (high early risk, low middle, high late risk — standard in reliability and oncology), be **multimodal** (mixed patient populations), or cross between groups (treatments whose effect reverses over time).

Two classical families dominate practice. The **Cox proportional hazards (PH)** model is semiparametric and robust, but assumes proportional hazards and provides no smooth event-time distribution. **Accelerated failure time (AFT)** models assume a parametric event-time law (Weibull, log-normal) but reward the practitioner with a fully probabilistic model and a uniquely intuitive effect measure — the **time ratio** (acceleration factor), which states how much a covariate stretches or shrinks expected lifetime. Their weakness is the parametric assumption itself: a Weibull hazard is monotone, so a Weibull AFT is structurally incapable of representing bathtub or multimodal risks — precisely the shapes that arise in real reliability and medical data.

Deep learning has entered survival analysis from two directions, and both leave something behind. **Discriminative deep models** (DeepSurv, DeepHit, PC-Hazard, SurvTRACE) achieve strong risk ranking but either inherit the PH assumption or discretize time into a grid, producing step-function survival curves with no smooth density. **Deep AFT variants** (deepAFT, DeepR-AFT, KAN-AFT) modernize the AFT's covariate mapping with neural networks but abandon the likelihood — estimating by Buckley–James iteration, inverse-probability weighting, or rank losses — and consequently forfeit the very thing that made the AFT attractive: a smooth, calibrated conditional event-time distribution. Meanwhile the first **generative** approaches (conditional normalizing flows, mixture density networks, deep parametric mixtures) deliver smooth densities but, as shown in the literature review, each gives up at least one of: exact tractability, interpretability, or demonstrated fidelity to non-standard hazard shapes.

## 1.2 Problem statement

There is currently no survival model that simultaneously provides:

1. an **arbitrary, learned event-time distribution** (bathtub, multimodal, crossing hazards included) with a **smooth density**;
2. **exact, closed-form** density, survival, quantile, and sampling functions — no ODE solvers, no time discretization, no numerical inversion;
3. the **interpretable acceleration-factor decomposition** that makes the AFT model the preferred effect-communication tool in clinical and reliability applications;
4. estimation by **exact full likelihood** under censoring — no inverse-probability weighting, no partial likelihood, no iterative imputation;
5. **evidence** — missing across the entire literature — of how such models behave as censoring proportion and hazard shape vary, in terms of modern calibration metrics rather than concordance alone.

## 1.3 Motivating context

This research extends a line of inquiry opened by Opatha & Jayasinghe (ISC 2024, Sri Lanka), who showed in a careful simulation study that the choice between Cox PH and Weibull AFT depends systematically on **sample size, censoring proportion, censoring type, and hazard pattern** — with the AFT model dominating for small samples and increasing hazards. Their study was limited to the two classical models and to Harrell's concordance. The natural next question — *where do deep learning models sit on this map, and can a deep AFT model be built that keeps the AFT's interpretability while removing its parametric shape restriction?* — is unanswered, and is the question this research takes up.

## 1.4 Research aim

**To develop FlowSurv-AFT — a deep accelerated failure time model whose residual distribution is a conditional rational-quadratic spline normalizing flow, yielding an interpretable, exactly tractable, hazard-shape-free survival model — and to establish, through the first systematic hazard-shape-recovery and calibration study across censoring regimes, when and why it should be preferred over classical, machine-learning, and deep survival models.**

## 1.5 Research objectives

**O1 (Methodological).** Design and implement FlowSurv-AFT: log T = μ(x) + σ(x)·ε with (μ, σ, flow conditioner) given by a neural encoder and the residual law p_ε parameterized by a conditional rational-quadratic spline flow, trained by exact censoring-aware full likelihood — delivering closed-form f(t|x), S(t|x), h(t|x), quantiles, and one-pass sampling.

**O2 (Empirical — flexibility).** Demonstrate, on ground-truth simulations (bathtub, multimodal, crossing-hazard, nonlinear-effect scenarios), that FlowSurv-AFT recovers conditional hazard shapes that parametric AFT models, mixture-based deep models (DSM), and discrete-time deep models (DeepHit) cannot, quantified by true-vs-estimated hazard L2 error and distributional fidelity (KS/W1).

**O3 (Empirical — censoring robustness).** Map model performance (discrimination, calibration, overall accuracy) across a full factorial grid of sample size × censoring proportion (0–80%) × censoring type (Type I/III) × hazard scenario, extending the Opatha–Jayasinghe design to deep learning methods and identifying the conditions under which each method class fails.

**O4 (Empirical — calibration).** Establish, with D-calibration and the integrated calibration index, whether exact-likelihood flow models produce better-calibrated individual survival distributions than discriminative deep models — particularly under heavy (≥50%) censoring.

**O5 (Interpretation and practice).** Show that FlowSurv-AFT's acceleration factors recover the interpretability of the classical AFT (concordant time ratios on real data) while capturing nonlinear covariate effects, and validate the full pipeline on five public clinical benchmark datasets.

## 1.6 Research questions

- **RQ1.** Can a spline-flow residual distribution trained by exact censored likelihood recover bathtub, multimodal, and crossing conditional hazard shapes with fidelity approaching the true data-generating mechanism? (→ O1, O2)
- **RQ2.** How do discrimination, calibration, and hazard-recovery error of classical, ML, and DL survival models vary with sample size, censoring proportion/type, and hazard shape — and where are the crossover regions where deep flexible models pay off? (→ O3)
- **RQ3.** Does exact-likelihood training yield better-calibrated survival distributions than partial-likelihood or discrete-time training, and does this advantage grow with the censoring proportion? (→ O4)
- **RQ4.** Do FlowSurv-AFT's learned acceleration factors agree with classical AFT time ratios where the classical model is adequate, and what additional nonlinear effects do they reveal where it is not? (→ O5)

## 1.7 Significance and expected contributions

1. **The first survival model that is simultaneously shape-free, exactly tractable in both directions, and AFT-interpretable.** The two nearest works each occupy half this space: Ausset et al. (2021) use continuous ODE flows (numerical integration for every evaluation, no interpretability); Li & Cai (2026, concurrent) use discrete monotone flows (exact density but 40-step numerical bisection for every quantile or sample, no interpretability). Rational-quadratic spline flows are analytically invertible both ways, and the explicit μ(x)/σ(x) decomposition restores the time-ratio semantics clinicians use.
2. **The first systematic hazard-shape-recovery and calibration study across censoring regimes** — extending a peer-reviewed classical simulation design (ISC 2024) into deep learning, with ground-truth hazard error and modern calibration instruments. No flexible-density survival paper currently benchmarks hazard recovery at all.
3. **Practical guidance and open tooling**: a reproducible benchmark (code, configs, meta-dataset) and evidence-based recommendations on when flexible deep survival models are worth their cost — calibrated to the finding of neutral benchmarks that on standard tabular data, discrimination alone rarely justifies them.

## 1.8 Scope and limitations

- Right censoring (Types I/III) is the primary setting; interval censoring is included as a secondary experiment; dependent/informative censoring is **out of scope** (addressed by recent copula-based work; see §2.5.4) and stated as an explicit assumption.
- The study targets tabular covariates of moderate dimension; image/text covariates and competing risks are future work.
- Claims about real-data predictive superiority are deliberately limited to calibration and distributional quality, in line with neutral-benchmark evidence that discrimination gains on standard tabular benchmarks are marginal.

---

# 2. Literature Review

## 2.1 Classical survival models

The Cox (1972) proportional hazards model estimates covariate effects via partial likelihood without specifying the baseline hazard; it is the field's workhorse but assumes proportional hazards and yields only step-function (Breslow) survival estimates. Parametric AFT models (Wei 1992) specify log T = β'x + σε with ε following a standard distribution (Weibull/extreme-value, log-normal, log-logistic), giving smooth densities and time-ratio interpretations at the cost of shape constraints: Weibull hazards are monotone, log-normal hazards are hump-shaped, and neither can produce bathtub or multimodal forms. **Royston–Parmar flexible parametric models** relax this with restricted cubic splines for the log cumulative hazard — the strongest classical shape-flexible baseline, included in all experiments. Opatha & Jayasinghe (2024) mapped Cox-vs-AFT performance across sample size, censoring proportion/type, and hazard shape, providing both the motivating design and the classical anchor for this study.

## 2.2 Machine-learning survival models

Random Survival Forests (Ishwaran et al. 2008) and gradient-boosted Cox models handle nonlinearity and interactions without a distribution assumption, but produce no smooth conditional density (RSF predicts a Nelson–Aalen-type ensemble curve), offer limited calibration guarantees, and — critically for this study — large-scale neutral comparisons (Burk et al. 2024, 19 models × 34 datasets) find **no method, ML included, significantly outperforms Cox PH in discrimination on standard tabular data**. This reality shapes the contribution: the prize is not C-index points but calibrated, shape-faithful distributions.

## 2.3 Deep learning survival models

### 2.3.1 Cox-family deep models
DeepSurv (Katzman et al. 2018, *BMC MRM*) replaces the Cox linear predictor with a neural network, retaining the PH assumption; Cox-Time (Kvamme et al. 2019, *JMLR*) adds time–covariate interactions to escape proportionality, but still factorizes hazard into a shared Breslow baseline scaled by covariates — step-function curves, partial (not full) likelihood.

### 2.3.2 Discrete-time distribution models
DeepHit (Lee et al. 2018, *AAAI*) puts a softmax over a fixed time grid (PMF per subject, competing risks native); DRSA (2019, *AAAI*) predicts conditional event probabilities autoregressively; PC-Hazard (Kvamme & Borgan 2021, *Lifetime Data Analysis*) parameterizes piecewise-constant interval hazards; SurvTRACE (Wang & Sun 2022) uses a transformer encoder with discrete-hazard heads. All are shape-free **within grid resolution** but produce discontinuous densities, require grid-placement choices that trade bias against variance, and give quantiles only up to grid spacing — the "discretization horn."

### 2.3.3 Continuous-time flexible models
SODEN (Tang et al. 2022, *JMLR*) models survival through an ODE system generalizing DeepSurv/DeepHit — flexible but slow, numerically delicate, with an implicitly-defined density. SuMo-net (Rindt et al. 2022, *AISTATS*) parameterizes S(t|x) directly with a monotone network trained by proper scoring rules — smooth curves, but the density/hazard exists only through (noisy) differentiation.

## 2.4 Deep accelerated failure time models

The AFT's modern revival is real but incomplete. **deepAFT** (Norman et al. 2024, *Statistics in Medicine*) puts a DNN on the location μ(x) with an unspecified error, estimated by Buckley–James iteration, IPCW weighting, or data transformation — **likelihood-free**, yielding KM-rescaled step-function survival curves and no conditional density; IPCW is known to destabilize under heavy censoring. **DeepR-AFT** (Kim & Kang 2022) trains with a Gehan-type rank loss — ranking only, no distributions. **KAN-AFT** (Francis & Kattumannil 2025) swaps the DNN for a Kolmogorov–Arnold network to gain symbolic formulas, same likelihood-free estimation. **Neural CET** (Engelhard et al. 2020) is the closest in spirit: NN-parameterized location *and* scale trained by exact censored likelihood — but with a **fixed Gaussian residual**, i.e., a log-normal AFT with neural covariate maps; it is the natural ablation of the present work. DKAFT (Wu et al. 2021) uses a Gaussian-process output layer — again log-normal. Ranganath et al. (2016) parameterize Weibull event times through a deep exponential family — parametric shape constraint. **Summary: every published deep AFT either keeps the likelihood but fixes the residual family (Gaussian/Weibull), or frees the residual but abandons the likelihood.** The conjunction — free residual *and* exact likelihood *and* AFT form — is open.

## 2.5 Generative and flow-based survival models

### 2.5.1 Mixture-based densities
**Deep Survival Machines** (Nagpal et al. 2022, *IEEE JBHI*) parameterizes a mixture of K Weibull or log-normal components with an MLP — smooth densities, exact censored likelihood, the standard fully-parametric baseline. **DeepWeiSurv/DPWTE** (Bennis et al. 2020/2021) are precursors. **Survival MDN** (Han et al. 2022, *UAI*) uses Gaussian mixtures on transformed time with a change of variables. Shared limitation, stated by the MDN authors themselves: mixture primitives are unimodal with monotone or hump-shaped hazards, so bathtub/multimodal hazards require many components and K is a fragile hyperparameter.

### 2.5.2 Adversarial generators
Chapfuwa et al. (2018, *ICML*; 2021 survival-function matching) and Zhou et al. (2022) generate event times GAN-style — arbitrary shapes in principle, but no tractable likelihood, no closed-form survival function, unstable training.

### 2.5.3 Normalizing flows for survival — the direct lineage
- **Miscouridou et al. (2018, *MLHC*)** first used a (discrete) invertible transform of a Weibull base for survival, but inside a latent-variable model trained by black-box variational inference — an ELBO, not the exact censored likelihood.
- **Ausset et al. (2021, *IEEE DSAA*; PhD thesis, IP Paris)** introduced conditional **continuous** normalizing flows (FFJORD/Neural-ODE) on log survival time with the exact right-censored likelihood, remarking that the construction "can be seen as a generalization of the AFT model." Verified against the thesis: every density and survival evaluation requires numerical ODE integration; the thesis *reviews* discrete coupling/autoregressive flows and explicitly rejects them; experiments are small, use one multimodal synthetic at a single 80% censoring level, and report **Harrell's C only** — no hazard-shape recovery, no calibration analysis, no censoring sweep, and no AFT decomposition or time-ratio interpretation (the AFT link is a one-sentence framing remark).
- **Li & Cai (2026, STAI-X submission — concurrent unpublished work)** build **discrete monotone tanh-sum flows** on log-time with exact forward-only density and survival in the likelihood, plus a Soft–Nelson–Aalen Wasserstein regularizer and a shared-latent extension to informative censoring. Verified against the manuscript: their transform has **no closed-form inverse** — every quantile or sample requires 40-step numerical bisection; the map is not of location-scale form (knots depend on x), so no AFT interpretation; their simulations use standard distribution families only, with no bathtub/multimodal/crossing hazards, no calibration metrics, and no controlled censoring sweep.
- **Adjacent transformation-model work** (DRIFT, Kolb/Kook et al. 2024 *UAI*; deep conditional transformation models, Baumann et al. 2021) establishes the equivalence between 1-D conditional flows and flexible distributional regression with censored likelihoods — statistically the same object viewed from statistics; included in related work.

### 2.5.4 Dependent censoring
Gharari et al. (2023, *UAI*) address dependent censoring with deep copulas and identifiability guarantees — explicitly out of scope here, but the reference point if reviewers ask.

## 2.6 Evaluation of survival models

Modern evaluation practice supplies both the instruments and the traps. **Discrimination**: Harrell's C is ubiquitous but censoring-dependent; Uno's C (Uno et al. 2011) corrects this; Sonabend (2022, *Bioinformatics*) warns against "C-hacking" (reporting the friendliest of many variants) — hence one pre-specified variant. **Overall accuracy**: the Integrated Brier Score (Graf et al. 1999) with IPCW (Gerds & Schumacher 2006). **Calibration**: D-calibration and 1-calibration (Haider et al. 2020, *JMLR*), the integrated calibration index (Austin et al. 2020, *Statistics in Medicine*), X-CAL (Goldstein et al. 2020). **Position papers**: Lillelund et al. (2025, "Stop Chasing the C-index") demand metric–objective alignment and explicit censoring assumptions; Pearce et al. (2022) define the current simulation-rigor protocol (synthetic-with-synthetic-censoring + real-with-real-censoring, repeated splits). Burk et al. (2024) supply the empirical caution: on standard tabular benchmarks, discrimination differences among 19 methods are statistically negligible — flexible-density claims must be won on calibration and shape fidelity, not concordance.

## 2.7 Synthesis: the research gap

The literature separates cleanly into four camps, each surrendering one requirement:

| Camp | Smooth shape-free density | Exact f, S, quantiles, sampling (no solver/grid) | AFT interpretability | Exact censored likelihood | Shape/calibration study across censoring |
|---|---|---|---|---|---|
| Classical AFT / Royston–Parmar | ✗ (restricted shapes) | ✓ | ✓ | ✓ | partial (Opatha & Jayasinghe 2024 — classical models only) |
| Discrete-time DL (DeepHit family) | ✗ (grid steps) | ✗ | ✗ | ✓ (discrete) | ✗ |
| Deep AFT (deepAFT/DeepR/KAN) | ✗ (no density) | ✗ | ✓ | ✗ | ✗ |
| Generative (DSM/MDN; Ausset; Li & Cai) | ✓ / partial | ✗ (mixture-shape limits; ODE solver; bisection inverse) | ✗ | ✓ | ✗ |

**The empty cell is the conjunction.** FlowSurv-AFT occupies it by combining (i) an explicit AFT location-scale decomposition with (ii) a conditional rational-quadratic spline residual flow — analytically invertible in both directions — trained by (iii) the exact censoring-aware full likelihood, and validates it with (iv) the first hazard-shape-recovery and calibration study across censoring regimes.

---

# 3. Research Aim and Objectives (formal statement)

**Aim.** To develop and validate FlowSurv-AFT, an interpretable deep accelerated failure time model with a spline-flow residual distribution, and to establish its advantages in hazard-shape fidelity, calibration, and censoring robustness over classical, machine-learning, and deep survival models.

| Objective | Deliverable | Success criterion |
|---|---|---|
| O1 Method | FlowSurv-AFT model + open implementation | Exact closed-form f/S/h/quantiles verified against analytic cases (Weibull recovery test) |
| O2 Flexibility | Simulation evidence (S1–S6) | Hazard L2 error and KS/W1 significantly below DSM/DeepHit/parametric AFT on S2–S4; parity on S1 |
| O3 Censoring map | Full-factorial benchmark across censoring regimes | Published performance maps (per method × scenario × censoring) with mixed-effects analysis |
| O4 Calibration | Calibration comparison under heavy censoring | D-calibration pass rate and ICI superior to discriminative DL at ≥50% censoring |
| O5 Interpretation + real data | Real-data validation on 5 public datasets | Time ratios concordant with classical AFT where adequate; calibrated distributions (D-cal) on real data |

**Hypotheses.** H1: On non-standard hazard shapes, FlowSurv-AFT attains lower hazard-recovery error than all baselines. H2: The calibration advantage of exact-likelihood training over discrete-time/partial-likelihood deep models increases with censoring proportion. H3: On standard-shaped (Weibull-like) data, FlowSurv-AFT matches parametric AFT and does not pay a meaningful flexibility penalty. H4: FlowSurv-AFT's acceleration factors are concordant with classical AFT time ratios on data where the classical model is adequate.

---

## References (Documents 1–3 share this list)

- Opatha O.M.D.N.W. & Jayasinghe C.L. (2024). *Investigating the Impact of Censoring Proportion and Hazard Patterns on the Performance of Traditional Survival Analysis Approaches.* Proc. ISC 2024, Sri Lanka, p. 55.
- Ausset G., Ciffreo T., Portier F., Clémençon S., Papin T. (2021). *Individual Survival Curves with Conditional Normalizing Flows.* IEEE DSAA. arXiv:2107.12825. + Ausset G. (2021). PhD thesis, Institut Polytechnique de Paris. HAL tel-03582658.
- Li Y. & Cai Z. (2026). *A Deep Generative Framework for Right-Censored Survival Data with Distribution-Matching Regularization.* STAI-X 2026 submission (OpenReview #101, concurrent work).
- Miscouridou X. et al. (2018). *Deep Survival Analysis: Nonparametrics and Missingness.* PMLR v85 (MLHC).
- Norman P. et al. (2024). *deepAFT: A nonlinear accelerated failure time model with artificial neural network.* Statistics in Medicine 43(19). — Kim & Kang (2022). DeepR-AFT. arXiv:2206.05974. — Francis & Kattumannil (2025). KAN-AFT. arXiv:2512.20305.
- Engelhard M. et al. (2020). *Neural Conditional Event Time Models.* PMLR v126 (ML4H). — Wu et al. (2021). DKAFT. PMLR v149.
- Nagpal C., Li X., Dubrawski A. (2022). *Deep Survival Machines.* IEEE JBHI. arXiv:2003.01176. — Nagpal et al. (2021). Deep Cox Mixtures. PMLR v149. — Bennis et al. (2020/2021). DeepWeiSurv/DPWTE.
- Han X., Goldstein M., Ranganath R. (2022). *Survival Mixture Density Networks.* PMLR v182 (UAI).
- Katzman J. et al. (2018). *DeepSurv.* BMC Medical Research Methodology 18:24. — Kvamme H., Borgan Ø., Scheel I. (2019). *Cox-Time.* JMLR 20(129). — Kvamme H. & Borgan Ø. (2021). *PC-Hazard.* Lifetime Data Analysis 27:710–736.
- Lee C. et al. (2018). *DeepHit.* AAAI-18. — Ren K. et al. (2019). *DRSA.* AAAI-19. — Wang Z. & Sun J. (2022). *SurvTRACE.* EMBC.
- Tang W. et al. (2022). *SODEN.* JMLR 23(34). — Rindt D. et al. (2022). *SuMo-net.* AISTATS (PMLR v151). — Kolb/Kook et al. (2024). *DRIFT.* UAI. arXiv:2405.05429. — Baumann et al. (2021). Deep conditional transformation models. ECML.
- Chapfuwa P. et al. (2018). *Adversarial Time-to-Event Modeling.* ICML (PMLR v80); (2021). *Survival Function Matching.* NeurIPS. arXiv:1905.08838.
- Gharari et al. (2023). *Deep Copula-Based Survival Analysis for Dependent Censoring.* UAI (PMLR v216).
- Ishwaran H. et al. (2008). *Random Survival Forests.* Annals of Applied Statistics. — Cox D.R. (1972). *Regression models and life-tables.* JRSS-B. — Wei L.J. (1992). *The accelerated failure time model: a useful alternative to the Cox regression model in survival analysis.* Statistics in Medicine. — Royston P. & Parmar M. (2002). *Flexible parametric proportional-hazards and proportional-odds models.* Statistics in Medicine.
- Haider H. et al. (2020). *Effective Ways to Build and Evaluate Individual Survival Distributions.* JMLR 21(85). — Austin P., Harrell F., van Klaveren D. (2020). *Graphical calibration curves (ICI).* Statistics in Medicine 39:2714. — Uno H. et al. (2011). *On the C-statistics for evaluating overall adequacy of risk prediction procedures with censored survival data.* Statistics in Medicine. — Graf E. et al. (1999). *Assessment and comparison of prognostic classification schemes (IBS).* Statistics in Medicine. — Gerds T. & Schumacher M. (2006). *Consistent estimation of the expected Brier score.* Biometrical Journal.
- Sonabend R. (2022). *Avoiding C-hacking.* Bioinformatics 38(17). — Lillelund C. et al. (2025). *Stop Chasing the C-index.* arXiv:2506.02075. — Pearce T. et al. (2022). *A benchmarking protocol for survival models.* arXiv:2205.13496. — Burk L. et al. (2024). *A large-scale neutral comparison study of survival models.* arXiv:2406.04098.
- Durkan C. et al. (2019). *Neural Spline Flows.* NeurIPS. — Wiegrebe S. et al. (2024). *Deep Learning for Survival Analysis: A Review.* arXiv:2305.14961. — Drysdale E. (2022). *SurvSet.* arXiv:2203.03094.
