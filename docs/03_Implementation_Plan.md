# FlowSurv-AFT — Document 3: Step-by-Step Implementation Plan

**Companion to:** Document 1 (Introduction/Literature/Aims) and Document 2 (Methodology). This plan is executable: every phase has concrete tasks, commands, artifacts, and an exit checklist. Total duration ≈ 22 weeks.

---

## Repository layout (created in Phase 0)

```
flowsurv/
├── README.md                  # project overview + reproduction instructions
├── environment.yml            # conda lock
├── preregistration.md         # hypotheses, contrasts, metrics, budgets (frozen Week 2)
├── configs/                   # one YAML per experiment cell
│   ├── sim/S1_weibull_n200_c0_typeI.yaml  ... (120 cells)
│   └── real/support.yaml ...
├── src/flowsurv/
│   ├── models/
│   │   ├── encoder.py         # shared MLP -> mu, log_sigma, conditioner embedding
│   │   ├── rqs_flow.py        # conditional monotone RQS transform (zuko wrapper)
│   │   ├── flowsurv.py        # FlowSurvAFT: f, S, h, quantile, sample (exact)
│   │   └── losses.py          # censored NLL (+ interval), optional Soft-NA regularizer
│   ├── data/
│   │   ├── dgp.py             # S1–S6 generators (inversion method)
│   │   ├── censoring.py       # Type I / Type III, calibrated to target proportion
│   │   └── real.py            # SUPPORT/METABRIC/GBSG/FLCHAIN/WHAS loaders
│   ├── baselines/
│   │   ├── classical.py       # Cox, Weibull/log-normal AFT (lifelines), RSF (sksurv)
│   │   ├── royston_parmar.py  # flexsurv via rpy2 bridge
│   │   ├── deep.py            # DeepSurv, DeepHit, DSM (pycox), common tuning harness
│   │   └── ausset_cnf.py      # optional FFJORD reimplementation (torchdiffeq)
│   ├── metrics/
│   │   ├── discrimination.py  # Uno's C
│   │   ├── brier.py           # IBS (IPCW)
│   │   ├── calibration.py     # D-calibration, ICI
│   │   ├── recovery.py        # hazard L2 error, KS/W1 fidelity
│   │   └── cost.py            # wall-clock audit
│   ├── eval/
│   │   ├── tune.py            # 30-config random search, frozen-then-audit
│   │   ├── run_cell.py        # one (cell, replication, method) -> metrics row
│   │   ├── grid.py            # shardable driver over cells x reps x methods
│   │   └── analyze.py         # bootstrap CIs, mixed-effects models, figures
├── tests/                     # the 6 pre-experiment gate tests (Methodology §6)
├── experiments/               # outputs: metrics.parquet, checkpoints, figures
└── notebooks/                 # exploration only — nothing here is reported
```

---

## PHASE 0 — Environment, verification sweep, pre-registration (Weeks 1–2)

**Tasks**
1. `conda env create -f environment.yml` — python 3.11, pytorch, zuko, nflows, pycox, scikit-survival, lifelines, rpy2 (+ R with `flexsurv`, `randomForestSRC`), pandas, pyarrow, matplotlib, pytest.
2. Final literature sweep (the last open items): ICLR 2026 accepted-poster list for any Cox+AFT+CNF item; arXiv/Scholar search "spline flow survival", "normalizing flow survival 2025/2026"; record findings in `preregistration.md` appendix.
3. Write and freeze `preregistration.md`: hypotheses H1–H4, primary contrasts (Methodology §3.4), metric list, tuning budget, freezing protocol, success criteria §4.3.
4. Create repo from the layout above; CI running pytest.

**Exit checklist**
- [ ] environment reproduces on a clean machine
- [ ] sweep documented; no new novelty threat found (or design adjusted)
- [ ] pre-registration frozen and timestamped (e.g., OSF or repo tag `prereg-v1`)

---

## PHASE 1 — Core model (Weeks 3–6)

**Tasks (in order)**
1. `rqs_flow.py`: wrap zuko's monotonic RQS transform — forward, inverse, log|det| — conditioned on embedding h(x). Confirm zuko's spline matches Durkan et al. (2019) semantics (bins, tails = identity outside bound).
2. `encoder.py`: residual MLP → (μ, log σ, h). Softplus for σ. Initialize final layers to zero so g ≈ identity at start (log-logistic warm start, Methodology §2.2.2).
3. `flowsurv.py`: implement the six exact outputs (f, S, h, quantile, sample, risk score) in log-space per the formulas in Methodology §2.2.1.
4. `losses.py`: right-censored NLL; interval-censored NLL; optional Soft-NA Wasserstein term.
5. **Gate tests** (`tests/`, Methodology §6): Weibull recovery; change-of-variables audit; inverse audit; censoring-likelihood gradient audit; quantile/sampling KS audit; D-calibration sanity. **All six must pass before proceeding.**
6. Smoke-train on one S1 cell (n=1000, 20% censored): confirm val-NLL decreases, curves look Weibull.

**Exit checklist**
- [ ] 6/6 gate tests green in CI
- [ ] S1 smoke fit recovers β within MC error; survival curves visually correct
- [ ] inference micro-benchmark logged (f, S, quantile, sample wall-clock per 1k subjects)

---

## PHASE 2 — Simulation framework (Weeks 7–8)

**Tasks**
1. `dgp.py`: S1–S5 generators per Methodology §3.1 (inversion method from explicit hazards; cross-check S1 against `simsurv` output for identical parameters). S6: SUPPORT/GBSG covariates + Weibull-frailty times.
2. `censoring.py`: Type I (fixed quantile) and Type III (Uniform(0,u)) censoring with per-cell numerical calibration to target proportion (bisection on a pilot sample); write achieved censoring % into the cell manifest (tolerance ±1%).
3. Seed machinery: master seed → deterministic per-(cell, replication) seeds; dataset cache as parquet.
4. Dry-run every cell once: assert n, censoring %, event-rate distribution plausible; plot 2 example hazard curves per scenario against truth.

**Exit checklist**
- [ ] all 120 cells generate; achieved censoring within ±1% of target
- [ ] S1 cross-check vs simsurv matches (KS p > 0.05 on pooled samples)
- [ ] seed determinism: same config → byte-identical dataset

---

## PHASE 3 — Baselines and tuning harness (Weeks 8–10, overlaps Phase 2)

**Tasks**
1. `classical.py`: Cox PH, Weibull AFT, log-normal AFT (lifelines); RSF (scikit-survival); `royston_parmar.py` via rpy2/flexsurv.
2. `deep.py`: pycox DeepSurv, DeepHit, DSM behind a **common interface** (`fit / predict_surv / predict_density`) so all methods share the runner; DeepHit grid set per pycox defaults and documented.
3. `tune.py`: 30-config random search on validation NLL (or package-equivalent objective), identical budget across DL methods; frozen-then-audit logic (tune on reps 1–5, freeze for 6–100; nested tuning on 10% of cells as audit).
4. Validate baselines on S1: Cox/Weibull-AFT should be near-oracle — sanity anchor.

**Exit checklist**
- [ ] every method runs through `run_cell.py` and emits the full metric set
- [ ] S1 replication 1: Weibull AFT ≈ oracle likelihood; DSM/DeepHit plausible
- [ ] tuning audit protocol implemented and logged

---

## PHASE 4 — Metrics (Weeks 9–10, parallel)

**Tasks**
1. `discrimination.py`: Uno's C with IPCW weights (implement + test against scikit-survival's concordance on synthetic cases where both defined).
2. `brier.py`: IBS on grid [0, τ], τ = 90th percentile observed time; test against `sksurv.metrics.integrated_brier_score`.
3. `calibration.py`: D-calibration (10-bin χ², report statistic + pass/fail at 0.05) and ICI (loess at τ).
4. `recovery.py`: hazard L2 error on a fixed time grid (needs each method's hazard: analytic for ours, differentiation for smooth baselines, grid hazards for DeepHit — documented per method); KS/W1 conditional-CDF fidelity at fixed profiles.
5. `cost.py`: wall-clock decorators for fit/eval/sample.

**Exit checklist**
- [ ] each metric has a unit test against a reference implementation or hand-computed case
- [ ] metrics handle edge cases: all-censored test fold, Ŝ = 0/1 exactly (log guards)

---

## PHASE 5 — Pilot study (Week 11)

**Tasks**
1. Run a **reduced grid**: 5 scenarios × n ∈ {200, 5000} × censoring ∈ {20%, 80%} × Type I × 10 replications × all methods.
2. Inspect: convergence rates, failure modes (NaNs, non-convergence per method — log them), runtime per method (reality-check the compute budget), metric distributions.
3. Fix what breaks; adjust spline bounds/bins if tail pathologies appear; update compute estimate.

**Exit checklist**
- [ ] ≥ 95% of (method, cell, rep) fits complete without error
- [ ] FlowSurv-AFT beats Weibull AFT on HRE in S2–S4 pilot and matches it in S1 (H1/H3 directional check)
- [ ] compute projection for the full grid ≤ budget (else trim configs/cells with justification logged)

---

## PHASE 6 — Full simulation grid (Weeks 12–15)

**Tasks**
1. Shard `grid.py` runs (e.g., by scenario) across available GPU/CPU; checkpoint after every (cell, rep); idempotent reruns.
2. Nightly monitoring: completion counts, failure logs; requeue failures once, then record as failures (reported per §2.8 of Methodology — failure rates are a finding).
3. On completion: `analyze.py` — bootstrap CIs per cell; mixed-effects models; pre-registered contrasts; figure set F1–F5 (performance maps: method × censoring × scenario heat tables; hazard-recovery curves; calibration-vs-censoring lines; wall-clock bars).

**Exit checklist**
- [ ] 12,000 × 10 (method, dataset) results in `metrics.parquet` (minus logged failures)
- [ ] H1–H3 evaluated with CIs; results consistent with pilot
- [ ] raw artifacts archived (checksum manifest)

---

## PHASE 7 — Real-data experiments (Weeks 16–17)

**Tasks**
1. `real.py` loaders with `pycox` standard preprocessing; document covariate lists.
2. 10 repeated 80/20 stratified splits × 5 datasets × all methods (Ausset-CNF on SUPPORT + METABRIC only).
3. Metrics: Uno's C, IBS, D-cal, ICI; Wilcoxon + Holm across splits.
4. Interpretability outputs (Methodology §5): time-ratio concordance vs Weibull AFT; μ(x) PDP/ICE for age-type covariates; 3–4 case-study densities.

**Exit checklist**
- [ ] success criteria §4.3 evaluated and honestly reported (calibration focus)
- [ ] FLCHAIN heavy-censoring comparison complete
- [ ] interpretability figure set drafted

---

## PHASE 8 — Analysis consolidation and writing (Weeks 18–22)

**Tasks**
1. Freeze all tables/figures from `metrics.parquet` (no manual re-computation — everything regenerated by `analyze.py`).
2. Write the paper: Intro + Lit Review from Document 1; Methods from Document 2; Results from Phases 6–7; Discussion framed around the four-camp table (Doc 1 §2.7) and limitations (scope, dependent censoring).
3. Polish repo: README reproduction walkthrough (`make reproduce` → runs one full cell end-to-end), license, citation file.
4. Post arXiv preprint (priority timestamp) → submit to first-choice venue (Methodology §2.10: Lifetime Data Analysis if quartile verified, else Computational Statistics / Journal of Applied Statistics).

**Exit checklist**
- [ ] every number in the paper traceable to `metrics.parquet` via script
- [ ] pre-registration deviations (if any) listed in an appendix
- [ ] arXiv posted; submission receipt

---

## Master timeline

| Weeks | Phase | Milestone |
|---|---|---|
| 1–2 | 0 | environment + pre-registration frozen |
| 3–6 | 1 | model passes all 6 gate tests |
| 7–8 | 2 | 120 cells generate, censoring calibrated |
| 8–10 | 3–4 | baselines + metrics through common runner |
| 11 | 5 | pilot confirms direction + compute budget |
| 12–15 | 6 | full grid complete, analysis run |
| 16–17 | 7 | real data + interpretability complete |
| 18–22 | 8 | paper written, arXiv, submission |

## Compute and hardware notes

- One consumer GPU (≥ 12 GB) suffices: DL fit ≈ 1–3 min at n ≤ 5000, batch 256; the frozen-tuning protocol caps total DL fits at ≈ 120×5×30 (tuning) + 12,000 (final) ≈ 30k fits ≈ 1,500–3,000 GPU-hours with parallelism ~4 → 3–5 weeks wall-clock. CPU-only alternative: double it and trim tuning to 15 configs (log the deviation).
- Classical/ML baselines are CPU-trivial; run them on CPU cores in parallel with GPU jobs.
- If a cluster is available, shard by scenario — the grid is embarrassingly parallel.

## Risk checkpoints (stop-and-review moments)

- **End Phase 1:** if gate test 1 (Weibull recovery) fails after debugging, the warm-start/architecture has a bug — do not proceed on faith.
- **End Phase 5:** if FlowSurv-AFT does not beat DSM on HRE in S2–S4 pilot, revisit flow capacity/conditioning *before* burning the full grid.
- **Mid Phase 6 (week 13):** if failure rate > 10% in any method class, pause and fix; do not average over silently broken fits.
- **Week 21:** final priority sweep (post-2026 preprints); if a competitor appeared, reposition framing before submission — the D2 (AFT interpretability) and D3 (censoring-regime study) axes are the fallback claims, verified absent from all known work.
