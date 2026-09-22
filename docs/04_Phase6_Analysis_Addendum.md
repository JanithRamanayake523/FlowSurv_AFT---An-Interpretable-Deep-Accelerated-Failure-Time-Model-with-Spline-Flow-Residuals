# Phase 6 — Execution & Analysis Addendum

**Status:** operational addendum, not itself a deviation except where it explicitly invokes one (§2, §6). It pre-specifies exactly how the already-frozen `preregistration.md` §1–§7 rules, plus Deviations 1–3 (§8), apply to the full grid, before any Phase 6 result exists.

## 1. Cell list (unchanged from `default_grid()`)

120 cells: **S1–S5 × n ∈ {200, 1000, 5000} × censoring ∈ {0%, 20%, 50%, 80%} × type ∈ {I, III}** (`src/flowsurv/data/cells.py:default_grid`). No cells are dropped, including S2/S4 at all n × censoring × type combinations — per Deviation 1, S2/S4 remain in the full factorial because censoring and sample size may still matter even where the DGP mechanism does not favor an AFT-type model (H1b is descriptive, not absent).

Real-data arm: 5 datasets (FLCHAIN, GBSG, METABRIC, SUPPORT, WHAS) × 10 repeated 80/20 splits (preregistration.md §4 "Splits").

## 2. Method roster and replication counts

11 methods: `flowsurv_aft, flowsurv_gauss, cox_ph, weibull_aft, log_normal_aft, rsf, deepsurv, deephit, dsm, royston_parmar, ausset_cnf`.

| Replications | Methods |
|---|---|
| R = 100 (as pre-registered, §4) | flowsurv_aft, flowsurv_gauss, cox_ph, weibull_aft, log_normal_aft, rsf, deepsurv, deephit, dsm, royston_parmar |
| **R = 20** (Deviation 3, §8) | ausset_cnf |

Royston-Parmar and Ausset-CNF were added to the grid on the strength of the backfill pilot completed 2026-09-22: 100% convergence across all 20 pilot cells × 10 reps for both, no scenario-specific failure pattern.

Two implementation fixes are standing configuration for every Phase 6 fit:
- **Deviation 2** (§8): Cox-PH / Weibull-AFT / log-normal AFT fit with fixed `penalizer=0.01` by default, applied uniformly.
- **GPU device wiring** (`run_cell.py::_fit_config`, commit `52a269b`): Ausset-CNF receives the runner's requested device like the other `_FlowSurvWrapper` methods; previously it silently ran on CPU. Performance fix only, no deviation-log entry required — it does not change any method's fitted objective.

**Ausset-CNF replication reduction** (Deviation 3, §8): a post-fix GPU benchmark (n=5000, S1 vs S3) showed only ~1.0–2.2x speedup over the CPU numbers from the backfill pilot — the RK4 integration's cost is dominated by ~160 sequential small kernel launches per training pass (20 steps × 4 RK4 stages × velocity-plus-divergence), not matmul throughput, so it barely benefits from batching or a faster device. Projected full-grid cost at R=100 was ~265 GPU-hours (~11 days) for this one baseline alone. Ausset-CNF now runs at **R=20** (reps 1–20 of the same deterministic seed sequence as every other method); all confirmatory contrasts (C1–C3) are unaffected since Ausset-CNF is not a party to any of them.

## 3. Hypothesis → contrast → cell mapping (post–Deviation 1)

| Contrast | Hypothesis | Cells | Status |
|---|---|---|---|
| C1 (narrowed) | H1a | S3 (+ S5 if full grid confirms pilot's directional pattern) vs DSM, HRE | Confirmatory |
| C1 (secondary) | H1b | S2, S4 vs DSM, HRE, all n × censoring × type | Exploratory, Holm-corrected, reported regardless of direction |
| C2 | H2 | All scenarios, censoring ∈ {0,20,50,80}% | Confirmatory (unchanged) |
| C3 | H3 | S1 | Confirmatory (unchanged) |
| C4 | H4 | S1 + real datasets | Descriptive (unchanged) |

This table is the operational reading of preregistration.md §2 + §8 Deviation 1; it does not introduce new claims.

## 4. Tuning protocol (reaffirmed, §4)

30-config random search per DL method per macro-cell (a "macro-cell" = one (scenario, n, censoring-type) combination pooled across censoring levels, per Implementation Plan §3 `tune.py`), tuned on reps 1–5, frozen for reps 6–100 (reps 6–20 for Ausset-CNF, per its reduced R); nested full tuning on a random 10% of cells as the freeze-bias audit. Royston-Parmar and Ausset-CNF follow the same protocol as the other DL-classed methods (`is_deep=True` for `ausset_cnf`; Royston-Parmar remains `is_deep=False`, package-default df-selection only).

## 5. Analysis plan (reaffirmed, §5)

Per-cell median + 10,000-replicate bootstrap CI (computed from R=20 for Ausset-CNF, R=100 for every other method — CIs for Ausset-CNF are correspondingly wider, reported as-is); cross-cell linear mixed-effects model (method × scenario × n × censoring % × censoring type, random intercept for replication batch); Holm correction on every non-C1/C2/C3 comparison; failure/non-convergence rates reported per (method, cell), never silently averaged over.

## 6. Compute budget

Resolved via Deviation 3 (§8): Ausset-CNF at R=20 brings its full-grid cost to roughly 53 GPU-hours (~2.2 days), down from the ~265-hour R=100 projection. The other 10 methods are substantially cheaper per fit (seconds to low tens of seconds for classical/RSF/Royston-Parmar; well under Ausset-CNF's per-fit cost for the deep methods based on Phase 5 pilot timings) and are not separately budget-constrained.

## 7. Mid-Phase-6 checkpoint (reaffirmed, Implementation Plan)

If failure rate > 10% in any method class, pause and fix rather than averaging over silently broken fits (Implementation Plan, "Mid Phase 6" risk checkpoint) — unchanged.
