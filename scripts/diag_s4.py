"""Deviation 1 diagnostic (supervisor item 2): is the S4 loss an AFT-structure problem?

S4 is exactly a Weibull AFT (log T = log lam_g + W/k_g, W min-Gumbel independent of x).
FlowSurv-Gumbel (identity flow, Gumbel base, deep mu(x) and sigma(x)) represents it exactly
with NO flow conditioning. If it also loses to DSM on S4, the cause is optimization or the
discontinuity at x1 = 0, not the AFT structure. All methods at their class-default configs
(no tuning) so the comparison isolates model structure. S3 (non-AFT) and S1 for context.
Run from the repo root:  python scripts/diag_s4.py 5 cuda
Writes per-row results to experiments/diagnostics/diag_s4.parquet after every rep and prints
median metrics plus per-cell median HRE ratios to DSM at the end. Cited by the "Correction to
Deviation 1" in preregistration.md (seeds are deterministic; class-default configs, no tuning).
"""
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd

from flowsurv.data import default_grid
from flowsurv.eval.run_cell import run_sim_rep

OUT = Path(__file__).resolve().parents[1] / "experiments" / "diagnostics" / "diag_s4.parquet"
OUT.parent.mkdir(parents=True, exist_ok=True)
METHODS = ["flowsurv_aft", "flowsurv_gumbel", "flowsurv_gauss", "dsm"]
CELLS = ["S4_n5000_c20_typeI", "S4_n1000_c20_typeI", "S3_n5000_c20_typeI", "S1_n5000_c20_typeI"]
REPS = range(1, int(sys.argv[1]) + 1) if len(sys.argv) > 1 else range(1, 6)
DEVICE = sys.argv[2] if len(sys.argv) > 2 else "cuda"

cells = {c.cell_id: c for c in default_grid()}
rows = []
t0 = time.perf_counter()
for cid in CELLS:
    for rep in REPS:
        for m in METHODS:
            row = run_sim_rep(cells[cid], rep, m, tuner=None, device=DEVICE)
            rows.append(row)
        pd.DataFrame(rows).to_parquet(OUT, index=False)
        print(f"[{time.perf_counter() - t0:6.0f}s] {cid} rep {rep} done", flush=True)

df = pd.DataFrame(rows)
cols = ["hre", "hre_cum", "ks", "w1", "ibs", "ici", "dcal_pass", "unos_c"]
print(df.groupby(["cell_id", "method"])[cols].median().round(4).to_string())
for cid, sub_df in df.groupby("cell_id"):
    piv = sub_df.pivot(index="rep", columns="method", values="hre")
    print(cid, "median HRE ratio to DSM:", piv.drop(columns="dsm").div(piv["dsm"], axis=0).median().round(2).to_dict())
print("DONE", flush=True)
