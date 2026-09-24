"""H3 flexibility check on S1: does the RQS flow cost anything where the DGP is a plain Weibull AFT?

Compares FlowSurv-AFT (logistic base + conditional spline flow) with FlowSurv-Gumbel (identity flow,
minimum-Gumbel base = Weibull AFT with deep mu(x), sigma(x)), DSM and Weibull-AFT, all TUNED through the
real grid machinery (30 random configs on reps 1-5, frozen for reps 6+). Decides whether a minimum-Gumbel
base for FlowSurv-AFT is worth a logged deviation (supervisor review 2, section 2).

Produce the data (from the repo root; ~1 h on one GPU):

    python -m flowsurv.eval.grid --sim --scenario S1 --n 1000 5000 --censoring 20 --ctype I \\
        --methods flowsurv_aft flowsurv_gumbel dsm weibull_aft --reps 1-8 --device cuda \\
        --out experiments/diagnostics/s1_check/metrics \\
        --tuning-dir experiments/diagnostics/s1_check/tuning --predictions-dir none

Summarize:  python scripts/s1_flexibility_check.py [metrics_dir]
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

METRICS = ["hre", "hre_trunc", "hre_cum", "ks", "w1", "ibs", "ici", "dcal_pass", "unos_c"]


def main(metrics_dir: str = "experiments/diagnostics/s1_check/metrics") -> None:
    files = sorted(Path(metrics_dir).glob("*.parquet"))
    if not files:
        raise SystemExit(f"no parquet files in {metrics_dir}")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df = df[df["converged"]]
    pd.set_option("display.width", 200)

    for cell_id, g in df.groupby("cell_id"):
        print(f"\n=== {cell_id} (reps {int(g['rep'].min())}-{int(g['rep'].max())}, tuned={bool(g['tuned'].any())}) ===")
        print(g.groupby("method")[METRICS].median().round(4).to_string())
        for metric in ("hre", "hre_trunc", "ks"):
            piv = g.pivot(index="rep", columns="method", values=metric)
            for ref in ("weibull_aft", "flowsurv_gumbel", "dsm"):
                if ref in piv:
                    ratio = piv.drop(columns=ref).div(piv[ref], axis=0).median().round(2).to_dict()
                    print(f"  {metric} median ratio to {ref}: {ratio}")


if __name__ == "__main__":
    main(*sys.argv[1:2])
