"""Live-refreshing progress bar for the Phase 6 grid run.

Run in a separate terminal from the fit processes; polls experiments/metrics/
every 20s via flowsurv.eval.grid.grid_status() and renders a text progress bar.
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, "src")

from flowsurv.eval.grid import grid_status  # noqa: E402

SIM_SCENARIOS = ["S1", "S2", "S3", "S4", "S5"]
REAL_DATASETS = ["flchain", "gbsg", "metabric", "support", "whas"]
R100_METHODS = [
    "flowsurv_aft", "flowsurv_gauss", "cox_ph", "weibull_aft", "log_normal_aft",
    "rsf", "deepsurv", "deephit", "dsm", "royston_parmar",
]
R20_METHODS = ["ausset_cnf"]

CELLS_PER_SCENARIO = 24  # 3 n x 4 censoring x 2 type
REAL_REPS = 10


def targets() -> dict[tuple[str, str], int]:
    t: dict[tuple[str, str], int] = {}
    for m in R100_METHODS:
        for s in SIM_SCENARIOS:
            t[(m, s)] = CELLS_PER_SCENARIO * 100
        for d in REAL_DATASETS:
            t[(m, d)] = REAL_REPS
    for m in R20_METHODS:
        for s in SIM_SCENARIOS:
            t[(m, s)] = CELLS_PER_SCENARIO * 20
        for d in REAL_DATASETS:
            t[(m, d)] = REAL_REPS
    return t


def bar(frac: float, width: int = 30) -> str:
    n = int(round(frac * width))
    return "[" + "#" * n + "-" * (width - n) + f"] {frac*100:5.1f}%"


def render() -> None:
    df = grid_status()
    tgt = targets()
    done = {(r.method, r.scenario): int(r.rows) for r in df.itertuples()}
    fails = {(r.method, r.scenario): int(r.failures) for r in df.itertuples()}

    total_done = sum(min(done.get(k, 0), v) for k, v in tgt.items())
    total_target = sum(tgt.values())
    total_fail = sum(fails.values())

    if sys.stdout.isatty():
        os.system("cls" if os.name == "nt" else "clear")
    print(f"Phase 6 progress -- {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(bar(total_done / total_target if total_target else 0))
    print(f"{total_done}/{total_target} rows  |  {total_fail} failures\n")

    for m in R100_METHODS + R20_METHODS:
        sim_done = sum(min(done.get((m, s), 0), tgt[(m, s)]) for s in SIM_SCENARIOS)
        sim_target = sum(tgt[(m, s)] for s in SIM_SCENARIOS)
        real_done = sum(min(done.get((m, d), 0), tgt[(m, d)]) for d in REAL_DATASETS)
        real_target = sum(tgt[(m, d)] for d in REAL_DATASETS)
        frac = (sim_done + real_done) / (sim_target + real_target)
        print(f"{m:16s} {bar(frac, 20)}  (sim {sim_done}/{sim_target}, real {real_done}/{real_target})")
    sys.stdout.flush()


def main() -> None:
    while True:
        try:
            render()
        except Exception as e:  # noqa: BLE001
            print(f"[watch_phase6] error: {e!r}")
        time.sleep(20)


if __name__ == "__main__":
    main()
