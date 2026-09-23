"""Shardable, checkpointing grid driver (Implementation Plan Phase 6).

Iterates cells x reps x methods, writing one part-file per (cell, rep):
``experiments/metrics/{cell_id}_part{rep}.parquet``. Part-files make runs
crash-safe and reruns idempotent: a (method, rep) row already present is
skipped. Sharding is by cell index (``idx % K == k``) so all reps of a cell
stay in the same shard.

A single :class:`~flowsurv.eval.tune.FrozenTuner` is shared across the run;
frozen configs persist under ``experiments/tuning/`` so interrupted shards
resume without re-tuning. Rows with ``converged=False`` are additionally
collected into ``experiments/failures.parquet`` (failure rates are a
pre-registered finding, prereg Sec. 5).

CLI:
    python -m flowsurv.eval.grid --scenario S2 --shard 0 4 \
        --methods flowsurv_aft dsm --reps 1-100 --device cuda
    python -m flowsurv.eval.grid --real --reps 1-10
    python -m flowsurv.eval.grid --status
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from ..baselines import METHODS
from ..data import REAL_DATASETS, default_grid
from .run_cell import run_real_rep, run_sim_rep
from .tune import FrozenTuner

METRICS_DIR = Path("experiments/metrics")
FAILURES_PATH = Path("experiments/failures.parquet")
REAL_REPS = range(1, 11)  # prereg Sec. 4: 10 repeated 80/20 splits for real data


def shard_items(items, k: int, K: int) -> list:
    """Deterministic strided sharding: shard ``k`` of ``K`` takes idx % K == k."""
    if not 0 <= k < K:
        raise ValueError(f"invalid shard ({k}, {K})")
    return [item for i, item in enumerate(items) if i % K == k]


def _part_path(out: Path, cell_id: str, rep: int) -> Path:
    return out / f"{cell_id}_part{rep}.parquet"


def _flush_failures(failures: list[dict], path: Path = FAILURES_PATH) -> None:
    """Read-concat-write failure rows, deduplicated on (cell_id, rep, method)."""
    if not failures:
        return
    new = pd.DataFrame(failures)
    if path.exists():
        new = pd.concat([pd.read_parquet(path), new], ignore_index=True)
    new = new.drop_duplicates(subset=["cell_id", "rep", "method"], keep="last")
    path.parent.mkdir(parents=True, exist_ok=True)
    new.to_parquet(path, index=False)
    failures.clear()


class _Progress:
    """Running [done/total] counter with elapsed time, printed per fit."""

    def __init__(self, total: int, verbose: bool) -> None:
        self.total = total
        self.verbose = verbose
        self.done = 0
        self.start = time.perf_counter()

    def tick(self, cell_id: str, rep: int, method_name: str, row: dict) -> None:
        self.done += 1
        if not self.verbose:
            return
        elapsed = time.perf_counter() - self.start
        print(
            f"[{self.done}/{self.total}] {cell_id} rep={rep} {method_name:14s} "
            f"conv={row['converged']!s:5s} time={row['time_fit_s']:6.1f}s "
            f"(elapsed {elapsed/60:.1f} min)"
            + (f"  ERR={row['error'][:100]}" if row["error"] else ""),
            flush=True,
        )


def _run_cell_reps(run_fn, cell, cell_id: str, reps, methods, tuner, device, out, failures, progress: "_Progress | None" = None) -> None:
    for rep in reps:
        part = _part_path(out, cell_id, rep)
        done: set[str] = set()
        if part.exists():
            done = set(pd.read_parquet(part)["method"])
        rows = []
        for method_name in methods:
            if method_name in done:
                if progress is not None:
                    progress.done += 1  # already-done rows still count toward total
                continue  # idempotent rerun
            row = run_fn(cell, rep, method_name, tuner=tuner, device=device)
            rows.append(row)
            if progress is not None:
                progress.tick(cell_id, rep, method_name, row)
        if rows:
            df = pd.DataFrame(rows)
            if part.exists():
                df = pd.concat([pd.read_parquet(part), df], ignore_index=True)
            df.to_parquet(part, index=False)
            failures.extend(r for r in rows if not r["converged"])


def run_grid(
    cells=None,
    methods=None,
    reps=range(1, 101),
    out: str | Path = METRICS_DIR,
    shard: tuple[int, int] = (0, 1),
    device: str = "cpu",
    real_only: bool = False,
    sim_only: bool = False,
    audit: bool = False,
    real_reps=REAL_REPS,
    tuning_dir: str | Path = "experiments/tuning",
    verbose: bool = True,
    no_tune: bool = False,
) -> None:
    """Run the evaluation grid with checkpointing and idempotent reruns.

    - ``cells``: ``CellConfig`` list; default :func:`default_grid`. Sharded by
      cell index so a cell's reps stay together.
    - ``methods``: method names; default all of ``METHODS``.
    - ``audit``: pass ``audit=True`` to the FrozenTuner (per-rep nested tuning;
      use only on the audit subset from ``audit_cells``).
    - Real datasets (reps 1-10, prereg Sec. 4.2) are included unless
      ``sim_only``; ``real_only`` skips the simulation grid.
    - ``verbose``: print a ``[done/total]`` line per fit to stdout (default
      on -- this call runs unattended for hours/days, so a silent terminal
      looks hung even when it isn't).
    - ``no_tune``: skip the pre-registered 30-config search entirely and fit
      every DL method at its class defaults (``tuner=None``, same as the
      Phase 5 pilot). Faster, but NOT the pre-registered protocol (prereg
      Sec. 4) -- results from a ``no_tune`` run cannot be used for the
      confirmatory H1-H3 contrasts, only for a quick sanity/direction check.
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    methods = list(methods) if methods else list(METHODS)
    unknown = [m for m in methods if m not in METHODS]
    if unknown:
        raise ValueError(f"unknown methods {unknown}; known: {list(METHODS)}")

    tuner = None if no_tune else FrozenTuner(tuning_dir, audit=audit, verbose=verbose)
    failures: list[dict] = []

    sim_cells = []
    if not real_only:
        sim_cells = list(cells) if cells is not None else list(default_grid())
        sim_cells = shard_items(sim_cells, shard[0], shard[1])
    real_names = list(REAL_DATASETS) if not sim_only else []

    total = len(methods) * (len(sim_cells) * len(list(reps)) + len(real_names) * len(list(real_reps)))
    progress = _Progress(total, verbose) if total else None
    if verbose and total:
        print(f"[grid] {len(sim_cells)} sim cells + {len(real_names)} real datasets x {len(methods)} methods = {total} fits", flush=True)

    for cell in sim_cells:
        _run_cell_reps(
            run_sim_rep, cell, cell.cell_id, reps, methods, tuner, device, out, failures, progress
        )
        _flush_failures(failures)

    for name in real_names:
        cell_id = f"{name}_real"
        _run_cell_reps(
            lambda _c, rep, m, **kw: run_real_rep(name, rep, m, **kw),
            None,
            cell_id,
            real_reps,
            methods,
            tuner,
            device,
            out,
            failures,
            progress,
        )
        _flush_failures(failures)

    _flush_failures(failures)
    if verbose and total:
        elapsed = time.perf_counter() - progress.start
        print(f"[grid] DONE in {elapsed/60:.1f} min", flush=True)


def grid_status(out: str | Path = METRICS_DIR) -> pd.DataFrame:
    """Completion counts and failure rates per (method, scenario) — nightly check."""
    files = sorted(Path(out).glob("*.parquet"))
    columns = ["method", "scenario", "rows", "reps", "cells", "failures", "failure_rate"]
    if not files:
        return pd.DataFrame(columns=columns)
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    grouped = (
        df.groupby(["method", "scenario"])
        .agg(
            rows=("rep", "size"),
            reps=("rep", "nunique"),
            cells=("cell_id", "nunique"),
            failures=("converged", lambda s: int((~s.astype(bool)).sum())),
        )
        .reset_index()
    )
    grouped["failure_rate"] = grouped["failures"] / grouped["rows"].clip(lower=1)
    return grouped.sort_values(["method", "scenario"]).reset_index(drop=True)


def _parse_reps(spec: str) -> list[int]:
    """'1-100' or '1,2,5' -> list of ints."""
    spec = spec.strip()
    if "-" in spec:
        lo, hi = spec.split("-", 1)
        return list(range(int(lo), int(hi) + 1))
    return [int(p) for p in spec.split(",") if p]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scenario", nargs="*", default=None, help="scenario filter, e.g. S2 S3")
    parser.add_argument("--n", nargs="*", type=int, default=None, help="sample-size filter, e.g. 200 1000")
    parser.add_argument("--censoring", nargs="*", type=float, default=None, help="censoring %% filter (0-100), e.g. 0 20")
    parser.add_argument("--ctype", nargs="*", default=None, choices=["I", "III"], help="censoring type filter, e.g. I")
    parser.add_argument("--shard", type=int, nargs=2, metavar=("K", "N"), default=(0, 1))
    parser.add_argument("--methods", nargs="*", default=None)
    parser.add_argument("--reps", type=_parse_reps, default=range(1, 101), help="e.g. 1-100")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default=str(METRICS_DIR))
    parser.add_argument("--tuning-dir", default="experiments/tuning")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--real", action="store_true", help="real datasets only")
    mode.add_argument("--sim", action="store_true", help="simulation grid only")
    parser.add_argument("--audit", action="store_true", help="per-rep nested tuning (audit cells)")
    parser.add_argument(
        "--no-tune", action="store_true",
        help="skip the 30-config search, fit DL methods at class defaults (NOT the pre-registered protocol -- sanity-check runs only)",
    )
    parser.add_argument("--status", action="store_true", help="print grid status and exit")
    parser.add_argument("--quiet", action="store_true", help="suppress the per-fit progress line")
    args = parser.parse_args(argv)

    if args.status:
        print(grid_status(args.out).to_string(index=False))
        return

    cells = None
    if args.scenario or args.n or args.censoring is not None or args.ctype:
        cells = list(default_grid())
        if args.scenario:
            cells = [c for c in cells if c.scenario in set(args.scenario)]
        if args.n:
            cells = [c for c in cells if c.n in set(args.n)]
        if args.censoring is not None:
            # cell.censoring is a fraction (0.0-1.0); CLI takes percent (0-100)
            wanted = {round(c / 100.0, 4) for c in args.censoring}
            cells = [c for c in cells if round(c.censoring, 4) in wanted]
        if args.ctype:
            cells = [c for c in cells if c.censoring_type in set(args.ctype)]
        if not cells:
            raise SystemExit(
                f"no cells match filters: scenario={args.scenario} n={args.n} "
                f"censoring={args.censoring} ctype={args.ctype}"
            )

    run_grid(
        cells=cells,
        methods=args.methods,
        reps=args.reps,
        out=args.out,
        shard=tuple(args.shard),
        device=args.device,
        real_only=args.real,
        sim_only=args.sim,
        audit=args.audit,
        tuning_dir=args.tuning_dir,
        verbose=not args.quiet,
        no_tune=args.no_tune,
    )


if __name__ == "__main__":
    main()
