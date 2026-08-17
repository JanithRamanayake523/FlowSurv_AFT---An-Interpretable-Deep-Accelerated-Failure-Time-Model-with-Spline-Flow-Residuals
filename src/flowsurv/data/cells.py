"""Simulation design grid: cells, deterministic seeds, dataset generation and
parquet caching (Methodology Sec. 3.1; Implementation Plan Phase 2).

Seed protocol (pre-registration Sec. 4): a master seed plus deterministic
per-(cell, replication) seeds via SHA-256. Within a replication every method
sees byte-identical data, because methods receive the dataset generated from
``cell_seed(cell_id, rep)`` (event times) and ``cell_seed(cell_id, rep) + 1``
(censoring uniforms) -- the common-random-seeds requirement.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import torch
import yaml
from torch import Tensor

from .censoring import apply_censoring
from .dgp import SCENARIOS

#: master seed (pre-registration Sec. 4)
MASTER_SEED = 20260804


@dataclass(frozen=True)
class CellConfig:
    """One cell of the 120-cell simulation design grid (Methodology Sec. 3.1).

    ``cell_id`` format: ``"S2_n1000_c50_typeIII"``.
    """

    cell_id: str
    scenario: str  # "S1".."S5" ("S6" is semi-synthetic, not on the grid)
    n: int
    censoring: float  # one of 0.0, 0.2, 0.5, 0.8
    censoring_type: str  # "I" or "III"
    n_reps: int = 100


def cell_seed(cell_id: str, rep: int, master_seed: int = MASTER_SEED) -> int:
    """Deterministic per-(cell, replication) seed.

    SHA-256 of ``f"{master_seed}:{cell_id}:{rep}"`` folded to a non-negative
    int32. Same inputs always give the same seed, independent of process or
    platform.
    """
    digest = hashlib.sha256(f"{master_seed}:{cell_id}:{rep}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def default_grid() -> list[CellConfig]:
    """The full 120-cell grid: S1-S5 x n in {200,1000,5000} x censoring in
    {0.0,0.2,0.5,0.8} x type in {I,III} (Methodology Sec. 3.1)."""
    cells: list[CellConfig] = []
    for scenario in ("S1", "S2", "S3", "S4", "S5"):
        for n in (200, 1000, 5000):
            for censoring in (0.0, 0.2, 0.5, 0.8):
                for ctype in ("I", "III"):
                    cell_id = f"{scenario}_n{n}_c{int(round(censoring * 100))}_type{ctype}"
                    cells.append(
                        CellConfig(
                            cell_id=cell_id,
                            scenario=scenario,
                            n=n,
                            censoring=censoring,
                            censoring_type=ctype,
                        )
                    )
    return cells


def _truth_callables(scenario: str) -> dict:
    """Ground-truth conditional quantities for ``scenario`` (from SCENARIOS).

    All returned functions accept a 1-D evaluation grid ``t_grid (m,)`` and
    covariates ``x (n, p)`` and return a (m, n) tensor. ``dgp.py``'s true
    functions are written for ``t`` of shape ``(m, 1)`` (grid) or ``(n,)``
    (per-subject), so the wrappers below lift a 1-D grid to ``(m, 1)``.
    """
    spec = SCENARIOS[scenario]

    def _gridify(t_grid):
        tg = torch.as_tensor(t_grid, dtype=torch.float32)
        if tg.ndim == 1:
            tg = tg.unsqueeze(1)
        return tg

    def hazard(t_grid, x):
        return spec["hazard"](_gridify(t_grid), x)

    def cdf(t_grid, x):
        return spec["cdf"](_gridify(t_grid), x)

    def density(t_grid, x):
        return spec["density"](_gridify(t_grid), x)

    return {"hazard": hazard, "cdf": cdf, "density": density}


def generate_dataset(cell: CellConfig, rep: int) -> dict:
    """Generate one (cell, replication) dataset.

    Event times use seed ``cell_seed(cell.cell_id, rep)``; censoring uniforms
    use the separate stream ``cell_seed(cell.cell_id, rep) + 1``. Because both
    derive only from the cell and replication, every method sees identical
    data within a replication (common random seeds, pre-registration Sec. 4).

    Returns a dict with keys ``"x"`` (n, p) float32, ``"t"`` observed times,
    ``"d"`` event indicators, ``"scenario"``, ``"truth"`` (callables, see
    ``_truth_callables``), plus metadata ``"cell_id"`` and ``"rep"``.
    The 70/15/15 train/val/test split is the caller's job (eval/run_cell.py)
    via ``split_train_val_test``.
    """
    seed = cell_seed(cell.cell_id, rep)
    t, x = SCENARIOS[cell.scenario]["simulate"](cell.n, seed=seed)
    t_obs, d = apply_censoring(t, cell.censoring, cell.censoring_type, seed=seed + 1)
    return {
        "x": x,
        "t": t_obs,
        "d": d,
        "scenario": cell.scenario,
        "truth": _truth_callables(cell.scenario),
        "cell_id": cell.cell_id,
        "rep": rep,
    }


def split_train_val_test(
    t: Tensor,
    d: Tensor,
    x: Tensor,
    seed: int,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
) -> dict:
    """Deterministic 70/15/15 split (Methodology Sec. 2.4).

    Returns ``{"train": {...}, "val": {...}, "test": {...}}`` where each value
    is a dict of ``(t, d, x)``. The permutation depends only on ``seed``.
    """
    n = t.shape[0]
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=gen)
    n_test = int(round(n * test_frac))
    n_val = int(round(n * val_frac))
    idx = {
        "test": perm[:n_test],
        "val": perm[n_test : n_test + n_val],
        "train": perm[n_test + n_val :],
    }
    return {
        name: {"t": t[i], "d": d[i], "x": x[i]} for name, i in idx.items()
    }


# ---------------------------------------------------------------------------
# Dataset caching (parquet; Implementation Plan Phase 2, task 3)
# ---------------------------------------------------------------------------


def cache_path(cell: CellConfig, rep: int, root: str | Path = "experiments/datasets") -> Path:
    """Canonical parquet cache location for one (cell, replication)."""
    return Path(root) / f"{cell.cell_id}_rep{rep:03d}.parquet"


def save_dataset(dataset: dict, path: str | Path) -> Path:
    """Save a ``generate_dataset`` result to parquet.

    Stores x columns ``x0..x{p-1}``, ``t``, ``d`` and a constant ``scenario``
    column. Truth callables are *not* serialized; they are rebuilt from the
    scenario on load.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    x = dataset["x"]
    df = pd.DataFrame(x.numpy(), columns=[f"x{j}" for j in range(x.shape[1])])
    df["t"] = dataset["t"].numpy()
    df["d"] = dataset["d"].numpy()
    df["scenario"] = dataset["scenario"]
    df.to_parquet(path, index=False)
    return path


def load_dataset(path: str | Path) -> dict:
    """Load a cached dataset; truth callables are rebuilt from the scenario."""
    df = pd.read_parquet(path)
    xcols = sorted(
        (c for c in df.columns if c.startswith("x") and c[1:].isdigit()),
        key=lambda c: int(c[1:]),
    )
    x = torch.tensor(df[xcols].to_numpy(), dtype=torch.float32)
    t = torch.tensor(df["t"].to_numpy(), dtype=torch.float32)
    d = torch.tensor(df["d"].to_numpy(), dtype=torch.float32)
    scenario = str(df["scenario"].iloc[0])
    return {"x": x, "t": t, "d": d, "scenario": scenario, "truth": _truth_callables(scenario)}


def write_default_configs(root: str | Path = "configs/sim") -> list[Path]:
    """Write one YAML per cell of ``default_grid()``.

    Not run at import time; invoked by the grid driver once running code is
    allowed (Implementation Plan Phase 2, task 3).
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for cell in default_grid():
        path = root / f"{cell.cell_id}.yaml"
        with open(path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(asdict(cell), fh, sort_keys=False)
        paths.append(path)
    return paths
