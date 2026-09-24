"""Per-fit prediction store, so a metric fix is a recompute and not a refit.

Every metric in ``run_cell._evaluate`` is a function of a handful of arrays on
the test fold. Persisting them next to the metrics lets a future estimator fix
(as in prereg Deviation 4) be re-evaluated offline instead of refitting every
model. One compressed ``.npz`` per (cell, replication, method):

    <root>/<cell_id>/rep<RRR>__<method>.npz

Stored (float32 unless noted): ``grid`` (m,), ``surv`` (m, n) survival on the
grid, ``s_at_obs`` (n,) survival at each subject's own observed time,
``s_at_tau`` (n,), ``hazard`` (m, n) (absent when the method has no hazard),
``risk`` (n,), ``t`` (n,), ``d`` (n,), ``tau`` (scalar). Ground-truth hazards
and CDFs and the test covariates are NOT stored: they are regenerated exactly
from the cell definition and the deterministic per-(cell, rep) seed.

Disk: about 0.5 MB per simulation fit (0.3 x 200 x 2 arrays), so the grid
driver saves only replications <= ``prediction_reps`` (default 5) plus every
real-data split; see ``grid.run_grid``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

PREDICTIONS_DIR = Path("experiments/predictions")


def prediction_path(root: str | Path, cell_id: str, rep: int, method: str) -> Path:
    return Path(root) / cell_id / f"rep{int(rep):03d}__{method}.npz"


def save_predictions(root: str | Path, cell_id: str, rep: int, method: str, arrays: dict) -> Path:
    """Write the captured test-fold arrays for one fit (atomic replace)."""
    path = prediction_path(root, cell_id, rep, method)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {k: np.asarray(v, dtype=np.float32) for k, v in arrays.items() if v is not None}
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **payload)
    tmp.replace(path)
    return path


def load_predictions(root: str | Path, cell_id: str, rep: int, method: str) -> dict[str, np.ndarray]:
    """Load the arrays saved for one fit."""
    with np.load(prediction_path(root, cell_id, rep, method)) as f:
        return {k: f[k] for k in f.files}
