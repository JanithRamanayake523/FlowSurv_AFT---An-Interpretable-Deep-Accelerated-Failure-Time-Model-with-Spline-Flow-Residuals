"""Real-data loaders (Implementation Plan Phase 7, task 1).

Pre-processes the five public clinical benchmarks from the methodology:
SUPPORT, METABRIC, GBSG, FLCHAIN, and WHAS. Each loader returns a dict with
keys ``t`` (observed times), ``d`` (event indicators), ``x`` (standardized
covariates) as float32 CPU tensors, and ``name``.

- SUPPORT/METABRIC/GBSG use ``pycox``'s standard pre-processed DataFrames.
- FLCHAIN is loaded directly from the Rdatasets CSV (pycox's processed loader
  is fragile against upstream format changes) and processed with one-hot
  encoding of the categorical columns.
- WHAS is loaded from the DeepSurvK HDF5 release (Katzman et al. 2018).
"""

from __future__ import annotations

import io
import warnings
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import requests
import torch
from torch import Tensor

CACHE_DIR = Path(__file__).resolve().parents[3] / "data" / "external"

#: WHAS HDF5 from the DeepSurvK repository (DeepSurv paper, n=1638).
WHAS_URL = (
    "https://raw.githubusercontent.com/arturomoncadatorres/deepsurvk/"
    "master/deepsurvk/datasets/data/whas.h5"
)


def _ensure_cache_dir() -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR


def _download_whas(path: Path) -> None:
    """Download the WHAS HDF5 release to ``path``."""
    response = requests.get(WHAS_URL, timeout=60)
    response.raise_for_status()
    path.write_bytes(response.content)


def _standardize(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    """Z-standardize numeric columns in place (zero mean, unit variance)."""
    df = df.copy()
    for col in numeric_cols:
        if df[col].std() > 0:
            df[col] = (df[col] - df[col].mean()) / df[col].std()
        else:
            df[col] = df[col] - df[col].mean()
    return df


def _df_to_tensors(df: pd.DataFrame) -> dict:
    """Build the canonical dict from a processed DataFrame."""
    x_cols = [c for c in df.columns if c not in ("duration", "event")]
    x = torch.tensor(df[x_cols].to_numpy(dtype=np.float32), dtype=torch.float32)
    t = torch.tensor(df["duration"].to_numpy(dtype=np.float32), dtype=torch.float32)
    d = torch.tensor(df["event"].to_numpy(dtype=np.float32), dtype=torch.float32)
    return {"t": t, "d": d, "x": x, "name": "unknown"}


# ---------------------------------------------------------------------------
# pycox datasets
# ---------------------------------------------------------------------------


def _load_pycox(name: str) -> dict:
    """Generic loader for pycox datasets with x/duration/event layout."""
    try:
        from pycox import datasets as pycox_datasets
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            f"pycox is required for the {name!r} real dataset. "
            "Install the optional Phase-3 dependencies from environment.yml."
        ) from e

    df = getattr(pycox_datasets, name).read_df()
    df = df.rename(columns={"duration": "duration", "event": "event"})
    # pycox already encodes categoricals numerically; standardize all features
    feature_cols = [c for c in df.columns if c not in ("duration", "event")]
    df = _standardize(df, feature_cols)
    out = _df_to_tensors(df)
    out["name"] = name
    return out


def load_support() -> dict:
    """SUPPORT (n ≈ 8,873, censoring ≈ 32%)."""
    return _load_pycox("support")


def load_metabric() -> dict:
    """METABRIC (n ≈ 1,904, censoring ≈ 42%)."""
    return _load_pycox("metabric")


def load_gbsg() -> dict:
    """GBSG / Rotterdam-GBSG (n ≈ 2,232, censoring ≈ 43%)."""
    return _load_pycox("gbsg")


# ---------------------------------------------------------------------------
# FLCHAIN (custom loader)
# ---------------------------------------------------------------------------


def _load_flchain_raw() -> pd.DataFrame:
    """Load the raw FLCHAIN CSV from Rdatasets (bypass pycox's fragile loader)."""
    try:
        from pycox.datasets.from_rdatasets import download_from_rdatasets

        raw, _ = download_from_rdatasets("survival", "flchain")
        return raw
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Could not download FLCHAIN from Rdatasets. "
            "Ensure internet connectivity or install pycox."
        ) from e


def load_flchain() -> dict:
    """FLCHAIN serum free-light-chain assay (n ≈ 7,874, censoring ≈ 72%)."""
    raw = _load_flchain_raw()
    df = (
        raw.drop(columns=["rownames", "chapter"], errors="ignore")
        .dropna(subset=["creatinine"])
        .reset_index(drop=True)
    )
    # sex: True for male (M)
    df["sex"] = (df["sex"] == "M").astype(np.float32)
    # categorical columns
    cat_cols = ["sample.yr", "flc.grp"]
    for col in cat_cols:
        df[col] = df[col].astype("category")

    numeric_cols = [c for c in df.columns if c not in cat_cols + ["futime", "death"]]
    df[numeric_cols] = df[numeric_cols].astype(np.float32)
    df = _standardize(df, numeric_cols)
    df = pd.get_dummies(df, columns=cat_cols, drop_first=False, dtype=np.float32)
    df = df.rename(columns={"futime": "duration", "death": "event"})
    out = _df_to_tensors(df)
    out["name"] = "flchain"
    return out


# ---------------------------------------------------------------------------
# WHAS (DeepSurvK HDF5)
# ---------------------------------------------------------------------------


def load_whas() -> dict:
    """Worcester Heart Attack Study (WHAS, n = 1,638, censoring ≈ 57%).

    Loads the train/test partition from the DeepSurvK HDF5 release and
    returns the combined, standardized dataset.
    """
    try:
        import h5py
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "h5py is required for the WHAS dataset. Install the optional "
            "Phase-3 dependencies from environment.yml."
        ) from e

    path = _ensure_cache_dir() / "whas.h5"
    if not path.exists():
        _download_whas(path)

    with h5py.File(path, "r") as f:
        x = np.concatenate([f["train/x"][()], f["test/x"][()]], axis=0).astype(np.float32)
        t = np.concatenate([f["train/t"][()], f["test/t"][()]], axis=0).astype(np.float32)
        d = np.concatenate([f["train/e"][()], f["test/e"][()]], axis=0).astype(np.float32)

    # standardize covariates
    x_mean = x.mean(axis=0)
    x_std = x.std(axis=0)
    x_std[x_std == 0] = 1.0
    x = (x - x_mean) / x_std

    return {
        "t": torch.tensor(t, dtype=torch.float32),
        "d": torch.tensor(d, dtype=torch.float32),
        "x": torch.tensor(x, dtype=torch.float32),
        "name": "whas",
    }


# ---------------------------------------------------------------------------
# public registry and split helper
# ---------------------------------------------------------------------------

REAL_DATASETS: dict[str, Callable[[], dict]] = {
    "support": load_support,
    "metabric": load_metabric,
    "gbsg": load_gbsg,
    "flchain": load_flchain,
    "whas": load_whas,
}


def real_split(
    t: Tensor,
    d: Tensor,
    x: Tensor,
    seed: int,
    test_frac: float = 0.20,
) -> dict:
    """Stratified 80/20 train/test split, deterministic by ``seed``.

    Splits event and censored subjects separately so the test fold preserves
    the observed event proportion (Methodology Sec. 4.2).
    """
    n = int(t.shape[0])
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=gen)
    event_idx = perm[(d[perm] > 0).nonzero(as_tuple=True)[0]]
    cens_idx = perm[(d[perm] == 0).nonzero(as_tuple=True)[0]]

    n_test_event = max(1, int(round(test_frac * event_idx.numel())))
    n_test_cens = max(1, int(round(test_frac * cens_idx.numel())))

    test_idx = torch.cat([event_idx[:n_test_event], cens_idx[:n_test_cens]])
    train_idx = torch.cat([event_idx[n_test_event:], cens_idx[n_test_cens:]])

    # shuffle within each fold so rows are not ordered by event status
    train_idx = train_idx[torch.randperm(train_idx.numel(), generator=torch.Generator().manual_seed(seed + 1))]
    test_idx = test_idx[torch.randperm(test_idx.numel(), generator=torch.Generator().manual_seed(seed + 2))]

    return {
        "train": {"t": t[train_idx], "d": d[train_idx], "x": x[train_idx]},
        "test": {"t": t[test_idx], "d": d[test_idx], "x": x[test_idx]},
    }
