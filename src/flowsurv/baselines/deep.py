"""Deep-learning baselines behind the common SurvivalMethod interface.

Wraps pycox's DeepSurv (CoxPH) and DeepHit (DeepHitSingle) with the same
fit/predict contract as FlowSurv-AFT.  pycox expects ``torchtuples`` optimizers
and label transforms; this module handles those details internally so the
runner remains package-agnostic.

All baselines return float32 CPU tensors at the interface boundary.
"""

from __future__ import annotations

import contextlib
import os
import tempfile
import time
import uuid
import warnings
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import Tensor

from .common import FitResult, SurvivalMethod, as_output, density_from_survival, hazard_from_survival, interp_surv, to_numpy


_NO_VAL_EVENTS = "validation fold has no events; early stopping on a partial-likelihood/ranking loss is undefined"


def _val_has_no_events(val) -> bool:
    """True when a supplied validation fold contains no observed event.

    Cox partial likelihood (DeepSurv) is 0/0 = NaN with no events, so early stopping never
    sees an improvement and silently keeps near-untrained weights (Uno's C ~ 0.5, reported as
    converged); DeepHit's discretizer crashes on an empty event set. Either way the fit is
    not meaningful, so it is reported as a failure instead.
    """
    if val is None:
        return False
    d_v = torch.as_tensor(val[1])
    return float(d_v.sum()) == 0.0


class _PycoxMLP(nn.Module):
    """Simple MLP used by DeepSurv and DeepHit."""

    def __init__(
        self,
        in_features: int,
        hidden: int = 128,
        n_blocks: int = 2,
        dropout: float = 0.1,
        out_features: int = 1,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(in_features, hidden), nn.ReLU(), nn.Dropout(dropout)]
        for _ in range(max(0, n_blocks - 1)):
            layers += [nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout)]
        layers.append(nn.Linear(hidden, out_features))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


@contextlib.contextmanager
def _weights_only_torch_load():
    """Load torchtuples' early-stopping checkpoint with ``weights_only=True``.

    torchtuples reloads the best weights with a bare ``torch.load`` (a
    FutureWarning: the default flips to weights_only=True). The file is a plain
    tensor ``state_dict``, so the safe mode loads it fine -- this makes it what
    the warning asks for instead of silencing it.
    """
    original = torch.load

    def patched(*args, **kwargs):
        kwargs.setdefault("weights_only", True)
        return original(*args, **kwargs)

    torch.load = patched
    try:
        yield
    finally:
        torch.load = original


def _pycox_fit(
    model,
    x_train: np.ndarray,
    y_train: tuple[np.ndarray, np.ndarray],
    x_val: np.ndarray | None = None,
    y_val: tuple[np.ndarray, np.ndarray] | None = None,
    batch_size: int = 256,
    epochs: int = 500,
    patience: int = 30,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str = "cpu",
    verbose: bool = False,
) -> tuple[bool, dict, Any]:
    """Fit a pycox model with AdamW + early stopping on validation loss.

    Returns ``(converged, info, model)``.  ``info`` contains ``val_loss``
    (the best validation loss observed) when a validation fold is supplied.
    """
    try:
        import torchtuples as tt
    except ImportError as e:  # pragma: no cover
        return False, {"error": f"torchtuples not installed: {e}"}, model

    opt = tt.optim.AdamW(lr=lr, decoupled_weight_decay=weight_decay, nb_epochs=epochs)
    model.optimizer = opt
    model.set_device(device)

    callbacks = None
    ckpt_path = None
    if x_val is not None and y_val is not None:
        val_data = (x_val, y_val)
        # Checkpoint in the temp dir, not the cwd: an interrupted run otherwise
        # leaves weight_checkpoint_*.pt files wherever it was launched from.
        ckpt_path = os.path.join(tempfile.gettempdir(), f"weight_checkpoint_{uuid.uuid4().hex}.pt")
        early_stop = tt.callbacks.EarlyStopping(patience=patience, file_path=ckpt_path)
        callbacks = [early_stop]
    else:
        val_data = None

    start = time.perf_counter()
    try:
        with _weights_only_torch_load():
            log = model.fit(
                x_train,
                y_train,
                batch_size=batch_size,
                epochs=epochs,
                callbacks=callbacks,
                verbose=verbose,
                val_data=val_data,
            )
        converged = True
        info: dict = {"epochs": len(log.epochs)}
        if val_data is not None:
            # Best validation loss = the loss of the weights EarlyStopping restores.
            info["val_loss"] = float(log.to_pandas()["val_loss"].min())
    except Exception as e:
        converged = False
        info = {"error": repr(e)}
        warnings.warn(f"pycox fit failed: {e!r}")
    finally:
        if ckpt_path is not None and os.path.exists(ckpt_path):
            os.remove(ckpt_path)
    wall_time_s = time.perf_counter() - start
    info["wall_time_s"] = wall_time_s
    return converged, info, model


def _df_to_surv(surv_df: pd.DataFrame, t_query: np.ndarray) -> np.ndarray:
    """Interpolate a pycox ``predict_surv_df`` DataFrame onto ``t_query``."""
    times = surv_df.index.to_numpy()
    surv = surv_df.to_numpy()
    return interp_surv(times, surv, t_query)


def _median_from_curves(times: np.ndarray, surv: np.ndarray) -> np.ndarray:
    """Median survival time per column of a survival-curve matrix ``surv`` (m, n) on ``times`` (m,).

    Linear interpolation of S between the two time points bracketing 0.5. A subject whose
    curve never falls to 0.5 within the model's time range has its median beyond the last
    cut; it is placed at ``t_last * (1 + (S_last - 0.5))``, which keeps such subjects ordered
    by S_last (higher survival = longer median) instead of tying them all.
    """
    reached = surv <= 0.5
    has = reached.any(axis=0)
    k = reached.argmax(axis=0)  # first index with S <= 0.5 (0 if never; masked below)
    cols = np.arange(surv.shape[1])
    k_prev = np.clip(k - 1, 0, None)
    s0, s1 = surv[k_prev, cols], surv[k, cols]
    t0, t1 = times[k_prev], times[k]
    frac = np.where(s0 > s1, (s0 - 0.5) / np.where(s0 > s1, s0 - s1, 1.0), 0.0)
    interp = np.where(k == 0, times[0], t0 + np.clip(frac, 0.0, 1.0) * (t1 - t0))
    beyond = times[-1] * (1.0 + (surv[-1] - 0.5))
    return np.where(has, interp, beyond)


class DeepSurv(SurvivalMethod):
    """DeepSurv (pycox.models.CoxPH) — Cox proportional hazards with an MLP risk function."""

    name: str = "deepsurv"
    is_deep: bool = True
    supports_density: bool = True
    supports_hazard: bool = True

    def __init__(self) -> None:
        self.model: Any | None = None
        self.tuning_space: dict[str, list] = {
            "hidden": [64, 128],
            "n_blocks": [1, 2, 3],
            "dropout": [0.1, 0.2],
            "lr": [1e-3, 5e-4],
            "batch_size": [256],
        }

    def fit(
        self,
        t: Tensor,
        d: Tensor,
        x: Tensor,
        *,
        val: tuple[Tensor, Tensor, Tensor] | None = None,
        seed: int = 0,
        **hyper,
    ) -> FitResult:
        try:
            from pycox.models import CoxPH
        except ImportError as e:  # pragma: no cover
            return FitResult(wall_time_s=0.0, converged=False, info={"error": f"pycox not installed: {e}"})

        if _val_has_no_events(val):
            return FitResult(wall_time_s=0.0, converged=False, info={"error": _NO_VAL_EVENTS})
        t_np, d_np, x_np = to_numpy(t, d, x)
        x_np = x_np.astype(np.float32)
        t_np = t_np.astype(np.float32)
        d_np = d_np.astype(np.float32)
        torch.manual_seed(seed)
        net = _PycoxMLP(
            x_np.shape[1],
            hidden=hyper.get("hidden", 128),
            n_blocks=hyper.get("n_blocks", 2),
            dropout=hyper.get("dropout", 0.1),
            out_features=1,
        )
        self.model = CoxPH(net, optimizer=None, device=hyper.get("device", "cpu"))

        y_train = (t_np, d_np)
        if val is not None:
            t_v, d_v, x_v = (v.detach().cpu().numpy() for v in val)
            x_v = x_v.astype(np.float32)
            t_v = t_v.astype(np.float32)
            d_v = d_v.astype(np.float32)
            y_val = (t_v, d_v)
        else:
            x_v, y_val = None, None

        converged, info, self.model = _pycox_fit(
            self.model,
            x_np,
            y_train,
            x_val=x_v,
            y_val=y_val,
            batch_size=hyper.get("batch_size", 256),
            epochs=hyper.get("max_epochs", 500),
            patience=hyper.get("patience", 30),
            lr=hyper.get("lr", 1e-3),
            weight_decay=hyper.get("weight_decay", 1e-4),
            device=hyper.get("device", "cpu"),
            verbose=hyper.get("verbose", False),
        )
        if converged:
            try:
                self.model.compute_baseline_hazards(x_np, y_train)
            except Exception as e:
                converged = False
                info["error"] = repr(e)
                warnings.warn(f"DeepSurv baseline-hazard computation failed: {e!r}")
        return FitResult(wall_time_s=info.get("wall_time_s", time.perf_counter()), converged=converged, info=info)

    def _surv64(self, t: Tensor, x: Tensor) -> tuple[np.ndarray, np.ndarray]:
        """Survival as float64 (m, n) plus the query times; never cast to float32."""
        if self.model is None:
            raise RuntimeError("DeepSurv has not been fit")
        t_np, _, x_np = to_numpy(t, torch.zeros_like(t), x)
        x_np = x_np.astype(np.float32)
        t_max = float(np.max(t_np))
        surv_df = self.model.predict_surv_df(x_np, max_duration=t_max)
        return _df_to_surv(surv_df, t_np), t_np

    def predict_surv(self, t: Tensor, x: Tensor) -> Tensor:
        return as_output(self._surv64(t, x)[0])

    def predict_density(self, t: Tensor, x: Tensor) -> Tensor:
        surv, t_np = self._surv64(t, x)
        return as_output(density_from_survival(t_np, surv))

    def predict_hazard(self, t: Tensor, x: Tensor) -> Tensor:
        surv, t_np = self._surv64(t, x)
        return as_output(hazard_from_survival(t_np, surv))

    def predict_risk(self, x: Tensor) -> Tensor:
        if self.model is None:
            raise RuntimeError("DeepSurv has not been fit")
        _, _, x_np = to_numpy(torch.zeros(x.shape[0]), torch.zeros(x.shape[0]), x)
        x_np = x_np.astype(np.float32)
        # pycox predict returns risk score (log partial hazard)
        risk = self.model.predict(x_np).ravel()
        return as_output(risk)


class DeepHit(SurvivalMethod):
    """DeepHit single-risk (pycox.models.DeepHitSingle) with discrete-time interpolation."""

    name: str = "deephit"
    is_deep: bool = True
    supports_density: bool = True
    supports_hazard: bool = True

    def __init__(self) -> None:
        self.model: Any | None = None
        self.interpolator: Any | None = None
        self.tuning_space: dict[str, list] = {
            "hidden": [64, 128],
            "n_blocks": [1, 2, 3],
            "dropout": [0.1, 0.2],
            "num_durations": [20, 50],
            "alpha": [0.2, 1.0],
            "sigma": [0.1, 0.5],
            "lr": [1e-3, 5e-4],
            "batch_size": [256],
        }

    def fit(
        self,
        t: Tensor,
        d: Tensor,
        x: Tensor,
        *,
        val: tuple[Tensor, Tensor, Tensor] | None = None,
        seed: int = 0,
        **hyper,
    ) -> FitResult:
        try:
            from pycox.models import DeepHitSingle
            from pycox.preprocessing.label_transforms import LabTransDiscreteTime
        except ImportError as e:  # pragma: no cover
            return FitResult(wall_time_s=0.0, converged=False, info={"error": f"pycox not installed: {e}"})

        if _val_has_no_events(val):
            return FitResult(wall_time_s=0.0, converged=False, info={"error": _NO_VAL_EVENTS})
        t_np, d_np, x_np = to_numpy(t, d, x)
        x_np = x_np.astype(np.float32)
        t_np = t_np.astype(np.float32)
        d_np = d_np.astype(np.float32)
        num_durations = hyper.get("num_durations", 50)
        labtrans = LabTransDiscreteTime(num_durations)
        y_train = labtrans.fit_transform(t_np, d_np)
        # y_train is a tuple (discrete_time, event); durations transformed
        # We need the cuts for the network output dimension and duration_index.
        cuts = labtrans.cuts

        torch.manual_seed(seed)
        net = _PycoxMLP(
            x_np.shape[1],
            hidden=hyper.get("hidden", 128),
            n_blocks=hyper.get("n_blocks", 2),
            dropout=hyper.get("dropout", 0.1),
            out_features=len(cuts),
        )
        self.model = DeepHitSingle(
            net,
            optimizer=None,
            device=hyper.get("device", "cpu"),
            duration_index=cuts,
            alpha=hyper.get("alpha", 0.2),
            sigma=hyper.get("sigma", 0.1),
        )

        if val is not None:
            t_v, d_v, x_v = (v.detach().cpu().numpy() for v in val)
            x_v = x_v.astype(np.float32)
            t_v = t_v.astype(np.float32)
            d_v = d_v.astype(np.float32)
            y_val = labtrans.transform(t_v, d_v)
        else:
            x_v, y_val = None, None

        converged, info, self.model = _pycox_fit(
            self.model,
            x_np,
            y_train,
            x_val=x_v,
            y_val=y_val,
            batch_size=hyper.get("batch_size", 256),
            epochs=hyper.get("max_epochs", 500),
            patience=hyper.get("patience", 30),
            lr=hyper.get("lr", 1e-3),
            weight_decay=hyper.get("weight_decay", 1e-4),
            device=hyper.get("device", "cpu"),
            verbose=hyper.get("verbose", False),
        )
        # Not exposed as "val_loss": DeepHit's loss mixes an NLL and a ranking
        # term weighted by alpha/sigma/num_durations, which are themselves in the
        # tuning space, so it is not comparable across configs (it would simply
        # favor the smallest alpha). The tuner scores DeepHit by the exact
        # censored validation NLL of its interpolated survival curve instead.
        if "val_loss" in info:
            info["package_val_loss"] = info.pop("val_loss")
        if converged:
            self.interpolator = self.model.interpolate(sub=10, scheme="const_pdf")
        return FitResult(wall_time_s=info.get("wall_time_s", time.perf_counter()), converged=converged, info=info)

    def _surv64(self, t: Tensor, x: Tensor) -> tuple[np.ndarray, np.ndarray]:
        """Survival as float64 (m, n) plus the query times; never cast to float32."""
        if self.model is None or self.interpolator is None:
            raise RuntimeError("DeepHit has not been fit")
        t_np, _, x_np = to_numpy(t, torch.zeros_like(t), x)
        x_np = x_np.astype(np.float32)
        surv_df = self.interpolator.predict_surv_df(x_np)
        return _df_to_surv(surv_df, t_np), t_np

    def predict_surv(self, t: Tensor, x: Tensor) -> Tensor:
        return as_output(self._surv64(t, x)[0])

    def predict_density(self, t: Tensor, x: Tensor) -> Tensor:
        surv, t_np = self._surv64(t, x)
        return as_output(density_from_survival(t_np, surv))

    def predict_hazard(self, t: Tensor, x: Tensor) -> Tensor:
        surv, t_np = self._surv64(t, x)
        return as_output(hazard_from_survival(t_np, surv))

    def predict_risk(self, x: Tensor) -> Tensor:
        if self.model is None:
            raise RuntimeError("DeepHit has not been fit")
        # Risk score: negative median survival time read off the model's own
        # time cuts (a fixed 0-10 grid gave every subject the same risk whenever
        # times were in days/months, i.e. Uno's C = 0.5 on real data).
        _, _, x_np = to_numpy(torch.zeros(x.shape[0]), torch.zeros(x.shape[0]), x)
        x_np = x_np.astype(np.float32)
        if self.interpolator is None:
            return as_output(np.zeros(x_np.shape[0]))
        surv_df = self.interpolator.predict_surv_df(x_np)
        return as_output(-_median_from_curves(surv_df.index.to_numpy(dtype=np.float64), surv_df.to_numpy(dtype=np.float64)))
