"""SurvivalMethod wrappers for FlowSurv-AFT and the FlowSurv-Gauss ablation.

These expose the core models through the common :class:`~flowsurv.baselines.common.SurvivalMethod`
interface so that ``run_cell.py``, ``tune.py`` and ``cost.py`` can treat them like
any other baseline (Implementation Plan Phase 3, task 2).
"""

from __future__ import annotations

import time

import torch
from torch import Tensor

from ..models import FlowSurvAFT, FlowSurvGauss, TrainConfig, fit, right_censored_nll
from .common import FitResult, SurvivalMethod, as_output, to_numpy


class _FlowSurvWrapper(SurvivalMethod):
    """Base wrapper with shared fit/predict logic."""

    is_deep: bool = True
    supports_density: bool = True
    supports_hazard: bool = True
    _cls: type = FlowSurvAFT

    def __init__(self) -> None:
        self.model: FlowSurvAFT | FlowSurvGauss | None = None
        self.name: str = "flowsurv_aft"
        self.tuning_space: dict[str, list] = {
            "hidden": [64, 128],
            "n_blocks": [1, 2, 3],
            "dropout": [0.1, 0.2],
            "bins": [8, 16],
            "n_spline_blocks": [1, 2, 3],
        }

    def _make_model(self, n_features: int, **hyper) -> FlowSurvAFT | FlowSurvGauss:
        raise NotImplementedError

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
        t, d, x = (torch.as_tensor(v, dtype=torch.float32) for v in (t, d, x))
        device = hyper.pop("device", "cpu")
        config = TrainConfig(
            lr=hyper.pop("lr", 1e-3),
            final_lr=hyper.pop("final_lr", 1e-5),
            weight_decay=hyper.pop("weight_decay", 1e-4),
            batch_size=hyper.pop("batch_size", 256),
            max_epochs=hyper.pop("max_epochs", 500),
            patience=hyper.pop("patience", 30),
            grad_clip=hyper.pop("grad_clip", 1.0),
            seed=seed,
            device=device,
        )
        self.model = self._make_model(int(x.shape[1]), **hyper)
        if val is not None:
            # early stopping already uses a validation split; ignore external val
            pass
        start = time.perf_counter()
        try:
            res = fit(self.model, t, d, x, config)
        except RuntimeError as e:
            # Some CUDA installs are missing the NVRTC JIT component that
            # torch.special.{erfc,erfinv,log_ndtr} compile through (needed by
            # FlowSurv-Gauss; FlowSurv-AFT's Logistic base never hits this
            # path). erf/ndtr have native CUDA kernels and are unaffected.
            # Retry once on CPU rather than failing the whole fit.
            if "nvrtc" in str(e).lower() and config.device != "cpu":
                config = TrainConfig(**{**config.__dict__, "device": "cpu"})
                self.model = self._make_model(int(x.shape[1]), **hyper)
                res = fit(self.model, t, d, x, config)
            else:
                raise
        info = {
            "train_nll": res.train_nll[-1] if res.train_nll else float("nan"),
            "best_val_nll": res.best_val_nll,
            "best_epoch": res.best_epoch,
        }
        return FitResult(
            wall_time_s=res.wall_time_s,
            converged=True,
            info=info,
        )

    def _device(self) -> torch.device:
        return next(self.model.parameters()).device

    def predict_surv(self, t: Tensor, x: Tensor) -> Tensor:
        if self.model is None:
            raise RuntimeError("fit() must be called before predict_surv()")
        t, x = (torch.as_tensor(v, dtype=torch.float32) for v in (t, x))
        dev = self._device()
        t, x = t.to(dev), x.to(dev)
        with torch.no_grad():
            return as_output(self.model.survival(t, x).cpu().numpy())

    def predict_density(self, t: Tensor, x: Tensor) -> Tensor:
        if self.model is None:
            raise RuntimeError("fit() must be called before predict_density()")
        t, x = (torch.as_tensor(v, dtype=torch.float32) for v in (t, x))
        dev = self._device()
        t, x = t.to(dev), x.to(dev)
        with torch.no_grad():
            return as_output(self.model.density(t, x).cpu().numpy())

    def predict_hazard(self, t: Tensor, x: Tensor) -> Tensor:
        if self.model is None:
            raise RuntimeError("fit() must be called before predict_hazard()")
        t, x = (torch.as_tensor(v, dtype=torch.float32) for v in (t, x))
        dev = self._device()
        t, x = t.to(dev), x.to(dev)
        with torch.no_grad():
            return as_output(self.model.hazard(t, x).cpu().numpy())

    def predict_risk(self, x: Tensor) -> Tensor:
        if self.model is None:
            raise RuntimeError("fit() must be called before predict_risk()")
        x = torch.as_tensor(x, dtype=torch.float32).to(self._device())
        with torch.no_grad():
            return as_output(-self.model.quantile(0.5, x).cpu().numpy())

    def sample(self, x: Tensor, n: int = 1000) -> Tensor:
        if self.model is None:
            raise RuntimeError("fit() must be called before sample()")
        x = torch.as_tensor(x, dtype=torch.float32).to(self._device())
        with torch.no_grad():
            return as_output(self.model.sample(x, n=n).cpu().numpy())


class FlowSurvAFTMethod(_FlowSurvWrapper):
    """FlowSurv-AFT with the full conditional RQS residual flow."""

    name: str = "flowsurv_aft"
    _cls = FlowSurvAFT

    def _make_model(self, n_features: int, **hyper) -> FlowSurvAFT:
        return FlowSurvAFT(
            n_features,
            hidden=hyper.get("hidden", 128),
            n_blocks=hyper.get("n_blocks", 2),
            dropout=hyper.get("dropout", 0.1),
            bins=hyper.get("bins", 8),
            bound=hyper.get("bound", 6.0),
            n_spline_blocks=hyper.get("n_spline_blocks", 1),
        )


class FlowSurvGaussMethod(_FlowSurvWrapper):
    """FlowSurv-Gauss ablation: identity flow with a Gaussian residual law."""

    name: str = "flowsurv_gauss"
    _cls = FlowSurvGauss

    def __init__(self) -> None:
        super().__init__()
        self.tuning_space = {
            "hidden": [64, 128],
            "n_blocks": [1, 2, 3],
            "dropout": [0.1, 0.2],
        }

    def _make_model(self, n_features: int, **hyper) -> FlowSurvGauss:
        return FlowSurvGauss(
            n_features,
            hidden=hyper.get("hidden", 128),
            n_blocks=hyper.get("n_blocks", 2),
            dropout=hyper.get("dropout", 0.1),
        )
