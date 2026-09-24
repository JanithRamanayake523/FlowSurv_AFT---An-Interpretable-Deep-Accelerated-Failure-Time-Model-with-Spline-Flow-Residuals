"""SurvivalMethod wrappers for FlowSurv-AFT and the FlowSurv-Gauss ablation.

These expose the core models through the common :class:`~flowsurv.baselines.common.SurvivalMethod`
interface so that ``run_cell.py``, ``tune.py`` and ``cost.py`` can treat them like
any other baseline (Implementation Plan Phase 3, task 2).
"""

from __future__ import annotations

import time

import torch
from torch import Tensor

from ..models import FlowSurvAFT, FlowSurvGauss, FlowSurvGumbel, TrainConfig, fit
from .common import FitResult, SurvivalMethod, as_output, to_numpy


def mode_kwargs(method, paired: bool) -> dict:
    """Explicit grid/paired prediction mode for the FlowSurv wrappers.

    A 200-point evaluation grid and a test fold with exactly 200 subjects are
    indistinguishable by shape, so callers state which one they mean. Other
    methods do not take the argument and get no extra kwargs.
    """
    return {"paired": paired} if isinstance(method, _FlowSurvWrapper) else {}


class _FlowSurvWrapper(SurvivalMethod):
    """Base wrapper with shared fit/predict logic."""

    is_deep: bool = True
    supports_density: bool = True
    supports_hazard: bool = True
    _cls: type = FlowSurvAFT
    #: pre-registered L2 penalty on spline derivatives (prereg Sec. 4); 0 for models without a spline
    _spline_l2: float = 1e-5

    def __init__(self) -> None:
        self.model: FlowSurvAFT | FlowSurvGauss | FlowSurvGumbel | None = None
        self.name: str = "flowsurv_aft"
        self.tuning_space: dict[str, list] = {
            "hidden": [64, 128],
            "n_blocks": [1, 2, 3],
            "dropout": [0.1, 0.2],
            "bins": [8, 16],
            "n_spline_blocks": [1, 2, 3],
        }

    def _make_model(self, n_features: int, **hyper) -> FlowSurvAFT | FlowSurvGauss | FlowSurvGumbel:
        raise NotImplementedError

    def _new_model(self, n_features: int, mean_log_t: float, hyper: dict):
        """Build the model and centre its identity warm start on the data log-time scale."""
        model = self._make_model(n_features, **hyper)
        model.encoder.mu_offset.fill_(mean_log_t)
        return model

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
            spline_l2=hyper.pop("spline_l2", self._spline_l2),
            seed=seed,
            device=device,
        )
        # Centre the warm start (mu = 0 means t = 1) on the data log-time scale.
        mean_log_t = float(t.clamp_min(torch.finfo(t.dtype).tiny).log().mean())
        # The caller's validation split is used for early stopping; without one
        # the trainer carves 15% out of the training data.
        val_data = None if val is None else tuple(torch.as_tensor(v, dtype=torch.float32) for v in val)
        self.model = self._new_model(int(x.shape[1]), mean_log_t, hyper)
        start = time.perf_counter()
        try:
            res = fit(self.model, t, d, x, config, val=val_data)
        except RuntimeError as e:
            # Some CUDA installs are missing the NVRTC JIT component that
            # torch.special.{erfc,erfinv,log_ndtr} compile through (needed by
            # FlowSurv-Gauss; FlowSurv-AFT's Logistic base never hits this
            # path). erf/ndtr have native CUDA kernels and are unaffected.
            # Retry once on CPU rather than failing the whole fit.
            if "nvrtc" in str(e).lower() and config.device != "cpu":
                config = TrainConfig(**{**config.__dict__, "device": "cpu"})
                self.model = self._new_model(int(x.shape[1]), mean_log_t, hyper)
                res = fit(self.model, t, d, x, config, val=val_data)
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

    def predict_surv(self, t: Tensor, x: Tensor, paired: bool | None = None) -> Tensor:
        if self.model is None:
            raise RuntimeError("fit() must be called before predict_surv()")
        t, x = (torch.as_tensor(v, dtype=torch.float32) for v in (t, x))
        dev = self._device()
        t, x = t.to(dev), x.to(dev)
        with torch.no_grad():
            return as_output(self.model.survival(t, x, paired).cpu().numpy())

    def predict_density(self, t: Tensor, x: Tensor, paired: bool | None = None) -> Tensor:
        if self.model is None:
            raise RuntimeError("fit() must be called before predict_density()")
        t, x = (torch.as_tensor(v, dtype=torch.float32) for v in (t, x))
        dev = self._device()
        t, x = t.to(dev), x.to(dev)
        with torch.no_grad():
            return as_output(self.model.density(t, x, paired).cpu().numpy())

    def predict_hazard(self, t: Tensor, x: Tensor, paired: bool | None = None) -> Tensor:
        if self.model is None:
            raise RuntimeError("fit() must be called before predict_hazard()")
        t, x = (torch.as_tensor(v, dtype=torch.float32) for v in (t, x))
        dev = self._device()
        t, x = t.to(dev), x.to(dev)
        with torch.no_grad():
            return as_output(self.model.hazard(t, x, paired).cpu().numpy())

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
    _conditional_flow: bool = True

    def _make_model(self, n_features: int, **hyper) -> FlowSurvAFT:
        return FlowSurvAFT(
            n_features,
            hidden=hyper.get("hidden", 128),
            n_blocks=hyper.get("n_blocks", 2),
            dropout=hyper.get("dropout", 0.1),
            bins=hyper.get("bins", 8),
            bound=hyper.get("bound", 4.0),  # prereg Sec. 4: spline domain [-4, 4]
            n_spline_blocks=hyper.get("n_spline_blocks", 1),
            conditional_flow=self._conditional_flow,
        )


class FlowSurvStrictAFTMethod(FlowSurvAFTMethod):
    """Strict-AFT ablation: unconditional residual flow (eps independent of x).

    Identical to FlowSurv-AFT except the spline parameters are global, so
    exp(mu(a) - mu(b)) is exactly an AFT time ratio (supervisor review item 3,
    used for H4).
    """

    name: str = "flowsurv_strict_aft"
    _conditional_flow: bool = False

    def __init__(self) -> None:
        super().__init__()
        self.name = "flowsurv_strict_aft"


class FlowSurvGaussMethod(_FlowSurvWrapper):
    """FlowSurv-Gauss ablation: identity flow with a Gaussian residual law."""

    name: str = "flowsurv_gauss"
    _cls = FlowSurvGauss
    _spline_l2: float = 0.0  # no spline in the identity-flow ablations

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


class FlowSurvGumbelMethod(FlowSurvGaussMethod):
    """Identity-flow minimum-Gumbel ablation: a Weibull AFT with deep mu(x), sigma(x).

    Diagnostic for Deviation 1 (supervisor review item 2): scenarios whose
    covariate effect enters only through mu(x) and sigma(x) around a fixed
    residual law need no flow conditioning.
    """

    name: str = "flowsurv_gumbel"
    _cls = FlowSurvGumbel

    def __init__(self) -> None:
        super().__init__()
        self.name = "flowsurv_gumbel"

    def _make_model(self, n_features: int, **hyper) -> FlowSurvGumbel:
        return FlowSurvGumbel(
            n_features,
            hidden=hyper.get("hidden", 128),
            n_blocks=hyper.get("n_blocks", 2),
            dropout=hyper.get("dropout", 0.1),
        )
