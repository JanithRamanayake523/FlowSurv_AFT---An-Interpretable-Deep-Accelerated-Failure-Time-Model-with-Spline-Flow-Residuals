"""Phase 1 smoke fit + inference micro-benchmark (Implementation Plan Phase 1, task 6).

Smoke-train FlowSurv-AFT on one S1 cell (n = 1000, 20% Type-I censoring):
  - confirm validation NLL decreases,
  - confirm the linear-encoder fit recovers the AFT slopes -beta/k within
    Monte-Carlo error,
  - save true-vs-estimated survival / hazard curves for visual inspection,
  - log the inference micro-benchmark (f, S, quantile, sample wall-clock
    per 1k subjects).

Run:  python -m flowsurv.eval.smoke
Outputs (git-ignored): experiments/phase1_smoke/{smoke.json, benchmark.json, *.png}
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from ..data.censoring import censor_type1
from ..data.dgp import S1_BETA, S1_K, simulate_s1
from ..models import FlowSurvAFT, TrainConfig, fit

OUT_DIR = Path(__file__).resolve().parents[3] / "experiments" / "phase1_smoke"
N = 1000
CENSORING = 0.20
SEED = 2026

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def true_survival(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    beta = torch.tensor(S1_BETA)
    lin = x[..., :5] @ beta
    return torch.exp(-(t.unsqueeze(-1) / 1.0) ** S1_K * lin.exp()).squeeze(-1) if t.dim() > x.dim() - 1 else torch.exp(-t**S1_K * lin.exp())


def true_hazard(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    beta = torch.tensor(S1_BETA)
    lin = x[..., :5] @ beta
    return S1_K * t ** (S1_K - 1) * lin.exp()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)

    # ---- one S1 cell: n = 1000, 20% Type-I censoring -------------------
    t_event, x = simulate_s1(N, seed=SEED)
    t, d = censor_type1(t_event, CENSORING)
    print(f"cell: n={N}, achieved censoring={1 - d.mean().item():.3f} (target {CENSORING}), device={DEVICE}")

    cfg = TrainConfig(batch_size=256, max_epochs=300, patience=30, seed=SEED, device=DEVICE, verbose=True)

    # linear encoder: beta recovery is directly readable from the mu head
    linear_model = FlowSurvAFT(x.shape[1], n_blocks=0)
    res_lin = fit(linear_model, t, d, x, cfg)

    # default MLP encoder: the configuration used in experiments
    mlp_model = FlowSurvAFT(x.shape[1], n_blocks=2, hidden=128, bins=8)
    res_mlp = fit(mlp_model, t, d, x, cfg)

    # ---- report ---------------------------------------------------------
    w = linear_model.encoder.mu_head.weight.detach().cpu().squeeze(0)
    true_slopes = -torch.tensor(S1_BETA) / S1_K
    report = {
        "cell": {"scenario": "S1", "n": N, "censoring_target": CENSORING, "type": "I", "seed": SEED},
        "device": DEVICE,
        "linear_encoder": {
            "val_nll_first": res_lin.val_nll[0],
            "val_nll_best": res_lin.best_val_nll,
            "epochs_ran": res_lin.epochs_ran,
            "wall_time_s": res_lin.wall_time_s,
            "estimated_slopes": w[:5].tolist(),
            "true_slopes(-beta/k)": true_slopes.tolist(),
            "max_active_slope_error": (w[:5] - true_slopes).abs().max().item(),
            "max_inactive_slope_abs": w[5:].abs().max().item(),
        },
        "mlp_encoder": {
            "val_nll_first": res_mlp.val_nll[0],
            "val_nll_best": res_mlp.best_val_nll,
            "epochs_ran": res_mlp.epochs_ran,
            "wall_time_s": res_mlp.wall_time_s,
        },
    }
    (OUT_DIR / "smoke.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

    # ---- figures: true vs estimated curves for 3 profiles ---------------
    profiles = torch.zeros(3, x.shape[1])
    profiles[1, :5] = torch.tensor(S1_BETA)  # high linear predictor
    profiles[2, :5] = -torch.tensor(S1_BETA)  # low linear predictor
    grid = torch.linspace(0.02, 4.0, 400)
    dev = torch.device(DEVICE)
    profiles_dev, grid_dev = profiles.to(dev), grid.to(dev)
    xp = profiles_dev.unsqueeze(0).expand(grid.numel(), -1, -1)

    with torch.no_grad():
        s_lin = linear_model.survival(grid_dev.unsqueeze(1), xp).cpu()
        s_mlp = mlp_model.survival(grid_dev.unsqueeze(1), xp).cpu()
        h_lin = linear_model.hazard(grid_dev.unsqueeze(1), xp).cpu()
        h_mlp = mlp_model.hazard(grid_dev.unsqueeze(1), xp).cpu()
    s_true = torch.stack([true_survival(grid, profiles[j]) for j in range(3)], dim=1)
    h_true = torch.stack([true_hazard(grid, profiles[j]) for j in range(3)], dim=1)

    labels = ["x = 0", "x_active = +beta", "x_active = -beta"]
    for name, est_a, est_b, truth, ylab in [
        ("survival", s_lin, s_mlp, s_true, "S(t|x)"),
        ("hazard", h_lin, h_mlp, h_true, "h(t|x)"),
    ]:
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharex=True)
        for j, ax in enumerate(axes):
            ax.plot(grid, truth[:, j], "k-", lw=2, label="true")
            ax.plot(grid, est_a[:, j], "C0--", label="FlowSurv (linear)")
            ax.plot(grid, est_b[:, j], "C1--", label="FlowSurv (MLP)")
            ax.set_title(labels[j])
            ax.set_xlabel("t")
            ax.set_ylabel(ylab)
            if name == "hazard":
                ax.set_ylim(0, min(h_true.max().item() * 1.5, 10))
        axes[0].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"{name}_curves.png", dpi=120)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(res_lin.val_nll, "C0-", label="linear")
    ax.plot(res_mlp.val_nll, "C1-", label="MLP")
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation NLL")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "val_nll.png", dpi=120)
    plt.close(fig)

    # ---- inference micro-benchmark (per 1k subjects) --------------------
    n_bench = 1000
    xb = torch.randn(n_bench, x.shape[1], device=dev)
    tb = torch.rand(n_bench, device=dev) * 3 + 0.05
    bench = {"device": DEVICE, "n_subjects": n_bench, "repeats": 5}
    with torch.no_grad():
        for key, fn in [
            ("density", lambda: mlp_model.density(tb, xb)),
            ("survival", lambda: mlp_model.survival(tb, xb)),
            ("quantile", lambda: mlp_model.quantile(0.5, xb)),
            ("sample_1000", lambda: mlp_model.sample(xb, n=1000)),
        ]:
            fn()  # warm-up
            times = []
            for _ in range(bench["repeats"]):
                if DEVICE == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                fn()
                if DEVICE == "cuda":
                    torch.cuda.synchronize()
                times.append(time.perf_counter() - t0)
            bench[f"{key}_ms_per_1k"] = 1e3 * min(times)
    (OUT_DIR / "benchmark.json").write_text(json.dumps(bench, indent=2))
    print(json.dumps(bench, indent=2))
    print(f"artifacts written to {OUT_DIR}")


if __name__ == "__main__":
    main()
