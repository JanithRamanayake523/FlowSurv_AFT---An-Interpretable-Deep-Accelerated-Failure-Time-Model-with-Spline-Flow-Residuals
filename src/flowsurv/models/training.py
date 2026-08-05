"""Training loop for FlowSurv-AFT (optimization protocol of Methodology Sec. 2.4).

AdamW + cosine decay + gradient clipping, early stopping on validation NLL
with best-weights restore. Shared by the Phase 1 gate tests / smoke fit and
reused later by the tuning harness (eval/tune.py, Phase 3).
"""

from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass, field

import torch
from torch import Tensor

from .losses import right_censored_nll


@dataclass
class TrainConfig:
    lr: float = 1e-3
    final_lr: float = 1e-5
    weight_decay: float = 1e-4
    batch_size: int = 256
    max_epochs: int = 500
    patience: int = 30
    grad_clip: float = 1.0
    val_frac: float = 0.15
    lambda_softna: float = 0.0
    seed: int = 0
    device: str = "cpu"
    verbose: bool = False


@dataclass
class TrainResult:
    train_nll: list[float] = field(default_factory=list)
    val_nll: list[float] = field(default_factory=list)
    best_epoch: int = -1
    best_val_nll: float = math.inf
    epochs_ran: int = 0
    wall_time_s: float = 0.0


def fit(model, t: Tensor, d: Tensor, x: Tensor, config: TrainConfig | None = None) -> TrainResult:
    """Fit ``model`` by right-censored maximum likelihood; returns history."""
    cfg = config or TrainConfig()
    device = torch.device(cfg.device)
    model.to(device)
    t, d, x = t.to(device).float(), d.to(device).float(), x.to(device).float()

    gen = torch.Generator(device="cpu").manual_seed(cfg.seed)
    perm = torch.randperm(t.shape[0], generator=gen)
    n_val = max(1, int(round(cfg.val_frac * t.shape[0])))
    idx_val, idx_tr = perm[:n_val].to(device), perm[n_val:].to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cfg.max_epochs, eta_min=cfg.final_lr
    )

    result = TrainResult()
    best_state = None
    stale = 0
    start = time.perf_counter()

    for epoch in range(cfg.max_epochs):
        model.train()
        order = idx_tr[torch.randperm(idx_tr.shape[0], generator=gen)]
        epoch_nll, n_seen = 0.0, 0
        for batch in order.split(cfg.batch_size):
            if cfg.lambda_softna > 0:
                from .losses import total_nll

                loss = total_nll(model, t[batch], d[batch], x[batch], cfg.lambda_softna)
            else:
                loss = right_censored_nll(model, t[batch], d[batch], x[batch])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            epoch_nll += loss.item() * batch.shape[0]
            n_seen += batch.shape[0]
        sched.step()

        model.eval()
        with torch.no_grad():
            val_nll = right_censored_nll(model, t[idx_val], d[idx_val], x[idx_val]).item()

        result.train_nll.append(epoch_nll / n_seen)
        result.val_nll.append(val_nll)
        result.epochs_ran = epoch + 1

        if cfg.verbose and (epoch % 10 == 0 or epoch == cfg.max_epochs - 1):
            print(f"epoch {epoch:4d}  train NLL {result.train_nll[-1]:.4f}  val NLL {val_nll:.4f}")

        if val_nll < result.best_val_nll - 1e-6:
            result.best_val_nll, result.best_epoch = val_nll, epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= cfg.patience:
                break

    result.wall_time_s = time.perf_counter() - start
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return result
