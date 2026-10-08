"""优化器与学习率调度（按 epoch 步进）。"""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn


def build_optimizer(hp: Mapping[str, Any], model: nn.Module) -> torch.optim.Optimizer:
    name = str(hp.get("optimizer", "adam")).lower()
    lr = float(hp["learning_rate"])
    weight_decay = float(hp.get("weight_decay", 0.0))
    params = model.parameters()
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=float(hp.get("momentum", 0.9)), weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"未知 optimizer: {name}（可选 sgd / adam / adamw）")


def build_scheduler(hp: Mapping[str, Any], optimizer: torch.optim.Optimizer, num_epochs: int):
    name = str(hp.get("lr_scheduler", "") or "none").lower()
    if name == "none":
        return None
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=int(hp.get("lr_step_size", 50)), gamma=float(hp.get("lr_gamma", 0.5))
        )
    if name == "poly":
        power = float(hp.get("lr_poly_power", 0.9))
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda epoch: (1.0 - min(epoch, num_epochs) / num_epochs) ** power
        )
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=int(hp.get("lr_cosine_t_max", num_epochs)), eta_min=float(hp.get("lr_min", 1e-6))
        )
    if name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max",
            factor=float(hp.get("lr_plateau_factor", 0.5)),
            patience=int(hp.get("lr_plateau_patience", 15)),
            min_lr=float(hp.get("lr_min", 1e-6)),
        )
    raise ValueError(f"未知 lr_scheduler: {name}（可选 none / step / poly / cosine / plateau）")


def step_scheduler(scheduler, metric: float) -> None:
    if scheduler is None:
        return
    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step(metric)
    else:
        scheduler.step()
