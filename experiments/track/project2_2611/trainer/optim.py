"""优化器与学习率调度；支持主干 / 头部分组与冻结后重建。"""

from __future__ import annotations

import math
from typing import Any, Mapping

import torch
from torch import nn

from src.models.track.SiamSPN import SiamSPN_pp


def build_optimizer(
    hp: Mapping[str, Any],
    model: SiamSPN_pp,
    *,
    train_backbone: bool,
) -> torch.optim.Optimizer:
    name = str(hp.get("optimizer", "sgd")).lower()
    lr = float(hp["learning_rate"])
    wd = float(hp.get("weight_decay", 0.0))
    momentum = float(hp.get("momentum", 0.9))
    bb_mult = float(hp.get("backbone_lr_mult", 0.1))

    head_params = [p for p in list(model.neck.parameters()) + list(model.rpn_head.parameters()) if p.requires_grad]
    groups: list[dict[str, Any]] = [{"params": head_params, "lr": lr, "weight_decay": wd}]
    if train_backbone:
        bb_params = [p for p in model.backbone.parameters() if p.requires_grad]
        if bb_params:
            groups.append({"params": bb_params, "lr": lr * bb_mult, "weight_decay": wd})

    if name == "sgd":
        return torch.optim.SGD(groups, lr=lr, momentum=momentum, weight_decay=0.0)
    if name == "adam":
        return torch.optim.Adam(groups, lr=lr, betas=(momentum, 0.999))
    if name == "adamw":
        return torch.optim.AdamW(groups, lr=lr, betas=(momentum, 0.999))
    raise ValueError(f"未知 optimizer: {name}（可选 sgd / adam / adamw）")


def build_scheduler(hp: Mapping[str, Any], optimizer: torch.optim.Optimizer, num_epochs: int):
    name = str(hp.get("lr_scheduler", "none")).lower()
    lrf = float(hp.get("lr_final_factor", 0.1))
    if name == "none":
        return None
    if name == "linear":
        fn = lambda e: (1 - e / max(num_epochs, 1)) * (1 - lrf) + lrf  # noqa: E731
    elif name == "cosine":
        fn = lambda e: ((1 - math.cos(e * math.pi / max(num_epochs, 1))) / 2) * (lrf - 1) + 1  # noqa: E731
    else:
        raise ValueError(f"未知 lr_scheduler: {name}（可选 none / linear / cosine）")
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=fn)


def step_scheduler(scheduler) -> None:
    if scheduler is not None:
        scheduler.step()
