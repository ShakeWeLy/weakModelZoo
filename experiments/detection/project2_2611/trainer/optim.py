"""优化器与学习率调度（按 epoch 步进）。

参数分组与官方 YOLOv5 一致：BN 权重与所有 bias 不做 weight decay，其余卷积权重才衰减。
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import torch
from torch import nn


def _param_groups(model: nn.Module, weight_decay: float) -> list[dict[str, Any]]:
    g_bn, g_w, g_b = [], [], []
    for module in model.modules():
        for name, p in module.named_parameters(recurse=False):
            if not p.requires_grad:
                continue
            if name == "bias":
                g_b.append(p)
            elif isinstance(module, nn.BatchNorm2d):
                g_bn.append(p)
            else:
                g_w.append(p)
    return [
        {"params": g_bn, "weight_decay": 0.0},
        {"params": g_w, "weight_decay": weight_decay},
        {"params": g_b, "weight_decay": 0.0},
    ]


def build_optimizer(hp: Mapping[str, Any], model: nn.Module) -> torch.optim.Optimizer:
    name = str(hp.get("optimizer", "sgd")).lower()
    lr = float(hp["learning_rate"])
    wd = float(hp.get("weight_decay", 0.0))
    momentum = float(hp.get("momentum", 0.937))
    groups = _param_groups(model, wd)
    if name == "sgd":
        return torch.optim.SGD(groups, lr=lr, momentum=momentum, nesterov=True)
    if name == "adam":
        return torch.optim.Adam(groups, lr=lr, betas=(momentum, 0.999))
    if name == "adamw":
        return torch.optim.AdamW(groups, lr=lr, betas=(momentum, 0.999))
    raise ValueError(f"未知 optimizer: {name}（可选 sgd / adam / adamw）")


def build_scheduler(hp: Mapping[str, Any], optimizer: torch.optim.Optimizer, num_epochs: int):
    name = str(hp.get("lr_scheduler", "none")).lower()
    lrf = float(hp.get("lr_final_factor", 0.01))
    if name == "none":
        return None
    if name == "linear":
        fn = lambda e: (1 - e / num_epochs) * (1 - lrf) + lrf  # noqa: E731
    elif name == "cosine":
        fn = lambda e: ((1 - math.cos(e * math.pi / num_epochs)) / 2) * (lrf - 1) + 1  # noqa: E731
    else:
        raise ValueError(f"未知 lr_scheduler: {name}（可选 none / linear / cosine）")
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=fn)


def step_scheduler(scheduler) -> None:
    if scheduler is not None:
        scheduler.step()
