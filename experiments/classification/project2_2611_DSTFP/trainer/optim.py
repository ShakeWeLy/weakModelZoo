"""优化器 / 学习率调度占位。"""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn


def build_optimizer(model: nn.Module, cfg: Mapping[str, Any]) -> torch.optim.Optimizer:
    hp = cfg["hyperparameters"]
    name = str(hp.get("optimizer", "adamw")).lower()
    lr = float(hp.get("learning_rate", 1e-3))
    wd = float(hp.get("weight_decay", 0.0))
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=float(hp.get("momentum", 0.9)), weight_decay=wd)
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    raise ValueError(f"未知 optimizer: {name}")
