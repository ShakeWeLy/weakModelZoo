"""Dice + CrossEntropy 组合损失。"""

from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch import nn


class DiceLoss(nn.Module):
    """多类 soft Dice（忽略背景）；``skip_absent_classes`` 时只对 batch 中出现的类别求平均。"""

    def __init__(self, num_classes: int, *, class_weights: torch.Tensor | None = None,
                 skip_absent_classes: bool = True, eps: float = 1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.skip_absent_classes = skip_absent_classes
        self.eps = eps
        if class_weights is not None and class_weights.numel() != num_classes - 1:
            raise ValueError(f"dice_class_weights 需要 {num_classes - 1} 个值，当前 {class_weights.numel()}")
        self.register_buffer("class_weights", class_weights.float() if class_weights is not None else None)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)[:, 1:]
        one_hot = F.one_hot(targets, self.num_classes).permute(0, 3, 1, 2).float()[:, 1:]
        dims = (0, 2, 3)
        intersection = (probs * one_hot).sum(dims)
        cardinality = (probs + one_hot).sum(dims)
        loss = 1.0 - (2.0 * intersection + self.eps) / (cardinality + self.eps)
        weights = self.class_weights if self.class_weights is not None else torch.ones_like(loss)
        if self.skip_absent_classes:
            present = one_hot.sum(dims) > 0
            if not present.any():
                return logits.sum() * 0.0
            loss, weights = loss[present], weights[present]
        return (loss * weights).sum() / weights.sum()


class DiceCELoss(nn.Module):
    def __init__(self, num_classes: int, *, dice_weight: float = 1.0, ce_weight: float = 1.0,
                 dice_class_weights: torch.Tensor | None = None, ce_class_weights: torch.Tensor | None = None,
                 skip_absent_classes: bool = True):
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.dice = DiceLoss(num_classes, class_weights=dice_class_weights,
                             skip_absent_classes=skip_absent_classes)
        self.ce = nn.CrossEntropyLoss(weight=ce_class_weights)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss = logits.new_zeros(())
        if self.dice_weight:
            loss = loss + self.dice_weight * self.dice(logits, targets)
        if self.ce_weight:
            loss = loss + self.ce_weight * self.ce(logits, targets)
        return loss

    def describe(self) -> str:
        weighted = self.dice.class_weights is not None or self.ce.weight is not None
        return (
            f"DiceCE(dice_weight={self.dice_weight}, ce_weight={self.ce_weight}, "
            f"class_weights={'on' if weighted else 'off'}, "
            f"skip_absent_classes={'on' if self.dice.skip_absent_classes else 'off'})"
        )


def build_loss(train_cfg: Mapping[str, Any], num_classes: int) -> DiceCELoss:
    # 类别权重开关关闭时，忽略 dice_class_weights / ce_class_weights，所有类等权
    use_class_weights = bool(train_cfg.get("use_class_weights", False))
    dice_weights = (train_cfg.get("dice_class_weights") or None) if use_class_weights else None
    ce_weights = (train_cfg.get("ce_class_weights") or None) if use_class_weights else None
    dice_t = torch.tensor(dice_weights, dtype=torch.float32) if dice_weights else None
    if ce_weights:
        if len(ce_weights) != num_classes:
            raise ValueError(f"ce_class_weights 需要 {num_classes} 个值，当前 {len(ce_weights)}")
        ce_t = torch.tensor(ce_weights, dtype=torch.float32)
    elif dice_t is not None:
        ce_t = torch.cat([torch.ones(1), dice_t])
    else:
        ce_t = None
    return DiceCELoss(
        num_classes,
        dice_weight=float(train_cfg.get("dice_loss_weight", 1.0)),
        ce_weight=float(train_cfg.get("ce_loss_weight", 1.0)),
        dice_class_weights=dice_t,
        ce_class_weights=ce_t,
        skip_absent_classes=bool(train_cfg.get("skip_absent_classes", True)),
    )
