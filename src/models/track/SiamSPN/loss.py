"""SiamSPN++ 损失：均衡正负样本的分类 NLL + 加权 L1 回归。

对齐 pysot ``models/loss.py``（select_cross_entropy_loss + weight_l1_loss）。

约定：
- ``cls`` 训练前需经 ``log_softmax``，形状 ``(B, 2*A, H, W)`` 或已展平为 ``(N, 2)``
- ``label_cls``: ``(B, A, H, W)``，取值 0/1/-1（-1 忽略）
- ``label_loc``: ``(B, 4, A, H, W)``
- ``label_loc_weight``: ``(B, A, H, W)``，仅正样本处非零
"""

from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F


def _select_nll(pred: torch.Tensor, label: torch.Tensor, select: torch.Tensor) -> torch.Tensor:
    if select.numel() == 0:
        return pred.new_zeros(())
    pred = pred.index_select(0, select)
    label = label.index_select(0, select)
    return F.nll_loss(pred, label)


def select_cross_entropy_loss(pred: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    """正负样本各算一次 NLL 再平均，避免负样本淹没。"""
    pred = pred.reshape(-1, 2)
    label = label.reshape(-1)
    pos = (label == 1).nonzero(as_tuple=False).squeeze(1)
    neg = (label == 0).nonzero(as_tuple=False).squeeze(1)
    return 0.5 * _select_nll(pred, label, pos) + 0.5 * _select_nll(pred, label, neg)


def weight_l1_loss(
    pred_loc: torch.Tensor,
    label_loc: torch.Tensor,
    loss_weight: torch.Tensor,
) -> torch.Tensor:
    """按 ``loss_weight`` 加权的 L1 框回归。

    Args:
        pred_loc: ``(B, 4*A, H, W)``
        label_loc: ``(B, 4, A, H, W)``
        loss_weight: ``(B, A, H, W)``
    """
    b, _, sh, sw = pred_loc.shape
    pred = pred_loc.view(b, 4, -1, sh, sw)
    diff = (pred - label_loc).abs().sum(dim=1)  # (B, A, H, W)
    return (diff * loss_weight).sum() / b


class SiamSPNLoss(nn.Module):
    """SiamSPN++ 联合损失。

    Args:
        hyp: 可选超参，键 ``cls`` / ``loc`` 为两项权重（默认 1.0 / 1.2）。
    """

    item_names: tuple[str, ...] = ("cls", "loc")

    def __init__(self, hyp: Mapping[str, Any] | None = None) -> None:
        super().__init__()
        hyp = hyp or {}
        self.cls_weight = float(hyp.get("cls", 1.0))
        self.loc_weight = float(hyp.get("loc", 1.2))

    def describe(self) -> str:
        return f"损失 cls={self.cls_weight:.3f} loc={self.loc_weight:.3f}"

    @staticmethod
    def log_softmax(cls: torch.Tensor) -> torch.Tensor:
        """``(B, 2*A, H, W) -> (B, A, H, W, 2)`` 的 log_softmax。"""
        b, a2, h, w = cls.shape
        cls = cls.view(b, 2, a2 // 2, h, w).permute(0, 2, 3, 4, 1).contiguous()
        return F.log_softmax(cls, dim=4)

    def forward(
        self,
        cls: torch.Tensor,
        loc: torch.Tensor,
        label_cls: torch.Tensor,
        label_loc: torch.Tensor,
        label_loc_weight: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        cls_log = self.log_softmax(cls)
        cls_loss = select_cross_entropy_loss(cls_log, label_cls)
        loc_loss = weight_l1_loss(loc, label_loc, label_loc_weight)
        total = self.cls_weight * cls_loss + self.loc_weight * loc_loss
        return {
            "loss": total,
            "cls": cls_loss.detach(),
            "loc": loc_loss.detach(),
        }
