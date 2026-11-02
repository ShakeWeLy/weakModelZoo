"""YOLOv5 损失：CIoU 框回归 + 目标性 BCE + 类别 BCE，与官方 ``utils/loss.py::ComputeLoss`` 对齐。

要点：
- 目标分配：按锚框宽高比过滤（anchor_t），并把目标同时分配给中心格子及其上下左右相邻格子（offset 0.5），
  使每个目标在每个尺度上有 3 个正样本格子。
- 框回归：预测框经 sigmoid 解码后与真值计算 CIoU，损失 = 1 - CIoU；obj 目标值 = 对应 CIoU（detach、clamp≥0）。
- 三项损失按 ``balance`` 对 P3 / P4 / P5 加权，并按尺度数、类别数、输入尺寸缩放。

targets 约定：形状 (nt, 6)，列为 (img_idx, cls, x_center, y_center, w, h)，坐标均归一化到 [0, 1]。
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.detection.YOLO.box import bbox_iou
from src.models.detection.YOLO.yolov5 import Detect


class YOLOv5Loss(nn.Module):
    """
    Args:
        detect: 模型的 Detect 头（用于读取 anchors / stride / nc / nl / na）。
        hyp: 损失超参，键：box / obj / cls / anchor_t / balance / obj_pw / cls_pw。
        img_size: 训练输入尺寸，用于缩放 obj 损失权重（官方按 640 归一化）。
    """

    item_names: tuple[str, ...] = ("box", "obj", "cls")

    def __init__(self, detect: Detect, hyp: Mapping[str, Any], img_size: int = 640) -> None:
        super().__init__()
        self.nc, self.nl, self.na = detect.nc, detect.nl, detect.na
        self.anchor_t = float(hyp.get("anchor_t", 4.0))
        self.balance: Sequence[float] = [float(v) for v in hyp.get("balance", [4.0, 1.0, 0.4])]
        if len(self.balance) != self.nl:
            raise ValueError(f"loss.balance 需要 {self.nl} 个值（每个检测尺度一个）")

        # 与官方一致：损失增益按尺度数归一化
        scale = 3.0 / self.nl
        self.box_gain = float(hyp.get("box", 0.05)) * scale
        self.cls_gain = float(hyp.get("cls", 0.5)) * scale * (self.nc / 80)
        self.obj_gain = float(hyp.get("obj", 1.0)) * scale * (img_size / 640) ** 2

        # 锚框换算到特征图格子单位，形状 (nl, na, 2)
        anchors_grid = detect.anchors.detach().clone() / detect.stride.view(-1, 1, 1)
        self.register_buffer("anchors_grid", anchors_grid)
        self.register_buffer("cls_pw", torch.tensor([float(hyp.get("cls_pw", 1.0))]))
        self.register_buffer("obj_pw", torch.tensor([float(hyp.get("obj_pw", 1.0))]))

    def describe(self) -> str:
        return (
            f"损失 box={self.box_gain:.4f} obj={self.obj_gain:.4f} cls={self.cls_gain:.4f} | "
            f"balance={list(self.balance)} | anchor_t={self.anchor_t}"
        )

    def _bce(self, logits: torch.Tensor, target: torch.Tensor, pos_weight: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)

    def build_targets(self, p: list[torch.Tensor], targets: torch.Tensor):
        """把真值分配到各尺度的正样本格子。返回每尺度的 (tcls, tbox, indices, anchors)。"""
        device = targets.device
        na, nt = self.na, targets.shape[0]
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=device)
        ai = torch.arange(na, device=device, dtype=torch.float32).view(na, 1).repeat(1, nt)
        # (na, nt, 7)：最后一列为锚框索引
        t_all = torch.cat((targets.repeat(na, 1, 1), ai[..., None]), dim=2)
        g = 0.5  # 相邻格子偏移阈值
        off = torch.tensor([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]], device=device, dtype=torch.float32) * g

        for i in range(self.nl):
            anchors = self.anchors_grid[i]
            ny, nx = p[i].shape[2], p[i].shape[3]
            gain[2:6] = torch.tensor([nx, ny, nx, ny], device=device, dtype=torch.float32)
            t = t_all * gain

            if nt:
                # 按宽高比过滤：目标与锚框的宽高比都要在 anchor_t 之内
                r = t[..., 4:6] / anchors[:, None]
                j = torch.max(r, 1 / r).max(2)[0] < self.anchor_t
                t = t[j]
                gxy = t[:, 2:4]
                gxi = gain[[2, 3]] - gxy
                j1, k1 = ((gxy % 1 < g) & (gxy > 1)).T
                l1, m1 = ((gxi % 1 < g) & (gxi > 1)).T
                mask = torch.stack((torch.ones_like(j1), j1, k1, l1, m1))
                t = t.repeat((5, 1, 1))[mask]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[mask]
            else:
                t = t_all[0]
                offsets = 0

            b = t[:, 0].long()
            c = t[:, 1].long()
            gxy = t[:, 2:4]
            gwh = t[:, 4:6]
            gij = (gxy - offsets).long()
            gi, gj = gij.T
            a = t[:, 6].long()
            gi = gi.clamp(0, nx - 1)
            gj = gj.clamp(0, ny - 1)

            indices.append((b, a, gj, gi))
            tbox.append(torch.cat((gxy - gij, gwh), dim=1))
            anch.append(anchors[a])
            tcls.append(c)
        return tcls, tbox, indices, anch

    def forward(self, p: list[torch.Tensor], targets: torch.Tensor):
        """返回 (总损失, 分项损失 [box, obj, cls] 的 detach 结果)。总损失已乘以 batch size。"""
        device = targets.device
        lbox = torch.zeros((), device=device)
        lobj = torch.zeros((), device=device)
        lcls = torch.zeros((), device=device)
        tcls, tbox, indices, anchors = self.build_targets(p, targets)

        for i, pi in enumerate(p):
            b, a, gj, gi = indices[i]
            tobj = torch.zeros_like(pi[..., 0])
            n = b.shape[0]
            if n:
                ps = pi[b, a, gj, gi]
                pxy = ps[:, :2].sigmoid() * 2 - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), dim=1)
                iou = bbox_iou(pbox, tbox[i], xywh=True, CIoU=True).squeeze(-1)
                lbox = lbox + (1.0 - iou).mean()
                tobj[b, a, gj, gi] = iou.detach().clamp(0).to(tobj.dtype)
                if self.nc > 1:
                    t = torch.zeros_like(ps[:, 5:])
                    t[torch.arange(n, device=device), tcls[i]] = 1.0
                    lcls = lcls + self._bce(ps[:, 5:], t, self.cls_pw)
            lobj = lobj + self._bce(pi[..., 4], tobj, self.obj_pw) * self.balance[i]

        lbox = lbox * self.box_gain
        lobj = lobj * self.obj_gain
        lcls = lcls * self.cls_gain
        bs = p[0].shape[0]
        loss = (lbox + lobj + lcls) * bs
        return loss, torch.stack((lbox, lobj, lcls)).detach()
