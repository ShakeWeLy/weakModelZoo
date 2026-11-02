"""YOLOv8 损失：Task-Aligned 分配 + CIoU 框回归 + DFL + 分类 BCE（无单独 objectness）。"""

from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.detection.YOLO.box import bbox_iou, xywh2xyxy
from src.models.detection.YOLO.tal import TaskAlignedAssigner, bbox2dist, dist2bbox, make_anchors
from src.models.detection.YOLO.yolov8 import Detect


class DFLoss(nn.Module):
    def __init__(self, reg_max: int = 16) -> None:
        super().__init__()
        self.reg_max = reg_max

    def forward(self, pred_dist: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """pred_dist: (n, reg_max)；target: (n,) 连续距离。"""
        target = target.clamp(0, self.reg_max - 1 - 0.01)
        tl = target.long()
        tr = tl + 1
        wl = tr - target
        wr = 1 - wl
        logp = F.log_softmax(pred_dist, dim=1)
        return -(logp.gather(1, tl.view(-1, 1)).view(-1) * wl + logp.gather(1, tr.view(-1, 1)).view(-1) * wr)


class YOLOv8Loss(nn.Module):
    """
    Args:
        detect: 模型 Detect 头。
        hyp: 键 box / cls / dfl / tal_topk / tal_alpha / tal_beta。
        img_size: 训练输入边长（占位，与日志接口一致）。
    """

    item_names: tuple[str, ...] = ("box", "cls", "dfl")

    def __init__(self, detect: Detect, hyp: Mapping[str, Any], img_size: int = 640) -> None:
        super().__init__()
        self.nc = detect.nc
        self.reg_max = detect.reg_max
        self.stride = detect.stride
        self.img_size = img_size
        self.box_gain = float(hyp.get("box", 7.5))
        self.cls_gain = float(hyp.get("cls", 0.5))
        self.dfl_gain = float(hyp.get("dfl", 1.5))
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.assigner = TaskAlignedAssigner(
            topk=int(hyp.get("tal_topk", 10)),
            num_classes=self.nc,
            alpha=float(hyp.get("tal_alpha", 0.5)),
            beta=float(hyp.get("tal_beta", 6.0)),
            stride_val=float(self.stride[1]) if len(self.stride) > 1 else float(self.stride[0]),
        )
        self.dfl_loss = DFLoss(self.reg_max) if self.reg_max > 1 else None
        self.register_buffer("proj", torch.arange(self.reg_max, dtype=torch.float32))

    def describe(self) -> str:
        return (
            f"损失 box={self.box_gain:.4f} cls={self.cls_gain:.4f} dfl={self.dfl_gain:.4f} | "
            f"TAL topk={self.assigner.topk} α={self.assigner.alpha} β={self.assigner.beta} | reg_max={self.reg_max}"
        )

    def _preprocess(self, targets: torch.Tensor, batch_size: int, imgsz_hw: torch.Tensor) -> torch.Tensor:
        """targets (nt,6)=img,cls,cx,cy,w,h 归一化 → (bs, max_gt, 5)=cls,xyxy 像素。"""
        if targets.numel() == 0:
            return torch.zeros(batch_size, 0, 5, device=targets.device)
        batch_idx = targets[:, 0].long()
        _, counts = batch_idx.unique(return_counts=True)
        n = int(counts.max())
        out = torch.zeros(batch_size, n, 5, device=targets.device)
        for i in range(batch_size):
            m = batch_idx == i
            if not m.any():
                continue
            t = targets[m, 1:].clone()
            t[:, 1:5] = xywh2xyxy(t[:, 1:5]) * imgsz_hw[[1, 0, 1, 0]]
            out[i, : t.shape[0]] = t
        return out

    def _decode_bboxes(self, anchor_points: torch.Tensor, pred_dist: torch.Tensor) -> torch.Tensor:
        """pred_dist (b, N, 4*reg_max) → xyxy 格子单位。"""
        if self.dfl_loss is not None:
            b, a, c = pred_dist.shape
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def forward(self, preds: dict[str, torch.Tensor], targets: torch.Tensor):
        device = targets.device
        pred_distri = preds["boxes"].permute(0, 2, 1).contiguous()   # (b, N, 4*reg_max)
        pred_scores = preds["scores"].permute(0, 2, 1).contiguous()  # (b, N, nc)
        feats = preds["feats"]
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)
        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=device, dtype=dtype) * self.stride[0]

        tgt = self._preprocess(targets.to(device), batch_size, imgsz)
        gt_labels, gt_bboxes = tgt.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        pred_bboxes = self._decode_bboxes(anchor_points, pred_distri)  # 格子单位 xyxy

        # TAL 在 fp32 下做，避免 AMP half 的 IoU / 指数产生 NaN 后经 nan*0 污染软标签
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().float().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).float(),
            (anchor_points * stride_tensor).float(),
            gt_labels.float(),
            gt_bboxes.float(),
            mask_gt.float(),
        )
        target_scores = torch.nan_to_num(target_scores, nan=0.0)

        target_scores_sum = target_scores.sum().clamp_(min=1)
        loss = torch.zeros(3, device=device)

        # cls
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum

        # box + dfl
        if fg_mask.sum():
            weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
            tgt_boxes = target_bboxes / stride_tensor
            iou = bbox_iou(pred_bboxes[fg_mask], tgt_boxes[fg_mask], xywh=False, CIoU=True)
            loss[0] = ((1.0 - iou) * weight).sum() / target_scores_sum

            if self.dfl_loss is not None:
                target_ltrb = bbox2dist(anchor_points, tgt_boxes, self.reg_max)
                pred_dist = pred_distri[fg_mask].view(-1, self.reg_max)
                target_ltrb = target_ltrb[fg_mask].view(-1)
                # 四边展平后与 weight 对齐：每条边共享同一权重
                dfl = self.dfl_loss(pred_dist, target_ltrb).view(-1, 4).mean(-1, keepdim=True)
                loss[2] = (dfl * weight).sum() / target_scores_sum

        loss[0] *= self.box_gain
        loss[1] *= self.cls_gain
        loss[2] *= self.dfl_gain
        return loss.sum() * batch_size, loss.detach()
