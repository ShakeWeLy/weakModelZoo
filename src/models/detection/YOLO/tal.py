"""Task-Aligned Assigner 与框距离编解码（对齐 ultralytics YOLOv8 思路的精简实现）。"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.detection.YOLO.box import bbox_iou, xywh2xyxy, xyxy2xywh


def make_anchors(
    feats: list[torch.Tensor],
    strides: torch.Tensor,
    grid_cell_offset: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """由多尺度特征图生成格子中心点（格子单位）与对应步长。"""
    anchor_points, stride_tensor = [], []
    dtype = feats[0].dtype
    device = feats[0].device
    for i, feat in enumerate(feats):
        h, w = feat.shape[2], feat.shape[3]
        sx = torch.arange(w, device=device, dtype=dtype) + grid_cell_offset
        sy = torch.arange(h, device=device, dtype=dtype) + grid_cell_offset
        sy, sx = torch.meshgrid(sy, sx, indexing="ij")
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), float(strides[i]), dtype=dtype, device=device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)


def dist2bbox(distance: torch.Tensor, anchor_points: torch.Tensor, xywh: bool = True, dim: int = -1) -> torch.Tensor:
    """ltrb 距离 → xywh 或 xyxy。"""
    lt, rb = distance.chunk(2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat((c_xy, wh), dim)
    return torch.cat((x1y1, x2y2), dim)


def bbox2dist(anchor_points: torch.Tensor, bbox: torch.Tensor, reg_max: float) -> torch.Tensor:
    """xyxy 框 → 相对格子中心的 ltrb，并截断到 [0, reg_max)。"""
    x1y1, x2y2 = bbox.chunk(2, -1)
    return torch.cat((anchor_points - x1y1, x2y2 - anchor_points), -1).clamp_(0, reg_max - 0.01)


class TaskAlignedAssigner(nn.Module):
    """按 t = s^α · u^β 为每个真值选 top-k 正样本格子。"""

    def __init__(
        self,
        topk: int = 10,
        num_classes: int = 80,
        alpha: float = 0.5,
        beta: float = 6.0,
        eps: float = 1e-9,
        stride_val: float = 16.0,
    ) -> None:
        super().__init__()
        self.topk = topk
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self.stride_val = stride_val

    @torch.no_grad()
    def forward(
        self,
        pd_scores: torch.Tensor,
        pd_bboxes: torch.Tensor,
        anc_points: torch.Tensor,
        gt_labels: torch.Tensor,
        gt_bboxes: torch.Tensor,
        mask_gt: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            pd_scores: (b, n_anchors, nc) 已 sigmoid
            pd_bboxes: (b, n_anchors, 4) 像素 xyxy
            anc_points: (n_anchors, 2) 像素坐标
            gt_labels: (b, max_gt, 1)
            gt_bboxes: (b, max_gt, 4) 像素 xyxy
            mask_gt: (b, max_gt, 1) 有效真值掩码
        """
        bs = pd_scores.shape[0]
        n_max = gt_bboxes.shape[1]
        if n_max == 0:
            return (
                torch.full_like(pd_scores[..., 0], self.num_classes),
                torch.zeros_like(pd_bboxes),
                torch.zeros_like(pd_scores),
                torch.zeros_like(pd_scores[..., 0], dtype=torch.bool),
                torch.zeros_like(pd_scores[..., 0]),
            )

        self.bs, self.n_max_boxes = bs, n_max
        mask_in = self._select_in_gts(anc_points, gt_bboxes, mask_gt)
        align, overlaps = self._box_metrics(pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_in * mask_gt)
        mask_topk = self._select_topk(align, mask_gt.expand(-1, -1, self.topk).bool())
        mask_pos = mask_topk * mask_in * mask_gt.bool()

        target_gt_idx, fg_mask, mask_pos = self._resolve_conflicts(mask_pos, overlaps, n_max)
        target_labels, target_bboxes, target_scores = self._get_targets(
            gt_labels, gt_bboxes, target_gt_idx, fg_mask
        )

        # nan*0 在 IEEE 里仍是 nan，必须先洗掉
        align = torch.nan_to_num(align, nan=0.0) * mask_pos
        overlaps = torch.nan_to_num(overlaps, nan=0.0) * mask_pos
        pos_align = align.amax(dim=-1, keepdim=True)
        pos_iou = overlaps.amax(dim=-1, keepdim=True)
        align = align * pos_iou / (pos_align + self.eps)
        target_scores = target_scores * torch.nan_to_num(align.amax(1).unsqueeze(-1), nan=0.0)
        return target_labels, target_bboxes, target_scores, fg_mask.bool(), target_gt_idx

    def _select_in_gts(self, xy_centers: torch.Tensor, gt_bboxes: torch.Tensor, mask_gt: torch.Tensor) -> torch.Tensor:
        gt_xywh = xyxy2xywh(gt_bboxes)
        tiny = gt_xywh[..., 2:] < self.stride_val
        gt_xywh = gt_xywh.clone()
        gt_xywh[..., 2:] = torch.where(tiny & mask_gt.bool(), torch.full_like(gt_xywh[..., 2:], self.stride_val), gt_xywh[..., 2:])
        gt = xywh2xyxy(gt_xywh)
        lt, rb = gt.unsqueeze(2).chunk(2, 3)
        mask = (xy_centers[:, 0] - lt[..., 0] > self.eps) & (xy_centers[:, 1] - lt[..., 1] > self.eps)
        mask &= (rb[..., 0] - xy_centers[:, 0] > self.eps) & (rb[..., 1] - xy_centers[:, 1] > self.eps)
        return mask

    def _box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt):
        na = pd_bboxes.shape[-2]
        mask_gt = mask_gt.bool()
        shape = (self.bs, self.n_max_boxes, na)
        idx = mask_gt.nonzero(as_tuple=True)
        scores = pd_scores[idx[0], idx[2], gt_labels[idx[0], idx[1], 0].long()]
        ious = bbox_iou(gt_bboxes[idx[0], idx[1]], pd_bboxes[idx[0], idx[2]], xywh=False, CIoU=True).squeeze(-1).clamp_(0)
        # AMP 下 scores 可能是 half，IoU 常为 float；写入前统一到目标张量 dtype
        overlaps = pd_bboxes.new_zeros(shape)
        align = pd_scores.new_zeros(shape)
        ious = ious.to(dtype=overlaps.dtype)
        align_v = (scores.float().pow(self.alpha) * ious.float().pow(self.beta)).to(dtype=align.dtype)
        overlaps[idx] = ious
        align[idx] = align_v
        return align, overlaps

    def _select_topk(self, metrics: torch.Tensor, topk_mask: torch.Tensor) -> torch.Tensor:
        topk_metrics, topk_idxs = torch.topk(metrics, self.topk, dim=-1, largest=True)
        if topk_mask is None:
            topk_mask = (topk_metrics.max(-1, keepdim=True)[0] > self.eps).expand_as(topk_idxs)
        topk_idxs.masked_fill_(~topk_mask, 0)
        count = torch.zeros(metrics.shape, dtype=torch.int8, device=metrics.device)
        ones = torch.ones_like(topk_idxs, dtype=torch.int8)
        count.scatter_add_(-1, topk_idxs, ones)
        count.masked_fill_(count > 1, 0)
        return count

    def _resolve_conflicts(self, mask_pos, overlaps, n_max):
        fg = mask_pos.sum(1)
        multi = (fg.unsqueeze(1) > 1).expand(-1, n_max, -1)
        max_idx = overlaps.max(1).indices
        is_max = torch.zeros_like(mask_pos)
        is_max.scatter_(1, max_idx.unsqueeze(1), 1)
        mask_pos = torch.where(multi, is_max, mask_pos)
        fg_mask, target_gt_idx = mask_pos.max(1)
        return target_gt_idx, fg_mask, mask_pos

    def _get_targets(self, gt_labels, gt_bboxes, target_gt_idx, fg_mask):
        batch_ind = torch.arange(self.bs, device=gt_labels.device)[:, None]
        flat_idx = target_gt_idx + batch_ind * self.n_max_boxes
        target_labels = gt_labels.long().flatten()[flat_idx].clamp_(0)
        target_bboxes = gt_bboxes.view(-1, 4)[flat_idx]
        target_scores = torch.zeros(
            (*target_labels.shape, self.num_classes), dtype=torch.float32, device=gt_labels.device
        )
        target_scores.scatter_(2, target_labels.unsqueeze(-1), 1.0)
        target_scores *= (fg_mask > 0).unsqueeze(-1)
        return target_labels, target_bboxes, target_scores
