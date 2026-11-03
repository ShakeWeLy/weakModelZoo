"""根据搜索图上的真值框生成 RPN 分类 / 回归标签（对齐 pysot AnchorTarget）。"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from src.models.track.SiamSPN.box import Anchors, encode_bbox


def _iou_xyxy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``a (N,4)`` 与单个 ``b (4,)`` 的 IoU。"""
    x1 = np.maximum(a[:, 0], b[0])
    y1 = np.maximum(a[:, 1], b[1])
    x2 = np.minimum(a[:, 2], b[2])
    y2 = np.minimum(a[:, 3], b[3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area_a = np.maximum(0.0, a[:, 2] - a[:, 0]) * np.maximum(0.0, a[:, 3] - a[:, 1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter + 1e-6
    return inter / union


class AnchorTarget:
    """生成 ``label_cls / label_loc / label_loc_weight``。

    - IoU >= ``pos_thr`` → 正样本 (1)
    - IoU < ``neg_thr`` → 负样本 (0)
    - 中间 → 忽略 (-1)
    """

    def __init__(
        self,
        stride: int = 8,
        ratios: Sequence[float] = (0.33, 0.5, 1.0, 2.0, 3.0),
        scales: Sequence[float] = (8.0,),
        score_size: int = 25,
        search_size: int = 255,
        pos_thr: float = 0.6,
        neg_thr: float = 0.3,
        pos_num: int = 16,
        neg_num: int = 16,
    ) -> None:
        self.anchors = Anchors(stride=stride, ratios=ratios, scales=scales)
        self.score_size = score_size
        self.search_size = search_size
        self.pos_thr = pos_thr
        self.neg_thr = neg_thr
        self.pos_num = pos_num
        self.neg_num = neg_num
        self.anchor_num = self.anchors.anchor_num
        self.anchors.generate_all(im_c=search_size // 2, size=score_size)

    def __call__(self, bbox_xywh: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            bbox_xywh: 搜索图上的目标 ``(x, y, w, h)``。

        Returns:
            label_cls: ``(A, H, W)`` long
            label_loc: ``(4, A, H, W)`` float
            label_loc_weight: ``(A, H, W)`` float
        """
        assert self.anchors.all_anchors is not None
        anc_xyxy, anc_xywh = self.anchors.all_anchors  # (4,A,H,W)
        a, h, w = self.anchor_num, self.score_size, self.score_size

        x, y, bw, bh = map(float, bbox_xywh)
        gt_xyxy = np.array([x, y, x + bw, y + bh], dtype=np.float32)

        flat_anc = anc_xyxy.reshape(4, -1).T  # (A*H*W, 4)
        ious = _iou_xyxy(flat_anc, gt_xyxy).reshape(a, h, w)

        label_cls = np.full((a, h, w), -1, dtype=np.int64)
        label_cls[ious >= self.pos_thr] = 1
        label_cls[ious < self.neg_thr] = 0

        # 采样，避免负样本过多
        pos_idx = np.argwhere(label_cls == 1)
        neg_idx = np.argwhere(label_cls == 0)
        if len(pos_idx) > self.pos_num:
            keep = np.random.choice(len(pos_idx), self.pos_num, replace=False)
            drop = np.ones(len(pos_idx), dtype=bool)
            drop[keep] = False
            for i, j, k in pos_idx[drop]:
                label_cls[i, j, k] = -1
            pos_idx = pos_idx[keep]
        if len(neg_idx) > self.neg_num:
            keep = np.random.choice(len(neg_idx), self.neg_num, replace=False)
            drop = np.ones(len(neg_idx), dtype=bool)
            drop[keep] = False
            for i, j, k in neg_idx[drop]:
                label_cls[i, j, k] = -1

        label_loc_weight = np.zeros((a, h, w), dtype=np.float32)
        label_loc_weight[label_cls == 1] = 1.0

        gt_t = torch.tensor(gt_xyxy, dtype=torch.float32)
        anc_t = torch.from_numpy(anc_xywh).permute(1, 2, 3, 0).contiguous()  # A,H,W,4
        # encode 需要广播 gt 到每个锚框
        gt_b = gt_t.view(1, 1, 1, 4).expand_as(anc_t)
        deltas = encode_bbox(gt_b, anc_t)  # A,H,W,4
        label_loc = deltas.permute(3, 0, 1, 2).contiguous()  # 4,A,H,W

        return (
            torch.from_numpy(label_cls),
            label_loc,
            torch.from_numpy(label_loc_weight),
        )
