"""跟踪框变换与锚框生成（xyxy / xywh / 中心格式）。"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import torch


def xyxy2xywh(boxes: torch.Tensor) -> torch.Tensor:
    """``(x1, y1, x2, y2) -> (cx, cy, w, h)``，最后一维为 4。"""
    x1, y1, x2, y2 = boxes.unbind(-1)
    return torch.stack(((x1 + x2) * 0.5, (y1 + y2) * 0.5, x2 - x1, y2 - y1), dim=-1)


def xywh2xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """``(cx, cy, w, h) -> (x1, y1, x2, y2)``，最后一维为 4。"""
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack((cx - w * 0.5, cy - h * 0.5, cx + w * 0.5, cy + h * 0.5), dim=-1)


def corner2center(x1, y1, x2, y2):
    """角点 → 中心格式（numpy / tensor 通用）。"""
    return (x1 + x2) * 0.5, (y1 + y2) * 0.5, x2 - x1, y2 - y1


def center2corner(cx, cy, w, h):
    """中心 → 角点格式。"""
    return cx - w * 0.5, cy - h * 0.5, cx + w * 0.5, cy + h * 0.5


def encode_bbox(gt_xyxy: torch.Tensor, anchors_xywh: torch.Tensor) -> torch.Tensor:
    """把真值框编码为相对锚框的回归目标 ``(dx, dy, dw, dh)``。

    Args:
        gt_xyxy: ``(..., 4)`` 真值角点框
        anchors_xywh: ``(..., 4)`` 锚框中心格式
    """
    gt = xyxy2xywh(gt_xyxy)
    dx = (gt[..., 0] - anchors_xywh[..., 0]) / anchors_xywh[..., 2]
    dy = (gt[..., 1] - anchors_xywh[..., 1]) / anchors_xywh[..., 3]
    dw = torch.log(gt[..., 2] / anchors_xywh[..., 2])
    dh = torch.log(gt[..., 3] / anchors_xywh[..., 3])
    return torch.stack((dx, dy, dw, dh), dim=-1)


def decode_bbox(deltas: torch.Tensor, anchors_xywh: torch.Tensor) -> torch.Tensor:
    """把回归增量解码为 ``(x1, y1, x2, y2)``。"""
    cx = deltas[..., 0] * anchors_xywh[..., 2] + anchors_xywh[..., 0]
    cy = deltas[..., 1] * anchors_xywh[..., 3] + anchors_xywh[..., 1]
    w = torch.exp(deltas[..., 2]) * anchors_xywh[..., 2]
    h = torch.exp(deltas[..., 3]) * anchors_xywh[..., 3]
    return xywh2xyxy(torch.stack((cx, cy, w, h), dim=-1))


class Anchors:
    """按 stride / ratios / scales 生成 RPN 锚框。

    单点锚框存为相对原点的 ``(x1, y1, x2, y2)``；调用 ``generate_all`` 后得到
    整幅响应图上的角点与中心两套锚框。
    """

    def __init__(
        self,
        stride: int = 8,
        ratios: Sequence[float] = (0.33, 0.5, 1.0, 2.0, 3.0),
        scales: Sequence[float] = (8.0,),
    ) -> None:
        self.stride = int(stride)
        self.ratios = list(ratios)
        self.scales = list(scales)
        self.anchor_num = len(self.ratios) * len(self.scales)
        self.anchors = self._generate_base()
        self.image_center: int | None = None
        self.size: int | None = None
        self.all_anchors: tuple[np.ndarray, np.ndarray] | None = None

    def _generate_base(self) -> np.ndarray:
        anchors = np.zeros((self.anchor_num, 4), dtype=np.float32)
        size = self.stride * self.stride
        count = 0
        for r in self.ratios:
            ws = int(math.sqrt(size / r))
            hs = int(ws * r)
            for s in self.scales:
                w, h = ws * s, hs * s
                anchors[count] = [-w * 0.5, -h * 0.5, w * 0.5, h * 0.5]
                count += 1
        return anchors

    def generate_all(self, im_c: int, size: int) -> bool:
        """在响应图尺寸 ``size x size``、图像中心 ``im_c`` 处铺满锚框。

        Returns:
            True 表示锚框已重新生成；False 表示缓存命中、无需更新。
        """
        if self.image_center == im_c and self.size == size:
            return False
        self.image_center, self.size = im_c, size

        a0x = im_c - size // 2 * self.stride
        ori = np.array([a0x] * 4, dtype=np.float32)
        zero = self.anchors + ori
        x1, y1, x2, y2 = (zero[:, i].reshape(self.anchor_num, 1, 1) for i in range(4))
        cx, cy, w, h = corner2center(x1, y1, x2, y2)

        disp_x = np.arange(0, size, dtype=np.float32).reshape(1, 1, -1) * self.stride
        disp_y = np.arange(0, size, dtype=np.float32).reshape(1, -1, 1) * self.stride
        cx = cx + disp_x
        cy = cy + disp_y
        zero_map = np.zeros((self.anchor_num, size, size), dtype=np.float32)
        cx, cy, w, h = cx + zero_map, cy + zero_map, w + zero_map, h + zero_map
        x1, y1, x2, y2 = center2corner(cx, cy, w, h)

        self.all_anchors = (
            np.stack([x1, y1, x2, y2]).astype(np.float32),
            np.stack([cx, cy, w, h]).astype(np.float32),
        )
        return True
