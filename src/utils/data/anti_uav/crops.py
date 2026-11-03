"""孪生跟踪裁剪：模板 / 搜索图（对齐 pysot context crop）。"""

from __future__ import annotations

import math

import cv2
import numpy as np


def box_context_size(w: float, h: float, context_amount: float = 0.5) -> float:
    """带上下文的等效正方形边长。"""
    pad = context_amount * (w + h)
    return float(math.sqrt((w + pad) * (h + pad)))


def crop_square(
    image: np.ndarray,
    cx: float,
    cy: float,
    model_sz: int,
    original_sz: float,
) -> tuple[np.ndarray, float]:
    """以 ``(cx, cy)`` 为中心裁边长 ``original_sz`` 的正方形并 resize 到 ``model_sz``。

    越界处用通道均值填充。返回 ``(crop, scale)``，``scale = model_sz / original_sz``。
    """
    if original_sz <= 1e-6:
        original_sz = float(model_sz)
    scale = model_sz / original_sz
    h, w = image.shape[:2]
    half = original_sz / 2.0

    x1 = int(round(cx - half))
    y1 = int(round(cy - half))
    x2 = int(round(cx + half))
    y2 = int(round(cy + half))
    # round 可能导致宽高差 1，以实际裁剪窗为准
    out_h = max(1, y2 - y1)
    out_w = max(1, x2 - x1)

    left = max(0, -x1)
    top = max(0, -y1)
    right = max(0, x2 - w)
    bottom = max(0, y2 - h)

    x1c, y1c = x1 + left, y1 + top
    x2c, y2c = x2 - right, y2 - bottom
    patch = image[y1c:y2c, x1c:x2c]
    ph, pw = patch.shape[:2]

    if image.ndim == 2:
        mean = float(image.mean()) if image.size else 0.0
        canvas = np.full((out_h, out_w), mean, dtype=image.dtype)
    else:
        mean = image.mean(axis=(0, 1)) if image.size else np.zeros(image.shape[2], dtype=np.float64)
        canvas = np.empty((out_h, out_w, image.shape[2]), dtype=image.dtype)
        canvas[:] = mean
    # 防止 pad 与 patch 尺寸因取整不一致
    canvas[top : top + ph, left : left + pw] = patch[: min(ph, out_h - top), : min(pw, out_w - left)]

    out = cv2.resize(canvas, (model_sz, model_sz), interpolation=cv2.INTER_LINEAR)
    return out, scale


def crop_template(
    image: np.ndarray,
    bbox_xywh: np.ndarray,
    exemplar_size: int = 127,
    context_amount: float = 0.5,
) -> tuple[np.ndarray, float]:
    """按目标框裁模板图。``bbox_xywh = (x, y, w, h)``。"""
    x, y, w, h = map(float, bbox_xywh)
    cx, cy = x + w * 0.5, y + h * 0.5
    s_z = box_context_size(w, h, context_amount)
    return crop_square(image, cx, cy, exemplar_size, s_z)


def crop_search(
    image: np.ndarray,
    bbox_xywh: np.ndarray,
    exemplar_size: int = 127,
    search_size: int = 255,
    context_amount: float = 0.5,
    center_shift: tuple[float, float] = (0.0, 0.0),
) -> tuple[np.ndarray, np.ndarray, float]:
    """裁搜索图，并返回目标在搜索图上的 ``xywh``（像素）。

    ``center_shift`` 为相对搜索区域边长的比例偏移（spatial-aware sampling）。
    """
    x, y, w, h = map(float, bbox_xywh)
    cx, cy = x + w * 0.5, y + h * 0.5
    s_z = box_context_size(w, h, context_amount)
    s_x = s_z * (search_size / exemplar_size)
    cx = cx + center_shift[0] * s_x
    cy = cy + center_shift[1] * s_x
    crop, scale = crop_square(image, cx, cy, search_size, s_x)

    # 原图像素 → 搜索图坐标
    gt_cx = (x + w * 0.5 - (cx - s_x * 0.5)) * scale
    gt_cy = (y + h * 0.5 - (cy - s_x * 0.5)) * scale
    gt_w, gt_h = w * scale, h * scale
    gt_xywh = np.array([gt_cx - gt_w * 0.5, gt_cy - gt_h * 0.5, gt_w, gt_h], dtype=np.float32)
    return crop, gt_xywh, scale
