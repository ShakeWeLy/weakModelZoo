"""二值分割指标：Dice、IoU、Precision、Recall、FP/FN ratio、面积占比等。"""

from __future__ import annotations

from typing import Any

import numpy as np


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return float(numerator / denominator)


def compute_binary_metrics(
    pred_mask: np.ndarray,
    target_mask: np.ndarray,
    total_pixels: int | None = None,
    eps: float = 1e-6,
) -> dict[str, Any]:
    """计算单器官二值掩码的混淆矩阵派生指标。

    FP ratio = FP / (FP + TN)
    FN ratio = FN / (TP + FN)
    """
    pred = pred_mask.astype(bool)
    target = target_mask.astype(bool)
    if total_pixels is None:
        total_pixels = pred.size

    tp = int(np.logical_and(pred, target).sum())
    fp = int(np.logical_and(pred, np.logical_not(target)).sum())
    fn = int(np.logical_and(np.logical_not(pred), target).sum())
    tn = int(np.logical_and(np.logical_not(pred), np.logical_not(target)).sum())

    gt_area = int(target.sum())
    pred_area = int(pred.sum())
    union = tp + fp + fn

    if gt_area == 0 and pred_area == 0:
        dice = iou = precision = recall = 1.0
        fp_ratio = fn_ratio = 0.0
    elif union == 0:
        dice = iou = precision = recall = 1.0
        fp_ratio = _safe_ratio(fp, fp + tn) or 0.0
        fn_ratio = _safe_ratio(fn, tp + fn) or 0.0
    else:
        dice = float((2.0 * tp + eps) / (2.0 * tp + fp + fn + eps))
        iou = float((tp + eps) / (union + eps))
        precision = float((tp + eps) / (tp + fp + eps)) if tp + fp > 0 else 0.0
        recall = float((tp + eps) / (tp + fn + eps)) if tp + fn > 0 else 0.0
        fp_ratio = _safe_ratio(fp, fp + tn)
        fn_ratio = _safe_ratio(fn, tp + fn)

    pred_gt_ratio = _safe_ratio(pred_area, gt_area)
    gt_percent = float(gt_area / total_pixels * 100.0)
    pred_percent = float(pred_area / total_pixels * 100.0)

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "fp_ratio": fp_ratio,
        "fn_ratio": fn_ratio,
        "gt_area": gt_area,
        "pred_area": pred_area,
        "pred_gt_ratio": pred_gt_ratio,
        "gt_percent": gt_percent,
        "pred_percent": pred_percent,
    }


def mean_ignore_none(values: list[float | None]) -> float | None:
    valid = [v for v in values if v is not None]
    if not valid:
        return None
    return float(np.mean(valid))


def aggregate_metric_lists(values: list[float | None]) -> dict[str, float | None]:
    valid = [v for v in values if v is not None]
    if not valid:
        return {"mean": None, "std": None, "mean_percent": None}
    mean = float(np.mean(valid))
    return {
        "mean": mean,
        "std": float(np.std(valid)),
        "mean_percent": mean * 100.0,
    }
