"""训练期（epoch 级累积 Dice）与评估期（切片级多指标 + HD95）的指标计算。"""

from __future__ import annotations

import sys
from typing import Any, Mapping

import numpy as np
import torch
from scipy.ndimage import binary_erosion, distance_transform_edt

from .config import ROOT
from .labels import LabelSpace

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.segmentation_metrics import aggregate_metric_lists, compute_binary_metrics  # noqa: E402

SLICE_METRIC_KEYS = (
    "dice", "iou", "precision", "recall", "fp_ratio", "fn_ratio",
    "gt_area", "pred_area", "pred_gt_ratio", "gt_percent", "pred_percent", "hd95",
)
PERCENT_KEYS = ("dice", "iou", "precision", "recall", "fp_ratio", "fn_ratio")
AREA_KEYS = ("gt_area", "pred_area", "pred_gt_ratio", "gt_percent", "pred_percent")


class DiceAccumulator:
    """在整个 epoch 上累积各类别 TP 与 |P|+|G|，得到全局 Dice（比逐 batch 平均更稳定）。"""

    def __init__(self, space: LabelSpace, eps: float = 1e-6):
        self.space = space
        self.class_ids = space.metric_class_ids
        self.eps = eps
        self.intersection = torch.zeros(len(self.class_ids), dtype=torch.float64)
        self.denominator = torch.zeros(len(self.class_ids), dtype=torch.float64)

    @torch.no_grad()
    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        preds = logits.argmax(dim=1)
        for i, class_id in enumerate(self.class_ids):
            pred_mask = preds == class_id
            target_mask = targets == class_id
            self.intersection[i] += (pred_mask & target_mask).sum().item()
            self.denominator[i] += pred_mask.sum().item() + target_mask.sum().item()

    def per_class(self) -> dict[int, float | None]:
        return {
            class_id: None if self.denominator[i] == 0
            else float((2 * self.intersection[i] + self.eps) / (self.denominator[i] + self.eps))
            for i, class_id in enumerate(self.class_ids)
        }

    def mean(self) -> float:
        values = [v for v in self.per_class().values() if v is not None]
        return float(np.mean(values)) if values else 0.0


def _surface(mask: np.ndarray) -> np.ndarray:
    return mask & ~binary_erosion(mask)


def compute_hd95(pred: np.ndarray, target: np.ndarray) -> float | None:
    """切片级 95% Hausdorff 距离（像素）；仅一方为空时返回 None。"""
    pred, target = pred.astype(bool), target.astype(bool)
    if not pred.any() and not target.any():
        return 0.0
    if not pred.any() or not target.any():
        return None
    pred_surface, target_surface = _surface(pred), _surface(target)
    d_pred = distance_transform_edt(~target)[pred_surface]
    d_target = distance_transform_edt(~pred)[target_surface]
    return float(max(np.percentile(d_pred, 95), np.percentile(d_target, 95)))


def slice_organ_metrics(pred: np.ndarray, label: np.ndarray, organ_map: Mapping[int, str]) -> dict[int, dict[str, Any]]:
    """GT 与预测均不含该器官的切片不计入。"""
    results: dict[int, dict[str, Any]] = {}
    for class_id in organ_map:
        pred_mask, target_mask = pred == class_id, label == class_id
        if not pred_mask.any() and not target_mask.any():
            continue
        metrics = compute_binary_metrics(pred_mask, target_mask, pred.size)
        metrics["hd95"] = compute_hd95(pred_mask, target_mask)
        results[class_id] = metrics
    return results


def summarize_organs(per_class_values: Mapping[int, Mapping[str, list]], space: LabelSpace) -> dict[str, Any]:
    """器官级：切片均值；整体：各器官均值再取平均（宏平均）。"""
    per_organ: dict[str, dict[str, Any]] = {}
    pooled: dict[str, list[float]] = {key: [] for key in (*PERCENT_KEYS, "hd95")}
    for class_id, organ in space.organ_map().items():
        values = per_class_values.get(class_id, {})
        entry: dict[str, Any] = {
            "class_id": class_id,
            "name_cn": space.class_name_cn(class_id),
            "num_slices": len(values.get("dice", [])),
        }
        for key in SLICE_METRIC_KEYS:
            agg = aggregate_metric_lists(list(values.get(key, [])))
            entry[key] = agg["mean"]
            if key in PERCENT_KEYS:
                entry[f"{key}_percent"] = agg["mean_percent"]
            if key in AREA_KEYS:
                entry[f"{key}_mean"] = agg["mean"]
            if key in pooled and agg["mean"] is not None:
                pooled[key].append(agg["mean"])
        per_organ[organ] = entry

    overall: dict[str, Any] = {}
    for key, values in pooled.items():
        agg = aggregate_metric_lists(values)
        overall[f"mean_{key}"] = agg["mean"]
        if key in PERCENT_KEYS:
            overall[f"mean_{key}_percent"] = agg["mean_percent"]
    return {**overall, "per_organ": per_organ}
