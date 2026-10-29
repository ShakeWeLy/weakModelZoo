"""按病例（3D 体数据）计算 Dice / HD95，口径与官方 TransUNet / Swin-Unet 的 ``test.py`` 一致。

官方流程（``utils.py`` 中的 ``test_single_volume``）：
1. 每个病例的切片预测按 z 轴拼成 3D 体；
2. 对每个器官计算整个 3D 体上的二值 Dice（medpy ``metric.binary.dc``）与 HD95；
3. 对所有病例取平均（``metric_list / len(db_test)``），再对器官取平均得到 mean_dice。

空掩码规则（``calculate_metric_percase``）：
- GT 与预测都有该器官：正常计算；
- 仅预测有该器官（GT 为空）：Dice 记为 1，HD95 记为 0；
- GT 有该器官而预测为空，或两者都为空：Dice 记为 0，HD95 记为 0。
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Mapping

import numpy as np
from medpy import metric

_SLICE_RE = re.compile(r"^(?P<case>.+?)_slice(?P<z>\d+)$")


def split_case_slice(name: str) -> tuple[str, int] | None:
    """``case0008_slice012`` -> ``("case0008", 12)``；无法解析时返回 None。"""
    match = _SLICE_RE.match(name)
    if match is None:
        return None
    return match["case"], int(match["z"])


def case_metric_percase(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    """与官方 ``calculate_metric_percase`` 相同的空掩码规则，返回 (dice, hd95)。"""
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    if pred.any() and gt.any():
        return float(metric.binary.dc(pred, gt)), float(metric.binary.hd95(pred, gt))
    if pred.any() and not gt.any():
        return 1.0, 0.0
    return 0.0, 0.0


class VolumeAccumulator:
    """收集同一病例的所有切片，最后按病例拼成 3D 体计算官方口径指标。"""

    def __init__(self) -> None:
        self.volumes: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]] = defaultdict(dict)
        self.skipped: list[str] = []

    def add(self, name: str, pred: np.ndarray, label: np.ndarray) -> None:
        parsed = split_case_slice(name)
        if parsed is None:
            self.skipped.append(name)
            return
        case, z = parsed
        # 只保存整数标签，降低内存占用
        self.volumes[case][z] = (pred.astype(np.uint8), label.astype(np.uint8))

    def summarize(self, organ_map: Mapping[int, str]) -> dict[str, Any]:
        """返回 ``{"mean_dice", "mean_hd95", "per_organ", "per_case"}``。"""
        per_case: list[dict[str, Any]] = []
        organ_dice: dict[str, list[float]] = {organ: [] for organ in organ_map.values()}
        organ_hd95: dict[str, list[float]] = {organ: [] for organ in organ_map.values()}

        for case in sorted(self.volumes):
            zs = sorted(self.volumes[case])
            pred_vol = np.stack([self.volumes[case][z][0] for z in zs])
            gt_vol = np.stack([self.volumes[case][z][1] for z in zs])
            for class_id, organ in organ_map.items():
                dice, hd95 = case_metric_percase(pred_vol == class_id, gt_vol == class_id)
                organ_dice[organ].append(dice)
                organ_hd95[organ].append(hd95)
                per_case.append({
                    "case": case, "num_slices": len(zs), "class_id": class_id,
                    "organ": organ, "dice": dice, "hd95": hd95,
                })

        per_organ = {
            organ: {
                "dice": float(np.mean(values)) if values else None,
                "hd95": float(np.mean(organ_hd95[organ])) if values else None,
                "num_cases": len(values),
            }
            for organ, values in organ_dice.items()
        }
        organ_means = [v["dice"] for v in per_organ.values() if v["dice"] is not None]
        hd95_means = [v["hd95"] for v in per_organ.values() if v["hd95"] is not None]
        return {
            "mean_dice": float(np.mean(organ_means)) if organ_means else None,
            "mean_hd95": float(np.mean(hd95_means)) if hd95_means else None,
            "num_cases": len(self.volumes),
            "per_organ": per_organ,
            "per_case": per_case,
            "skipped_slices": self.skipped,
        }
