"""二分类评估指标：AUC、ACC、SEN（敏感性）、SPE（特异性）、BAC（平衡准确率）。"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score

METRIC_KEYS = ("auc", "acc", "sen", "spe", "bac")


def binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    """y_true: [S] 0/1 标签；y_prob: [S] 正类（y=1）概率。"""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sen = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spe = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    acc = (tp + tn) / max(1, len(y_true))
    bac = (sen + spe) / 2.0
    # 验证折只有单一类别时 AUC 无定义
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) == 2 else float("nan")
    return {"auc": float(auc), "acc": float(acc), "sen": float(sen), "spe": float(spe), "bac": float(bac)}


def summarize_folds(fold_metrics: Sequence[Mapping[str, float]]) -> dict[str, dict[str, float]]:
    """对各折指标计算 mean / std（忽略 NaN）。"""
    summary: dict[str, dict[str, float]] = {}
    for key in METRIC_KEYS:
        values = np.array([m[key] for m in fold_metrics], dtype=np.float64)
        summary[key] = {"mean": float(np.nanmean(values)), "std": float(np.nanstd(values))}
    return summary
