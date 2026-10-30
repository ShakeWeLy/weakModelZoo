"""检测评估指标：逐图 TP 匹配（多 IoU 阈值）+ AP / mAP / Precision / Recall（101 点插值，与 YOLOv5 一致）。"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torchvision.ops import box_iou

_trapz = getattr(np, "trapezoid", None) or np.trapz


def process_batch(
    detections: torch.Tensor,
    labels_xyxy: torch.Tensor,
    labels_cls: torch.Tensor,
    iouv: torch.Tensor,
) -> torch.Tensor:
    """单张图的 TP 匹配。

    Args:
        detections: (N, 6) -> x1, y1, x2, y2, conf, cls。
        labels_xyxy: (M, 4) 真值框。
        labels_cls: (M,) 真值类别。
        iouv: (T,) IoU 阈值。
    Returns:
        (N, T) bool：每个预测在每个阈值下是否为 TP。
    """
    correct = torch.zeros(detections.shape[0], iouv.shape[0], dtype=torch.bool, device=detections.device)
    if detections.shape[0] == 0 or labels_xyxy.shape[0] == 0:
        return correct

    iou = box_iou(labels_xyxy, detections[:, :4])                 # (M, N)
    correct_class = labels_cls[:, None] == detections[:, 5][None]  # (M, N)
    for i, thr in enumerate(iouv):
        hits = torch.where((iou >= thr) & correct_class)
        if hits[0].shape[0] == 0:
            continue
        matches = torch.cat((torch.stack(hits, 1).float(), iou[hits[0], hits[1]][:, None]), 1).cpu().numpy()
        if hits[0].shape[0] > 1:
            # 贪心一对一匹配：按 IoU 从高到低，每个 GT 与每个预测只能用一次
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        det_idx = torch.from_numpy(matches[:, 1].astype(np.int64)).to(detections.device)
        correct[det_idx, i] = True
    return correct


def compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """101 点插值 AP（COCO 风格）。"""
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    return float(_trapz(np.interp(x, mrec, mpre), x))


def ap_per_class(
    tp: np.ndarray,
    conf: np.ndarray,
    pred_cls: np.ndarray,
    target_cls: np.ndarray,
    eps: float = 1e-16,
) -> dict[str, Any]:
    """Args:
        tp: (N, T) 0/1，每个预测在各 IoU 阈值下是否为 TP。
        conf / pred_cls: (N,)。
        target_cls: (M,) 所有真值的类别。
    Returns:
        ap: (C, T)；precision / recall: (C,)（取 IoU=0.5 下 F1 最大处）；classes: (C,)。
    """
    order = np.argsort(-conf, kind="stable")
    tp, pred_cls = tp[order].astype(np.float64), pred_cls[order]

    classes, n_gt = np.unique(target_cls, return_counts=True)
    n_c, n_t = classes.shape[0], tp.shape[1]
    ap = np.zeros((n_c, n_t))
    precision = np.zeros(n_c)
    recall_out = np.zeros(n_c)

    for ci, c in enumerate(classes):
        sel = pred_cls == c
        if sel.sum() == 0 or n_gt[ci] == 0:
            continue
        fpc = (1 - tp[sel]).cumsum(0)
        tpc = tp[sel].cumsum(0)
        recall = tpc / (n_gt[ci] + eps)
        prec = tpc / (tpc + fpc + eps)
        for j in range(n_t):
            ap[ci, j] = compute_ap(recall[:, j], prec[:, j])
        f1 = 2 * prec[:, 0] * recall[:, 0] / (prec[:, 0] + recall[:, 0] + eps)
        k = int(f1.argmax())
        precision[ci], recall_out[ci] = prec[k, 0], recall[k, 0]

    return {"ap": ap, "precision": precision, "recall": recall_out, "classes": classes}
