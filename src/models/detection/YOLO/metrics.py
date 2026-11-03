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
        另含官方风格曲线数据：
        px (1000,)；py (C, 1000) PR 曲线插值精度；
        p / r / f1 (C, 1000) 相对置信度阈值的曲线；conf_curve (1000,)。
    """
    order = np.argsort(-conf, kind="stable")
    tp = tp[order].astype(np.float64)
    conf = conf[order].astype(np.float64)
    pred_cls = pred_cls[order]

    classes, n_gt = np.unique(target_cls, return_counts=True)
    n_c, n_t = classes.shape[0], tp.shape[1]
    ap = np.zeros((n_c, n_t))
    precision = np.zeros(n_c)
    recall_out = np.zeros(n_c)

    # 与 Ultralytics 一致：1000 点插值，便于画 PR / F1 / P / R 曲线
    px = np.linspace(0, 1, 1000)
    conf_curve = np.linspace(0, 1, 1000)
    py = np.zeros((n_c, 1000))
    p_curve = np.zeros((n_c, 1000))
    r_curve = np.zeros((n_c, 1000))

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

        # IoU=0.5 曲线
        r50, p50 = recall[:, 0], prec[:, 0]
        conf_c = conf[sel]
        # PR：recall 升序后插值（compute_ap 同款包络）
        mrec = np.concatenate(([0.0], r50, [1.0]))
        mpre = np.concatenate(([1.0], p50, [0.0]))
        mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
        py[ci] = np.interp(px, mrec, mpre)

        # P / R / F1 vs confidence（置信度从高到低累积，再插到 conf_curve）
        # 降序 conf 对应的曲线：在 conf 处取当前累积 P/R
        # np.interp 要求 xp 升序，因此翻转
        xp = conf_c[::-1]
        p_curve[ci] = np.interp(conf_curve, xp, p50[::-1], left=1.0, right=0.0)
        r_curve[ci] = np.interp(conf_curve, xp, r50[::-1], left=0.0, right=0.0)

        f1 = 2 * p50 * r50 / (p50 + r50 + eps)
        k = int(f1.argmax())
        precision[ci], recall_out[ci] = p50[k], r50[k]

    f1_curve = 2 * p_curve * r_curve / (p_curve + r_curve + eps)
    return {
        "ap": ap,
        "precision": precision,
        "recall": recall_out,
        "classes": classes,
        "px": px,
        "py": py,
        "p_curve": p_curve,
        "r_curve": r_curve,
        "f1_curve": f1_curve,
        "conf_curve": conf_curve,
    }


def confusion_matrix(
    tp: np.ndarray,
    conf: np.ndarray,
    pred_cls: np.ndarray,
    target_cls: np.ndarray,
    nc: int,
    *,
    conf_thres: float = 0.25,
    iou_idx: int = 0,
) -> np.ndarray:
    """构造 (nc+1, nc+1) 混淆矩阵，最后一维为 background（FN / FP）。

    行 = 真值，列 = 预测。仅使用 conf >= conf_thres 的预测，以及 tp[:, iou_idx]。
    """
    mat = np.zeros((nc + 1, nc + 1), dtype=np.int64)
    if target_cls.size == 0 and (conf.size == 0 or not (conf >= conf_thres).any()):
        return mat

    # stats 已展平，按类别计数近似：TP 走对角；高 conf 的 FP 记到 background 行；
    # FN = n_gt - TP，记到 background 列。
    if conf.size:
        keep = conf >= conf_thres
        tp_k = tp[keep, iou_idx].astype(bool) if tp.ndim == 2 else tp[keep].astype(bool)
        pred_k = pred_cls[keep].astype(int)
        for c in range(nc):
            sel = pred_k == c
            n_tp = int(tp_k[sel].sum())
            n_fp = int((~tp_k[sel]).sum())
            mat[c, c] += n_tp
            mat[nc, c] += n_fp  # 背景真值被预测为 c
        # FN：各类 GT 数减去 TP
        for c in range(nc):
            n_gt = int((target_cls == c).sum())
            fn = max(n_gt - int(mat[c, c]), 0)
            mat[c, nc] += fn
    else:
        for c in range(nc):
            mat[c, nc] += int((target_cls == c).sum())
    return mat
