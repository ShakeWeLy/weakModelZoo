"""检测评估：推理 + NMS + TP 匹配 + mAP，训练每 epoch 验证与 analyze 共用同一套逻辑。"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.models.detection.YOLO.box import xywh2xyxy
from src.models.detection.YOLO.metrics import ap_per_class, confusion_matrix, process_batch
from src.models.detection.YOLO.postprocess import non_max_suppression


def summarize_stats(
    stats: Sequence[tuple[np.ndarray, ...]],
    names: Sequence[str],
    *,
    with_curves: bool = False,
    conf_thres_cm: float = 0.25,
) -> dict[str, Any]:
    """把逐图统计汇总为 mAP / P / R。stats 元素为 (correct, conf, pred_cls, target_cls)。

    with_curves=True 时额外返回 curves（PR/F1/P/R）与 confusion（含 background）。
    """
    empty = {"precision": 0.0, "recall": 0.0, "map50": 0.0, "map": 0.0, "per_class_ap50": {}}
    if not stats:
        return empty
    tp = np.concatenate([s[0] for s in stats], 0)
    conf = np.concatenate([s[1] for s in stats], 0)
    pred_cls = np.concatenate([s[2] for s in stats], 0)
    target_cls = np.concatenate([s[3] for s in stats], 0)
    if target_cls.size == 0:
        return empty

    res = ap_per_class(tp, conf, pred_cls, target_cls)
    ap = res["ap"]
    per_class = {
        names[int(c)] if int(c) < len(names) else str(int(c)): float(ap[i, 0])
        for i, c in enumerate(res["classes"])
    }
    out: dict[str, Any] = {
        "precision": float(res["precision"].mean()),
        "recall": float(res["recall"].mean()),
        "map50": float(ap[:, 0].mean()),
        "map": float(ap.mean()),
        "per_class_ap50": per_class,
    }
    if with_curves:
        class_names = [
            names[int(c)] if int(c) < len(names) else str(int(c)) for c in res["classes"]
        ]
        out["curves"] = {
            "names": class_names,
            "px": res["px"],
            "py": res["py"],
            "p_curve": res["p_curve"],
            "r_curve": res["r_curve"],
            "f1_curve": res["f1_curve"],
            "conf_curve": res["conf_curve"],
            "ap50": ap[:, 0],
        }
        out["confusion"] = confusion_matrix(
            tp, conf, pred_cls, target_cls, nc=len(names), conf_thres=conf_thres_cm
        )
    return out


def _criterion_call(criterion: torch.nn.Module, raw: Any, targets: torch.Tensor):
    """兼容 YOLOv5 raw=list 与 YOLOv8 raw=dict。"""
    if isinstance(raw, dict):
        return criterion(raw, targets)
    return criterion([r.float() for r in raw], targets)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    img_size: int,
    names: Sequence[str],
    criterion: torch.nn.Module | None = None,
    conf_thres: float = 0.001,
    iou_thres: float = 0.6,
    max_det: int = 300,
    use_amp: bool = False,
    with_curves: bool = False,
) -> dict[str, Any]:
    """在 loader 上计算 mAP@0.5、mAP@0.5:0.95、P、R；若给定 criterion 则同时统计验证损失。

    with_curves=True 时额外返回 curves / confusion，供官方风格指标图使用。
    """
    model.eval()
    iouv = torch.linspace(0.5, 0.95, 10, device=device)
    stats: list[tuple[np.ndarray, ...]] = []
    item_names = tuple(getattr(criterion, "item_names", ("box", "obj", "cls"))) if criterion is not None else ()
    loss_sum = torch.zeros(len(item_names)) if item_names else None
    n_batches = 0
    num_images = 0

    for imgs, targets in loader:
        imgs = imgs.to(device, non_blocking=True).float() / 255.0
        targets = targets.to(device)
        with torch.autocast(device.type, enabled=use_amp):
            pred, raw = model(imgs)

        if criterion is not None and loss_sum is not None:
            _, items = _criterion_call(criterion, raw, targets)
            loss_sum += items.detach().float().cpu()
            n_batches += 1

        dets = non_max_suppression(pred.float(), conf_thres, iou_thres, max_det)
        for si, det in enumerate(dets):
            labels = targets[targets[:, 0] == si]
            gt_cls = labels[:, 1]
            gt_xyxy = xywh2xyxy(labels[:, 2:6]) * img_size
            correct = process_batch(det, gt_xyxy, gt_cls, iouv)
            stats.append((
                correct.cpu().numpy(),
                det[:, 4].cpu().numpy(),
                det[:, 5].cpu().numpy(),
                gt_cls.cpu().numpy(),
            ))
            num_images += 1

    results = summarize_stats(stats, names, with_curves=with_curves)
    results["num_images"] = num_images
    if criterion is not None and loss_sum is not None and n_batches:
        means = (loss_sum / n_batches).tolist()
        for name, val in zip(item_names, means):
            results[f"{name}_loss"] = val
        results["val_loss"] = float(sum(means))
    return results
