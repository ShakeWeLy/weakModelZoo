"""推理期评估（不重训）：整体 mAP 与按面积分桶的 AP；可选把 P3 的 SAM 置为恒等（只保留 CAM）。

整体指标与 trainer/evaluator.py 的 evaluate() 同口径（process_batch + summarize_stats），
用于核对与已有结果一致。分桶 AP 与之同口径（YOLO 式 IoU 贪心一对一匹配），并加上面积区间的忽略规则：
区间外的 GT 视为忽略；先与区间内 GT 匹配，未匹配的检测若与忽略 GT 重叠或自身面积在区间外则不计入，
其余记为 FP。再用 ap_per_class 计算 AP。区间为 (0, inf) 时即为整体口径。

用法（仓库根目录，CPU）：
    CUDA_VISIBLE_DEVICES="" python experiments/detection/project2_2611/size_ap_eval.py --model ours --variant base
    CUDA_VISIBLE_DEVICES="" python experiments/detection/project2_2611/size_ap_eval.py --model ours --variant no_sam_p3
    CUDA_VISIBLE_DEVICES="" python experiments/detection/project2_2611/size_ap_eval.py --model yolo --variant base
输出：experiments/detection/project2_2611/size_ap/<model>_<variant>.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
import redundancy_analysis as ra  # noqa: E402
from diagnose_p4_cbam import MODELS  # noqa: E402
from trainer.config import ROOT  # noqa: E402
from trainer.data import YoloDetDataset, collate_fn  # noqa: E402
from trainer.evaluator import summarize_stats  # noqa: E402
from src.models.detection.YOLO.box import xywh2xyxy  # noqa: E402
from src.models.detection.YOLO.metrics import ap_per_class, process_batch  # noqa: E402
from src.models.detection.YOLO.postprocess import non_max_suppression  # noqa: E402

OUT = Path(__file__).resolve().parent / "size_ap"
IMG = 640
CONF, IOU, MAX_DET = 0.001, 0.6, 300  # 与 configs/global.toml 的 test 设置一致
IOUV = np.linspace(0.5, 0.95, 10)
# 面积区间（640 输入上的像素面积 w·h）
BUCKETS = [
    ("全部", 0.0, np.inf),
    ("<24²", 0.0, 24.0 ** 2),
    ("24²–32²", 24.0 ** 2, 32.0 ** 2),
    ("32²–96²", 32.0 ** 2, 96.0 ** 2),
    ("≥96²", 96.0 ** 2, np.inf),
]


# ---------------------------------------------------------------- 模型
class FlatSAM(nn.Module):
    """推理期消融：SAM 的注意力取全图均值（每张图一个标量），保留整体尺度、去掉空间结构。"""

    def __init__(self, sam: nn.Module) -> None:
        super().__init__()
        self.sam = sam

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = torch.mean(x, dim=1, keepdim=True)
        mx = torch.max(x, dim=1, keepdim=True).values
        att = torch.sigmoid(self.sam.conv(torch.cat([avg, mx], dim=1)))
        return att.mean(dim=(2, 3), keepdim=True) * x


def build(model_key: str, variant: str) -> nn.Module:
    spec = MODELS[model_key]
    model = ra.load_model(spec)
    if variant in ("no_sam_p3", "flat_sam_p3"):
        if "neck.cbam_p3.spatial_attention" not in dict(model.named_modules()):
            raise SystemExit(f"{spec['label']} 没有 P3 的 SAM（无 CBAM），无法做 {variant}")
        if variant == "no_sam_p3":
            model.neck.cbam_p3.spatial_attention = nn.Identity()  # 注意力 ≡ 1（会改变整体尺度，仅作参考）
        else:
            model.neck.cbam_p3.spatial_attention = FlatSAM(model.neck.cbam_p3.spatial_attention)
    return model.eval()


# ---------------------------------------------------------------- 分桶匹配
def iou_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def _greedy_one_to_one(sub: np.ndarray, thr: float) -> np.ndarray:
    """与 process_batch 相同的 YOLO 式匹配：IoU 从高到低，每个检测、每个 GT 各至多匹配一次。
    sub: (n_det, n_gt) 已把不同类别置 0。返回与 GT 匹配上的检测掩码。"""
    tp = np.zeros(sub.shape[0], dtype=bool)
    pi, pj = np.where(sub >= thr)
    if pi.size == 0:
        return tp
    v = sub[pi, pj]
    o = np.argsort(-v, kind="stable")
    pi, pj, v = pi[o], pj[o], v[o]
    _, first = np.unique(pi, return_index=True)  # 每个检测保留 IoU 最高的 GT
    pi, pj, v = pi[first], pj[first], v[first]
    o = np.argsort(-v, kind="stable")
    pi, pj = pi[o], pj[o]
    _, first = np.unique(pj, return_index=True)  # 每个 GT 保留 IoU 最高的检测
    tp[pi[first]] = True
    return tp


def match_bucket(det: np.ndarray, gt: np.ndarray, gt_cls: np.ndarray, lo: float, hi: float):
    """返回 (检测, status[T, n], GT 是否忽略, GT 类别)。status：1 = TP，0 = FP，-1 = 忽略。"""
    area_g = (gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1])
    g_ign = ~((area_g >= lo) & (area_g < hi))
    n = det.shape[0]
    area_d = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
    d_ign = ~((area_d >= lo) & (area_d < hi))
    status = np.repeat(np.where(d_ign, -1, 0).astype(np.int8)[None, :], len(IOUV), axis=0)
    if n == 0 or gt.shape[0] == 0:
        return det, status, g_ign, gt_cls
    iou = iou_np(det[:, :4], gt)
    iou = np.where(det[:, 5][:, None] == gt_cls[None, :], iou, 0.0)
    in_g, ig_g = np.where(~g_ign)[0], np.where(g_ign)[0]
    for ti, thr in enumerate(IOUV):
        tp = _greedy_one_to_one(iou[:, in_g], thr) if in_g.size else np.zeros(n, dtype=bool)
        overlap_ign = (iou[:, ig_g] >= thr).any(axis=1) if ig_g.size else np.zeros(n, dtype=bool)
        status[ti] = np.where(tp, 1, np.where(overlap_ign | d_ign, -1, 0)).astype(np.int8)
    return det, status, g_ign, gt_cls


def bucket_ap(per_img: list, lo: float, hi: float) -> dict:
    tp_l = [[] for _ in IOUV]
    conf_l = [[] for _ in IOUV]
    pcls_l = [[] for _ in IOUV]
    tcls_l = []
    for det, gt, gc in per_img:
        if gt.shape[0] == 0 and det.shape[0] == 0:
            continue
        d, status, g_ign, gcls = match_bucket(det, gt, gc, lo, hi)
        tcls_l.append(gcls[~g_ign])
        for ti in range(len(IOUV)):
            keep = status[ti] != -1
            tp_l[ti].append((status[ti][keep] == 1).astype(np.float64))
            conf_l[ti].append(d[keep, 4])
            pcls_l[ti].append(d[keep, 5])
    tcls = np.concatenate(tcls_l) if tcls_l else np.zeros(0)
    if tcls.size == 0:
        return {"n_gt": 0, "ap50": float("nan"), "map": float("nan")}
    aps = []
    for ti in range(len(IOUV)):
        tp = np.concatenate(tp_l[ti]) if tp_l[ti] else np.zeros(0)
        if tp.size == 0:
            aps.append(0.0)
            continue
        conf = np.concatenate(conf_l[ti])
        pcls = np.concatenate(pcls_l[ti])
        res = ap_per_class(np.repeat(tp[:, None], len(IOUV), axis=1), conf, pcls, tcls)
        aps.append(float(res["ap"][:, 0].mean()))
    return {"n_gt": int(tcls.size), "ap50": aps[0], "map": float(np.mean(aps))}


# ---------------------------------------------------------------- 主流程
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS), default="ours")
    ap.add_argument("--variant", choices=["base", "no_sam_p3", "flat_sam_p3"], default="base")
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()
    t0 = time.time()

    data_dir = ROOT / "data" / "Det_Fly"
    ds = YoloDetDataset(data_dir / "images" / "test", data_dir / "labels" / "test", IMG, train=False)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False, collate_fn=collate_fn)
    model = build(args.model, args.variant)

    per_img: list = []
    with torch.no_grad():
        for imgs, targets in loader:
            pred, _ = model(imgs.float() / 255.0)
            dets = non_max_suppression(pred.float(), CONF, IOU, MAX_DET)
            for si, det in enumerate(dets):
                labels = targets[targets[:, 0] == si]
                gt = (xywh2xyxy(labels[:, 2:6]) * IMG).numpy().astype(np.float64)
                gc = labels[:, 1].numpy().astype(np.float64)
                per_img.append((det.numpy().astype(np.float64), gt, gc))
    print(f"推理完成：{len(per_img)} 张，{time.time() - t0:.0f}s", flush=True)

    # 整体指标：与 evaluate() 同口径
    iouv_t = torch.from_numpy(IOUV).float()
    stats = []
    for det, gt, gc in per_img:
        correct = process_batch(torch.from_numpy(det).float(), torch.from_numpy(gt).float(),
                                torch.from_numpy(gc).float(), iouv_t)
        stats.append((correct.numpy(), det[:, 4], det[:, 5], gc))
    overall = summarize_stats(stats, names=["drone"])
    print(f"整体（evaluate 同口径）：mAP50={overall['map50']:.4f}  mAP50-95={overall['map']:.4f}  "
          f"P={overall['precision']:.4f}  R={overall['recall']:.4f}")

    buckets = {}
    print(f"{'面积区间':<10}{'GT 数':>7}{'AP50':>10}{'AP50-95':>10}")
    for label, lo, hi in BUCKETS:
        b = bucket_ap(per_img, lo, hi)
        buckets[label] = b
        print(f"{label:<10}{b['n_gt']:>7}{b['ap50']:>10.4f}{b['map']:>10.4f}")

    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"{args.model}_{args.variant}.json"
    out.write_text(json.dumps({
        "model": args.model, "label": MODELS[args.model]["label"], "variant": args.variant,
        "n_images": len(per_img),
        "overall": {"map50": overall["map50"], "map": overall["map"],
                    "precision": overall["precision"], "recall": overall["recall"]},
        "buckets": buckets, "seconds": round(time.time() - t0),
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"已保存 {out}（总耗时 {time.time() - t0:.0f}s）")


if __name__ == "__main__":
    main()
