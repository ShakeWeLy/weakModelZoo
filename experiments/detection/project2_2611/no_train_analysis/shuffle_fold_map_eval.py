"""置换折叠的精度验证（SGC，测试集 1501 张，同一预处理、同一 NMS 配置）。

四个变体：
  1) PyTorch BN 折叠（参考，即折叠前的图）
  2) PyTorch 置换折叠（FP32，数学等价应与 1) 完全一致）
  3) TRT FP16 原始（sgc_fp16.engine）
  4) TRT FP16 置换折叠（sgc_fold_fp16.engine）
并输出各变体相对 1) 的配对 bootstrap 差值（按图像重采样，B=300）。

用法（仓库根目录）：python experiments/detection/project2_2611/no_train_analysis/shuffle_fold_map_eval.py
输出：outputs/shuffle_fold/fold_map_2026-11-21.json，stats_fold.pkl
"""

from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

_HERE = Path(__file__).resolve().parent
_PROJ = _HERE.parent
_ROOT = _PROJ.parents[2]
for p in (_ROOT, _PROJ, _PROJ / "analysis_scripts", _HERE):
    sys.path.insert(0, str(p))

from trainer import build_config  # noqa: E402
from trainer.data import YoloDetDataset, collate_fn  # noqa: E402
from trainer.evaluator import summarize_stats  # noqa: E402
from src.models.detection.YOLO.box import xywh2xyxy  # noqa: E402
from src.models.detection.YOLO.metrics import process_batch  # noqa: E402
from src.models.detection.YOLO.postprocess import non_max_suppression  # noqa: E402
from shuffle_fold import OUT, fold_graph  # noqa: E402
from trt_deploy_bench import TRTRunner  # noqa: E402
from trt_map_eval import DATA_DIR, TRT_DIR  # noqa: E402

B = 300
REF = "PyTorch BN折叠（参考）"
VARIANTS = ["PyTorch 置换折叠 FP32", "TRT FP16 原始", "TRT FP16 置换折叠"]


def _trt_pred(runner: TRTRunner, x: torch.Tensor) -> torch.Tensor:
    runner.set_input(x)
    runner.run()
    torch.cuda.synchronize()
    return dict(runner.outputs)["out0"]


def main() -> None:
    t0 = time.time()
    dev = torch.device("cuda")
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    OUT.mkdir(parents=True, exist_ok=True)

    cfg = build_config("sgc_yolov5s_c3_sgdown_gbot_dw5")
    test = cfg["test"]
    conf_thres = float(test.get("conf_thres", 0.001))
    iou_thres = float(test.get("iou_thres", 0.6))
    max_det = int(test.get("max_det", 300))
    img_size = 640
    names = ["drone"]

    ref, new, _ = fold_graph()
    ref = ref.to(dev).eval()
    new = new.to(dev).eval()
    r_orig = TRTRunner((TRT_DIR / "sgc_fp16.engine").read_bytes())
    r_fold = TRTRunner((OUT / "sgc_fold_fp16.engine").read_bytes())

    variants = {
        REF: lambda x: ref(x)[0],
        "PyTorch 置换折叠 FP32": lambda x: new(x)[0],
        "TRT FP16 原始": lambda x: _trt_pred(r_orig, x),
        "TRT FP16 置换折叠": lambda x: _trt_pred(r_fold, x),
    }
    ds = YoloDetDataset(DATA_DIR / "images" / "test", DATA_DIR / "labels" / "test", img_size, train=False)
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=4)
    iouv = torch.linspace(0.5, 0.95, 10, device=dev)
    stats = {v: [] for v in variants}
    n_det = {v: 0 for v in variants}
    for i, (imgs, targets) in enumerate(loader):
        imgs = imgs.to(dev, non_blocking=True).float() / 255.0
        targets = targets.to(dev)
        for vname, fwd in variants.items():
            with torch.no_grad():
                pred = fwd(imgs)
            dets = non_max_suppression(pred.float(), conf_thres, iou_thres, max_det)
            for si, det in enumerate(dets):
                labels = targets[targets[:, 0] == si]
                gt_cls = labels[:, 1]
                gt_xyxy = xywh2xyxy(labels[:, 2:6]) * img_size
                correct = process_batch(det, gt_xyxy, gt_cls, iouv)
                stats[vname].append((correct.cpu().numpy(), det[:, 4].cpu().numpy(),
                                     det[:, 5].cpu().numpy(), gt_cls.cpu().numpy()))
                n_det[vname] += int(det.shape[0])
        if (i + 1) % 500 == 0:
            print(f"进度 {i + 1}/{len(ds)}（{time.time() - t0:.0f}s）", flush=True)

    res = {}
    for v in variants:
        s = summarize_stats(stats[v], names, with_curves=False)
        res[v] = {"mAP50": s["map50"], "mAP50_95": s["map"], "precision": s["precision"],
                  "recall": s["recall"], "total_detections": n_det[v]}
        print(f"{v:<22} mAP50 {s['map50']:.4f} | mAP50-95 {s['map']:.4f} | P {s['precision']:.4f} | "
              f"R {s['recall']:.4f} | 检测框数 {n_det[v]}", flush=True)

    # 配对 bootstrap：各变体相对参考的差值
    rng = np.random.default_rng(0)
    n = len(stats[REF])
    boot = {v: {"d50": [], "d95": []} for v in variants}
    for _ in range(B):
        idx = rng.integers(0, n, n)
        r = summarize_stats([stats[REF][i] for i in idx], names)
        for v in variants:
            s = summarize_stats([stats[v][i] for i in idx], names)
            boot[v]["d50"].append(s["map50"] - r["map50"])
            boot[v]["d95"].append(s["map"] - r["map"])
    for v in variants:
        d50, d95 = np.array(boot[v]["d50"]), np.array(boot[v]["d95"])
        res[v]["delta_mAP50"] = res[v]["mAP50"] - res[REF]["mAP50"] if REF in res else None
        res[v]["delta_mAP50_CI95"] = [float(np.percentile(d50, 2.5)), float(np.percentile(d50, 97.5))]
        res[v]["delta_mAP50_95"] = res[v]["mAP50_95"] - res[REF]["mAP50_95"] if REF in res else None
        res[v]["delta_mAP50_95_CI95"] = [float(np.percentile(d95, 2.5)), float(np.percentile(d95, 97.5))]
        print(f"  Δ {v:<22} ΔmAP50 {res[v]['delta_mAP50']:+.5f} CI95 {res[v]['delta_mAP50_CI95']} | "
              f"ΔmAP50-95 {res[v]['delta_mAP50_95']:+.5f} CI95 {res[v]['delta_mAP50_95_CI95']}", flush=True)
    res[REF]["delta_mAP50"] = 0.0
    res[REF]["delta_mAP50_95"] = 0.0

    with open(OUT / "stats_fold.pkl", "wb") as f:
        pickle.dump({"stats": stats, "names": names}, f)
    (OUT / "fold_map_2026-11-21.json").write_text(
        json.dumps({"B": B, "results": res, "seconds": round(time.time() - t0)}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    print(f"已保存 {OUT / 'fold_map_2026-11-21.json'}（总耗时 {time.time() - t0:.0f}s）")


if __name__ == "__main__":
    main()
