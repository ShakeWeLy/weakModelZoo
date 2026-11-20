"""无训练：五个模型 × 3 个 seed（共 15 个 checkpoint）在 test 集上的整体 mAP 与按面积分桶的 AP。

数据解码（3840×2160 JPEG）是主要耗时，所有模型共用一次数据遍历。
口径：整体 mAP 与 evaluate() 一致（process_batch + summarize_stats）；分桶 AP 为 COCO 式忽略规则（见 size_ap_eval.py）。
用法（仓库根目录）：python experiments/detection/project2_2611/no_train_analysis/size_ap_five.py
输出：outputs/size_ap/size_ap_five_2026-11-20.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

_PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJ))
sys.path.insert(0, str(_PROJ / "analysis_scripts"))
import redundancy_analysis as ra  # noqa: E402
from size_ap_eval import BUCKETS, CONF, IMG, IOU, IOUV, MAX_DET, bucket_ap  # noqa: E402
from trainer.config import ROOT  # noqa: E402
from trainer.data import YoloDetDataset, collate_fn  # noqa: E402
from trainer.evaluator import summarize_stats  # noqa: E402
from src.models.detection.YOLO.box import xywh2xyxy  # noqa: E402
from src.models.detection.YOLO.metrics import process_batch  # noqa: E402
from src.models.detection.YOLO.postprocess import non_max_suppression  # noqa: E402

OUT = Path(__file__).resolve().parent / "outputs" / "size_ap"

# 组名 -> [(seed 标签, 加载规格)]；规格的 model 为配置名（用于定位 runs 目录），run 为运行目录名
MODEL_RUNS = {
    "YOLOv5s": [
        ("s42", {"model": "yolov5s", "run": "2026-10-30_002_YOLOv5s_V3"}),
        ("s43", {"model": "yolov5s_seed43", "run": "2026-11-17_003_YOLOv5s_seed43_V1"}),
        ("s44", {"model": "yolov5s_seed44", "run": "2026-11-17_004_YOLOv5s_seed44_V1"}),
    ],
    "SGC 完整 dw5": [
        ("s42", {"model": "sgc_yolov5s_c3_sgdown_gbot_dw5",
                 "run": "2026-11-18_018_SGC_c3_sgdown_gbot_dw5_s_seed42_rerun_V2"}),
        ("s43", {"model": "sgc_yolov5s_c3_sgdown_gbot_dw5",
                 "run": "2026-11-17_001_SGC_c3_sgdown_gbot_dw5_s_seed43_V1"}),
        ("s44", {"model": "sgc_yolov5s_c3_sgdown_gbot_dw5",
                 "run": "2026-11-17_002_SGC_c3_sgdown_gbot_dw5_s_seed44_V1"}),
    ],
    "SGC 无 DFC + 无 CBAM": [
        ("s42", {"model": "sgc_yolov5s_c3_sgdown_gbot_dw5",
                 "run": "2026-11-17_015_SGC_c3_sgdown_gbot_dw5_s_noDFC_cbamNone_seed42_V1"}),
        ("s43", {"model": "sgc_yolov5s_c3_sgdown_gbot_dw5",
                 "run": "2026-11-17_016_SGC_c3_sgdown_gbot_dw5_s_noDFC_cbamNone_seed43_V2"}),
        ("s44", {"model": "sgc_yolov5s_c3_sgdown_gbot_dw5",
                 "run": "2026-11-17_017_SGC_c3_sgdown_gbot_dw5_s_noDFC_cbamNone_seed44_V1"}),
    ],
    "SGC Conv↓ DW5": [
        ("s42", {"model": "sgc_yolov5s_c3_ghostbot_dw5", "run": "2026-11-12_027_SGC_c3_ghostbot_dw5_s_V1"}),
        ("s43", {"model": "sgc_yolov5s_c3_ghostbot_dw5", "run": "2026-11-19_034_SGC_c3_ghostbot_dw5_s_seed43_V1"}),
        ("s44", {"model": "sgc_yolov5s_c3_ghostbot_dw5", "run": "2026-11-19_035_SGC_c3_ghostbot_dw5_s_seed44_V1"}),
    ],
    "SGC SGhost↓ DW3": [
        ("s42", {"model": "sgc_yolov5s_c3_sgdown_gbot_dw3",
                 "run": "2026-11-19_025_SGC_c3_sgdown_gbot_dw3_s_seed42_V1"}),
        ("s43", {"model": "sgc_yolov5s_c3_sgdown_gbot_dw3",
                 "run": "2026-11-19_026_SGC_c3_sgdown_gbot_dw3_s_seed43_V1"}),
        ("s44", {"model": "sgc_yolov5s_c3_sgdown_gbot_dw3",
                 "run": "2026-11-19_027_SGC_c3_sgdown_gbot_dw3_s_seed44_V1"}),
    ],
}


# 补充：只去掉一个模块（用于分离 DFC 与 CBAM 各自的作用；与主表同 seed）
SUPP_RUNS = {
    "补充：仅去 DFC": [
        ("s42", {"model": "sgc_yolov5s_c3_sgdown_gbot_dw5", "run": "2026-11-17_007_SGC_c3_sgdown_gbot_dw5_s_noDFC_V1"}),
        ("s43", {"model": "sgc_yolov5s_c3_sgdown_gbot_dw5",
                 "run": "2026-11-19_019_SGC_c3_sgdown_gbot_dw5_s_noDFC_seed43_V1"}),
        ("s44", {"model": "sgc_yolov5s_c3_sgdown_gbot_dw5",
                 "run": "2026-11-19_020_SGC_c3_sgdown_gbot_dw5_s_noDFC_seed44_V1"}),
    ],
    "补充：仅去 CBAM": [
        ("s42", {"model": "sgc_yolov5s_c3_sgdown_gbot_dw5", "run": "2026-11-17_005_SGC_c3_sgdown_gbot_dw5_s_cbamNone_V1"}),
        ("s43", {"model": "sgc_yolov5s_c3_sgdown_gbot_dw5",
                 "run": "2026-11-19_021_SGC_c3_sgdown_gbot_dw5_s_cbamNone_seed43_V1"}),
        ("s44", {"model": "sgc_yolov5s_c3_sgdown_gbot_dw5",
                 "run": "2026-11-19_022_SGC_c3_sgdown_gbot_dw5_s_cbamNone_seed44_V1"}),
    ],
}
ALL_RUNS = {**MODEL_RUNS, **SUPP_RUNS}


def group_stats(vals: list[float]) -> tuple[float, float]:
    a = np.array(vals, dtype=float)
    return float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else 0.0


def main() -> None:
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = ROOT / "data" / "Det_Fly"
    ds = YoloDetDataset(data_dir / "images" / "test", data_dir / "labels" / "test", IMG, train=False)
    loader = DataLoader(ds, batch_size=16, shuffle=False, collate_fn=collate_fn)

    runs = [(f"{g} {s}", g, s, spec) for g, items in ALL_RUNS.items() for s, spec in items]
    models = {name: ra.load_model(spec).eval().to(device) for name, _, _, spec in runs}
    per: dict[str, list] = {name: [] for name in models}
    with torch.no_grad():
        for imgs, targets in loader:
            x = imgs.float().to(device) / 255.0
            for name, m in models.items():
                pred, _ = m(x)
                dets = non_max_suppression(pred.float(), CONF, IOU, MAX_DET)
                for si, det in enumerate(dets):
                    labels = targets[targets[:, 0] == si]
                    gt = (xywh2xyxy(labels[:, 2:6]) * IMG).numpy().astype(np.float64)
                    gc = labels[:, 1].numpy().astype(np.float64)
                    per[name].append((det.cpu().numpy().astype(np.float64), gt, gc))
    print(f"推理完成（{device.type}）：{len(ds)} 张 × {len(models)} 个 checkpoint，{time.time() - t0:.0f}s", flush=True)

    iouv_t = torch.from_numpy(IOUV).float()
    results: dict[str, dict] = {}
    print(f"\n{'run':<24}{'mAP50':>8}{'mAP50-95':>10}{'<24² AP50':>11}{'<24² AP':>9}{'24–32':>8}{'32–96':>8}")
    for name, group, seed, _ in runs:
        stats = []
        for det, gt, gc in per[name]:
            correct = process_batch(torch.from_numpy(det).float(), torch.from_numpy(gt).float(),
                                    torch.from_numpy(gc).float(), iouv_t)
            stats.append((correct.numpy(), det[:, 4], det[:, 5], gc))
        overall = summarize_stats(stats, names=["drone"])
        buckets = {label: bucket_ap(per[name], lo, hi) for label, lo, hi in BUCKETS}
        results[name] = {"group": group, "seed": seed,
                         "overall": {k: overall[k] for k in ("map50", "map", "precision", "recall")},
                         "buckets": buckets}
        s, mid, big = buckets["<24²"], buckets["24²–32²"], buckets["32²–96²"]
        print(f"{name:<24}{overall['map50']:>8.4f}{overall['map']:>10.4f}{s['ap50']:>11.4f}{s['map']:>9.4f}"
              f"{mid['ap50']:>8.4f}{big['ap50']:>8.4f}", flush=True)

    # 组级汇总（三个 seed 的均值 ± 样本标准差）
    summary = {}
    for group in ALL_RUNS:
        rs = [v for v in results.values() if v["group"] == group]
        summary[group] = {
            "mAP50": group_stats([r["overall"]["map50"] for r in rs]),
            "mAP50-95": group_stats([r["overall"]["map"] for r in rs]),
            "small_AP50": group_stats([r["buckets"]["<24²"]["ap50"] for r in rs]),
            "small_AP50-95": group_stats([r["buckets"]["<24²"]["map"] for r in rs]),
            "mid1_AP50": group_stats([r["buckets"]["24²–32²"]["ap50"] for r in rs]),
            "mid2_AP50": group_stats([r["buckets"]["32²–96²"]["ap50"] for r in rs]),
        }
    print("\n组级（均值 ± std，n=3）：")
    for g, v in summary.items():
        print(f"  {g}: mAP50 {v['mAP50'][0]:.4f}±{v['mAP50'][1]:.4f}  <24² AP50 {v['small_AP50'][0]:.4f}±{v['small_AP50'][1]:.4f}"
              f"  mAP50-95 {v['mAP50-95'][0]:.4f}±{v['mAP50-95'][1]:.4f}")

    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "size_ap_five_2026-11-20.json"
    out.write_text(json.dumps({"n_images": len(ds), "runs": results, "groups": summary,
                               "seconds": round(time.time() - t0)}, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print(f"\n已保存 {out}（总耗时 {time.time() - t0:.0f}s）")


if __name__ == "__main__":
    main()
