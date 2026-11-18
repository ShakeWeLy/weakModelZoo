"""一次数据遍历，评估多个 run 在 test 集上的整体 mAP 与按面积分桶的 AP（<24² 为主指标）。

数据解码（3840×2160 JPEG）是主要耗时，所有模型共用同一次数据遍历。
口径与 size_ap_eval.py 一致：整体 mAP 同 evaluate()（process_batch + summarize_stats），
分桶 AP 采用 COCO 式忽略规则（bucket_ap）。

用法（仓库根目录）：
    python experiments/detection/project2_2611/size_ap_multi.py
输出：experiments/detection/project2_2611/size_ap/multi_2026-11-17.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
import redundancy_analysis as ra  # noqa: E402
from diagnose_p4_cbam import MODELS  # noqa: E402
from size_ap_eval import BUCKETS, CONF, IMG, IOU, IOUV, MAX_DET, bucket_ap  # noqa: E402
from trainer.config import ROOT  # noqa: E402
from trainer.data import YoloDetDataset, collate_fn  # noqa: E402
from trainer.evaluator import summarize_stats  # noqa: E402
from src.models.detection.YOLO.box import xywh2xyxy  # noqa: E402
from src.models.detection.YOLO.metrics import process_batch  # noqa: E402
from src.models.detection.YOLO.postprocess import non_max_suppression  # noqa: E402

OUT = Path(__file__).resolve().parent / "size_ap"
SEED_RUNS = {
    "ours_s42(V1)": MODELS["ours"],
    "ours_s43": {"model": "sgc_yolov5s_c3_sgdown_gbot_dw5_seed43",
                 "run": "2026-11-17_001_SGC_c3_sgdown_gbot_dw5_s_seed43_V1"},
    "ours_s44": {"model": "sgc_yolov5s_c3_sgdown_gbot_dw5_seed44",
                 "run": "2026-11-17_002_SGC_c3_sgdown_gbot_dw5_s_seed44_V1"},
    "去CBAM(s42)": {"model": "sgc_yolov5s_c3_sgdown_gbot_dw5_cbam_none",
                    "run": "2026-11-17_005_SGC_c3_sgdown_gbot_dw5_s_cbamNone_V1"},
    "P3去SAM(s42)": {"model": "sgc_yolov5s_c3_sgdown_gbot_dw5_cbam_cam_p3",
                     "run": "2026-11-17_006_SGC_c3_sgdown_gbot_dw5_s_cbamCamP3_V1"},
    "去DFC(s42)": {"model": "sgc_yolov5s_c3_sgdown_gbot_dw5_nodfc",
                   "run": "2026-11-17_007_SGC_c3_sgdown_gbot_dw5_s_noDFC_V1"},
    "残差恒等(s42)": {"model": "sgc_yolov5s_c3_sgdown_gbot_dw5_resid_identity",
                      "run": "2026-11-17_008_SGC_c3_sgdown_gbot_dw5_s_residIdentity_V1"},
    "YOLOv5s_s42(V3)": MODELS["yolo"],
    "YOLOv5s_s43": {"model": "yolov5s_seed43", "run": "2026-11-17_003_YOLOv5s_seed43_V1"},
    "YOLOv5s_s44": {"model": "yolov5s_seed44", "run": "2026-11-17_004_YOLOv5s_seed44_V1"},
}


def main() -> None:
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = ROOT / "data" / "Det_Fly"
    ds = YoloDetDataset(data_dir / "images" / "test", data_dir / "labels" / "test", IMG, train=False)
    loader = DataLoader(ds, batch_size=16, shuffle=False, collate_fn=collate_fn)

    models = {name: ra.load_model(spec).eval().to(device) for name, spec in SEED_RUNS.items()}
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
    print(f"推理完成（{device.type}）：{len(ds)} 张 × {len(models)} 个模型，{time.time() - t0:.0f}s", flush=True)

    iouv_t = torch.from_numpy(IOUV).float()
    results = {}
    print(f"\n{'run':<16}{'mAP50':>8}{'mAP50-95':>10}{'P':>8}{'R':>8} | "
          f"{'<24² AP50':>10}{'<24² AP':>9}{'24²–32²':>10}{'32²–96²':>10}")
    for name in models:
        stats = []
        for det, gt, gc in per[name]:
            correct = process_batch(torch.from_numpy(det).float(), torch.from_numpy(gt).float(),
                                    torch.from_numpy(gc).float(), iouv_t)
            stats.append((correct.numpy(), det[:, 4], det[:, 5], gc))
        overall = summarize_stats(stats, names=["drone"])
        buckets = {label: bucket_ap(per[name], lo, hi) for label, lo, hi in BUCKETS}
        results[name] = {"overall": {k: overall[k] for k in ("map50", "map", "precision", "recall")},
                         "buckets": buckets}
        b_small, b_mid, b_big = buckets["<24²"], buckets["24²–32²"], buckets["32²–96²"]
        print(f"{name:<16}{overall['map50']:>8.4f}{overall['map']:>10.4f}{overall['precision']:>8.4f}"
              f"{overall['recall']:>8.4f} | {b_small['ap50']:>10.4f}{b_small['map']:>9.4f}"
              f"{b_mid['ap50']:>10.4f}{b_big['ap50']:>10.4f}", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "multi_2026-11-17.json"
    out.write_text(json.dumps({"n_images": len(ds), "results": results,
                               "seconds": round(time.time() - t0)}, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print(f"\n已保存 {out}（总耗时 {time.time() - t0:.0f}s）")


if __name__ == "__main__":
    main()
