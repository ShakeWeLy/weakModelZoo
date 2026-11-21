"""同一测试集、同一预处理、同一 NMS 配置下，比较四个推理变体的 mAP：
  1) PyTorch 原始模型（best.pth，TF32 关闭）
  2) PyTorch BN 折叠模型
  3) TensorRT FP32 引擎（由 BN 折叠模型导出）
  4) TensorRT FP16 引擎（由 BN 折叠模型导出）

实现要点：
- 数据：data/Det_Fly/images/test（1501 张），YoloDetDataset(train=False)，即 letterbox 到 640；
- 每张图只解码与预处理一次，然后依次送入四个变体，保证输入完全相同；
- NMS、TP 匹配、mAP 汇总复用官方评估代码（non_max_suppression / process_batch / summarize_stats），
  与 trainer/evaluator.py::evaluate 的逻辑逐行一致，只是把四个变体放在同一循环里；
- 配置：conf_thres / iou_thres / max_det 取自 configs 的 test 段（与 analyze.py 相同）；
- batch 固定为 1（TRT 引擎为静态形状 batch=1）。

用法（仓库根目录）：python experiments/detection/project2_2611/no_train_analysis/trt_map_eval.py
输出：outputs/map_eval/map_eval_2026-11-21.json
"""

from __future__ import annotations

import copy
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

_HERE = Path(__file__).resolve().parent
_PROJ = _HERE.parent                      # experiments/detection/project2_2611
_ROOT = _PROJ.parents[2]                  # weakModelZoo（src/ 所在目录）
for p in (_ROOT, _PROJ, _PROJ / "analysis_scripts", _HERE):
    sys.path.insert(0, str(p))

from trainer import build_config, model_output_dir  # noqa: E402
from trainer.data import YoloDetDataset, collate_fn  # noqa: E402
from trainer.evaluator import summarize_stats  # noqa: E402
from src.models.detection.YOLO.box import xywh2xyxy  # noqa: E402
from src.models.detection.YOLO.metrics import process_batch  # noqa: E402
from src.models.detection.YOLO.postprocess import non_max_suppression  # noqa: E402
import redundancy_analysis as ra  # noqa: E402
from bn_fold import fold_bn_inplace  # noqa: E402
from trt_deploy_bench import TRTRunner  # noqa: E402

OUT = _HERE / "outputs" / "map_eval"
TRT_DIR = _HERE / "outputs" / "trt"
DATA_DIR = _ROOT / "data" / "Det_Fly"
MODELS = {
    "SGC 无注意力": {"key": "sgc_yolov5s_c3_sgdown_gbot_dw5", "tag": "sgc",
                     "spec": {"model": "sgc_yolov5s_c3_sgdown_gbot_dw5",
                              "run": "2026-11-17_015_SGC_c3_sgdown_gbot_dw5_s_noDFC_cbamNone_seed42_V1"}},
    "YOLOv5s": {"key": "yolov5s", "tag": "yolov5s",
                "spec": {"model": "yolov5s", "run": "2026-10-30_002_YOLOv5s_V3"}},
}


def main() -> None:
    t0 = time.time()
    dev = torch.device("cuda")
    torch.backends.cudnn.allow_tf32 = False       # PyTorch 两个变体使用严格 FP32
    torch.backends.cuda.matmul.allow_tf32 = False
    OUT.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {}

    for label, m in MODELS.items():
        print(f"\n== {label}", flush=True)
        cfg = build_config(m["key"])
        test = cfg["test"]
        conf_thres = float(test.get("conf_thres", 0.001))
        iou_thres = float(test.get("iou_thres", 0.6))
        max_det = int(test.get("max_det", 300))
        run_dir = model_output_dir(cfg) / m["spec"]["run"]
        ckpt = torch.load(run_dir / "checkpoints" / "best.pth", map_location="cpu", weights_only=False)
        names, img_size = list(ckpt["names"]), int(ckpt["img_size"])
        print(f"   run={m['spec']['run']} | img_size={img_size} | names={names} | "
              f"conf={conf_thres} iou={iou_thres} max_det={max_det}", flush=True)

        orig = ra.load_model(m["spec"]).eval().to(dev)
        folded = copy.deepcopy(orig).eval()
        n_fold = fold_bn_inplace(folded)
        print(f"   BN 折叠 block 数：{n_fold}", flush=True)
        runners = {}
        for prec in ("fp32", "fp16"):
            runners[prec] = TRTRunner((TRT_DIR / f"{m['tag']}_{prec}.engine").read_bytes())

        variants = {
            "PyTorch 原版": lambda x, net=orig: net(x)[0],
            "PyTorch BN折叠": lambda x, net=folded: net(x)[0],
            "TRT FP32": lambda x, r=runners["fp32"]: _trt_pred(r, x),
            "TRT FP16": lambda x, r=runners["fp16"]: _trt_pred(r, x),
        }

        ds = YoloDetDataset(DATA_DIR / "images" / "test", DATA_DIR / "labels" / "test", img_size, train=False)
        loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=4)
        iouv = torch.linspace(0.5, 0.95, 10, device=dev)
        stats = {v: [] for v in variants}
        n_det = {v: 0 for v in variants}
        t_loop = time.time()
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
                    stats[vname].append((
                        correct.cpu().numpy(),
                        det[:, 4].cpu().numpy(),
                        det[:, 5].cpu().numpy(),
                        gt_cls.cpu().numpy(),
                    ))
                    n_det[vname] += int(det.shape[0])
            if (i + 1) % 300 == 0:
                print(f"   进度 {i + 1}/{len(ds)}（{time.time() - t_loop:.0f}s）", flush=True)

        res = {}
        for vname in variants:
            s = summarize_stats(stats[vname], names, with_curves=False)
            res[vname] = {
                "num_images": len(stats[vname]),
                "mAP50": s["map50"], "mAP50_95": s["map"],
                "precision": s["precision"], "recall": s["recall"],
                "per_class_ap50": s["per_class_ap50"],
                "total_detections": n_det[vname],
            }
            print(f"   {vname:<14} mAP50 {s['map50']:.4f} | mAP50-95 {s['map']:.4f} | "
                  f"P {s['precision']:.4f} | R {s['recall']:.4f} | 检测框数 {n_det[vname]}", flush=True)
        # 保存逐图统计量，供配对 bootstrap 使用（trt_map_bootstrap.py）
        with open(OUT / f"stats_{m['tag']}.pkl", "wb") as f:
            pickle.dump({"stats": stats, "names": names, "img_size": img_size}, f)
        results[label] = {"run": m["spec"]["run"], "img_size": img_size, "names": names,
                          "conf_thres": conf_thres, "iou_thres": iou_thres, "max_det": max_det,
                          "bn_folded_blocks": n_fold, "variants": res}

        # 保存每个模型的结果，防止后续步骤中断丢失
        (OUT / "map_eval_2026-11-21.json").write_text(
            json.dumps({"results": results, "seconds": round(time.time() - t0)}, ensure_ascii=False, indent=1),
            encoding="utf-8")
        del orig, folded, runners
        torch.cuda.empty_cache()

    print(f"\n已保存 {OUT / 'map_eval_2026-11-21.json'}（总耗时 {time.time() - t0:.0f}s）")


def _trt_pred(runner: TRTRunner, x: torch.Tensor) -> torch.Tensor:
    runner.set_input(x)
    runner.run()
    torch.cuda.synchronize()
    return dict(runner.outputs)["out0"]


if __name__ == "__main__":
    main()
