"""无训练：五个模型的推理延迟（每个模型用 seed 42 的 checkpoint；延迟与 seed 无关）。

测量项（均为中位数，单位 ms）：
- GPU batch 1：模型前向（PyTorch eager，FP32）
- GPU batch 1 + NMS：加入后处理（conf 0.25、IoU 0.6、max_det 300，与部署阈值一致）
- GPU batch 8：前向耗时，并换算吞吐（img/s）
- CPU batch 1：模型前向（eager，默认线程数）
- 预处理：YoloDetDataset 单张（解码 3840×2160 JPEG + letterbox），所有模型共用，单独报告

注意：输入为随机张量，未导出 ONNX / 未折叠 BN，结果反映的是 eager 模式下的开销，不等同于目标运行时的速度。

用法（仓库根目录）：python experiments/detection/project2_2611/bench_latency_five.py
输出：bench_latency/latency_five_2026-11-20.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

# 本脚本位于 project2_2611/no_train_analysis/：trainer 在 project2_2611 下，redundancy_analysis 在 analysis_scripts 下
_PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJ))
sys.path.insert(0, str(_PROJ / "analysis_scripts"))
import redundancy_analysis as ra  # noqa: E402
from trainer.config import ROOT  # noqa: E402
from trainer.data import YoloDetDataset  # noqa: E402
from src.models.detection.YOLO.postprocess import non_max_suppression  # noqa: E402

OUT = Path(__file__).resolve().parent / "outputs" / "bench_latency"
IMG = 640
SPECS = {
    "YOLOv5s": {"model": "yolov5s", "run": "2026-10-30_002_YOLOv5s_V3", "flops_g": 15.94},
    "SGC 完整 dw5": {"model": "sgc_yolov5s_c3_sgdown_gbot_dw5",
                     "run": "2026-11-18_018_SGC_c3_sgdown_gbot_dw5_s_seed42_rerun_V2", "flops_g": 10.80},
    "SGC 无 DFC + 无 CBAM": {"model": "sgc_yolov5s_c3_sgdown_gbot_dw5",
                            "run": "2026-11-17_015_SGC_c3_sgdown_gbot_dw5_s_noDFC_cbamNone_seed42_V1", "flops_g": 10.75},
    "SGC Conv↓ DW5": {"model": "sgc_yolov5s_c3_ghostbot_dw5",
                      "run": "2026-11-12_027_SGC_c3_ghostbot_dw5_s_V1", "flops_g": 12.68},
    "SGC SGhost↓ DW3": {"model": "sgc_yolov5s_c3_sgdown_gbot_dw3",
                        "run": "2026-11-19_025_SGC_c3_sgdown_gbot_dw3_s_seed42_V1", "flops_g": 10.76},
}


def timeit(fn, sync, iters: int, warmup: int) -> tuple[float, float]:
    with torch.no_grad():
        for _ in range(warmup):
            fn()
    sync()
    ts = []
    with torch.no_grad():
        for _ in range(iters):
            t0 = time.perf_counter()
            fn()
            sync()
            ts.append(time.perf_counter() - t0)
    a = np.array(ts) * 1000
    return float(np.median(a)), float(np.percentile(a, 90))


def main() -> None:
    t_all = time.time()
    print(f"CPU 线程数 {torch.get_num_threads()}；GPU：{torch.cuda.get_device_name(0)}；torch {torch.__version__}")
    gpu = torch.device("cuda")
    cuda_sync = torch.cuda.synchronize
    no_sync = lambda: None  # noqa: E731

    data_dir = ROOT / "data" / "Det_Fly"
    ds = YoloDetDataset(data_dir / "images" / "test", data_dir / "labels" / "test", IMG, train=False)
    for i in range(3):
        ds[i]
    pre = []
    for i in range(3, 33):
        t0 = time.perf_counter()
        ds[i]
        pre.append((time.perf_counter() - t0) * 1000)
    pre_ms = float(np.median(pre))
    print(f"预处理（解码 + letterbox，单张，中位数）：{pre_ms:.1f} ms\n")

    rows = []
    for name, spec in SPECS.items():
        model = ra.load_model(spec).eval()
        params_m = sum(p.numel() for p in model.parameters()) / 1e6
        model = model.to(gpu)
        x1 = torch.rand(1, 3, IMG, IMG, device=gpu)
        x8 = torch.rand(8, 3, IMG, IMG, device=gpu)
        g1 = timeit(lambda: model(x1), cuda_sync, 200, 30)
        g1n = timeit(lambda: non_max_suppression(model(x1)[0], 0.25, 0.6, 300), cuda_sync, 100, 20)
        g8 = timeit(lambda: model(x8), cuda_sync, 100, 15)
        model = model.to("cpu")
        xc = torch.rand(1, 3, IMG, IMG)
        c1 = timeit(lambda: model(xc), no_sync, 30, 5)
        row = {"model": name, "params_m": params_m, "flops_g": spec["flops_g"],
               "gpu_b1_ms": g1[0], "gpu_b1_p90_ms": g1[1],
               "gpu_b1_nms_ms": g1n[0],
               "gpu_b8_ms": g8[0], "gpu_b8_img_per_s": 8 / (g8[0] / 1000),
               "cpu_b1_ms": c1[0], "cpu_b1_p90_ms": c1[1]}
        rows.append(row)
        print(f"{name:<22} 参数 {params_m:.2f}M  GPU b1 {g1[0]:6.2f} ms  +NMS {g1n[0]:6.2f} ms  "
              f"GPU b8 {g8[0]:7.2f} ms（{row['gpu_b8_img_per_s']:.0f} img/s）  CPU b1 {c1[0]:7.1f} ms（P90 {c1[1]:.1f}）",
              flush=True)
        del model

    print("\n表（中位数）：模型 | 参数量 | FLOPs | GPU b1 | GPU b1+NMS | GPU b8 img/s | CPU b1")
    for r in rows:
        print(f"{r['model']} | {r['params_m']:.2f}M | {r['flops_g']:.2f}G | {r['gpu_b1_ms']:.1f} ms | "
              f"{r['gpu_b1_nms_ms']:.1f} ms | {r['gpu_b8_img_per_s']:.0f} | {r['cpu_b1_ms']:.1f} ms")

    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "latency_five_2026-11-20.json"
    out.write_text(json.dumps({"preprocess_ms": pre_ms, "rows": rows, "gpu": torch.cuda.get_device_name(0),
                               "torch": torch.__version__, "cpu_threads": torch.get_num_threads(),
                               "seconds": round(time.time() - t_all)}, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print(f"\n已保存 {out}（总耗时 {time.time() - t_all:.0f}s）")


if __name__ == "__main__":
    main()
