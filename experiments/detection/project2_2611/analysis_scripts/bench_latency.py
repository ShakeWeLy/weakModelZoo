"""推理延迟基准（batch=1，640×640，随机输入；不训练、不改权重）。

FLOPs 不等于速度，这里直接测 CPU（以及有 GPU 时的 GPU）上的中位延迟与 FPS。
候选变体（proj_type=ghost）用同一份配置随机初始化构建，只用于测速，不代表精度。

用法（仓库根目录）：
    python experiments/detection/project2_2611/bench_latency.py
输出：bench_latency/latency.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import redundancy_analysis as ra  # noqa: E402
from diagnose_p4_cbam import MODELS  # noqa: E402
from trainer import RunRecorder, build_config, build_model, model_output_dir  # noqa: E402

OUT = Path(__file__).resolve().parent / "bench_latency"
IMG = 640


def ghost_proj_variant() -> torch.nn.Module:
    """本模型的配置，仅把 CSP 投影由稠密 1×1（conv）改为 GBS（ghost），随机初始化。"""
    spec = MODELS["ours"]
    run_dir = model_output_dir(build_config(spec["model"])) / spec["run"]
    cfg = RunRecorder.open(run_dir).load_config()
    if "params" not in cfg["model"]:
        raise SystemExit("保存的配置里没有 model.params，无法修改 proj_type")
    cfg["model"]["params"]["proj_type"] = "ghost"
    return build_model(cfg, 1).eval()


def bench(model: torch.nn.Module, device: torch.device, iters: int, warmup: int) -> dict:
    model = model.eval().to(device)
    x = torch.rand(1, 3, IMG, IMG, device=device)
    sync = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)
    ts = []
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        sync()
        for _ in range(iters):
            t0 = time.perf_counter()
            model(x)
            sync()
            ts.append(time.perf_counter() - t0)
    med = float(np.median(ts)) * 1000
    return {"median_ms": med, "fps": 1000.0 / med, "p90_ms": float(np.percentile(ts, 90)) * 1000}


def main() -> None:
    builders = {
        "YOLOv5s（40 ep，V3）": lambda: ra.load_model(MODELS["yolo"]),
        "本模型（40 ep，V1）": lambda: ra.load_model(MODELS["ours"]),
        "本模型 去CBAM（40 ep，s42）": lambda: ra.load_model(
            {"model": "sgc_yolov5s_c3_sgdown_gbot_dw5_cbam_none", "run": "2026-11-17_005_SGC_c3_sgdown_gbot_dw5_s_cbamNone_V1"}),
        "本模型 P3去SAM（40 ep，s42）": lambda: ra.load_model(
            {"model": "sgc_yolov5s_c3_sgdown_gbot_dw5_cbam_cam_p3", "run": "2026-11-17_006_SGC_c3_sgdown_gbot_dw5_s_cbamCamP3_V1"}),
        "本模型 去DFC（40 ep，s42）": lambda: ra.load_model(
            {"model": "sgc_yolov5s_c3_sgdown_gbot_dw5_nodfc", "run": "2026-11-17_007_SGC_c3_sgdown_gbot_dw5_s_noDFC_V1"}),
        "本模型 残差恒等（40 ep，s42）": lambda: ra.load_model(
            {"model": "sgc_yolov5s_c3_sgdown_gbot_dw5_resid_identity", "run": "2026-11-17_008_SGC_c3_sgdown_gbot_dw5_s_residIdentity_V1"}),
        "c3topo_1（Conv↓ + 稠密瓶颈，40 ep）": lambda: ra.load_model(MODELS["c3topo"]),
        "c3_sgdown（SGhost↓ + 稠密瓶颈，40 ep）": lambda: ra.load_model(
            {"model": "sgc_yolov5s_c3_sgdown", "run": "2026-11-12_024_SGC_c3_sgdown_s_V1"}),
        "c3_ghostbot_dw7（Conv↓ + Ghost 瓶颈，40 ep）": lambda: ra.load_model(
            {"model": "sgc_yolov5s_c3_ghostbot_dw7", "run": "2026-11-13_028_SGC_c3_ghostbot_dw7_s_V1"}),
        "本模型 + proj=ghost（随机初始化，仅测速）": ghost_proj_variant,
    }
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
    print(f"CPU 线程数 {torch.get_num_threads()}；GPU：{torch.cuda.get_device_name(0) if torch.cuda.is_available() else '无'}")

    results = []
    for name, build in builders.items():
        model = build()
        params_m = sum(p.numel() for p in model.parameters()) / 1e6
        row = {"model": name, "params_m": params_m}
        for dev in devices:
            iters, warmup = (30, 5) if dev.type == "cpu" else (200, 30)
            row[dev.type] = bench(model, dev, iters, warmup)
        results.append(row)
        model.to("cpu")
        print(f"{name:<40} 参数 {params_m:.2f} M  " + "  ".join(
            f"{d.type.upper()} {row[d.type]['median_ms']:.1f} ms（{row[d.type]['fps']:.1f} FPS）" for d in devices),
            flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "latency.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"已保存 {OUT / 'latency.json'}")


if __name__ == "__main__":
    main()
