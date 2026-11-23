"""Backbone / Neck / Detect / 整网：CUDA Graph vs eager（GPU，batch 1，640×640，FP32，无后处理）。

目的：去掉 eager 算子发射开销后，得到各模块在 GPU 侧的真实耗时分布。
不根据旧的 eager 模块表做替换决策。

方法：
1. 一次整网前向，捕获 backbone / neck / detect 的真实输入；
2. 对每个目标（whole / backbone / neck / detect）：
   - eager 逐次同步（CUDA Event 中位数）
   - CUDA Graph 回放（CUDA Event 中位数）——接近纯 GPU 执行时间
3. 报告 Graph 占比、与 YOLOv5s 的倍数。

注意：
- 模块 Graph 时间之和不必等于整网 Graph（独立 capture、独立回放，边界不同）。
- 本脚本不做 BN 折叠；与 `cuda层分析.md` 中 BN 折叠后的整网 Graph 数字不可直接混比。

用法（仓库根目录）：
  python experiments/detection/project2_2611/no_train_analysis/graph_module_timing.py
输出：outputs/module_graph/module_graph_YYYY-MM-DD.json
"""

from __future__ import annotations

import json
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import torch

_PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJ))
sys.path.insert(0, str(_PROJ / "analysis_scripts"))
import redundancy_analysis as ra  # noqa: E402

OUT = Path(__file__).resolve().parent / "outputs" / "module_graph"
IMG = 640
WARM = 30
EAGER_N = 150
GRAPH_N = 300
MODELS = {
    "YOLOv5s": {"model": "yolov5s", "run": "2026-10-30_002_YOLOv5s_V3"},
    "SGC 无注意力": {
        "model": "sgc_yolov5s_c3_sgdown_gbot_dw5",
        "run": "2026-11-17_015_SGC_c3_sgdown_gbot_dw5_s_noDFC_cbamNone_seed42_V1",
    },
}


def med(v: list[float]) -> float:
    return float(np.median(v))


def p90(v: list[float]) -> float:
    return float(np.percentile(v, 90))


def eager_ms(fn, n: int = EAGER_N) -> tuple[float, float]:
    with torch.no_grad():
        for _ in range(WARM):
            fn()
    torch.cuda.synchronize()
    ts: list[float] = []
    with torch.no_grad():
        for _ in range(n):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            fn()
            e.record()
            torch.cuda.synchronize()
            ts.append(s.elapsed_time(e))
    return med(ts), p90(ts)


def graph_ms(fn, n: int = GRAPH_N) -> tuple[float, float] | None:
    """Capture fn into a CUDA Graph and time replay. Returns None on failure."""
    try:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s), torch.no_grad():
            for _ in range(5):
                fn()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.no_grad(), torch.cuda.graph(g):
            fn()
        for _ in range(WARM):
            g.replay()
        torch.cuda.synchronize()
        ts: list[float] = []
        for _ in range(n):
            a = torch.cuda.Event(enable_timing=True)
            b = torch.cuda.Event(enable_timing=True)
            a.record()
            g.replay()
            b.record()
            torch.cuda.synchronize()
            ts.append(a.elapsed_time(b))
        return med(ts), p90(ts)
    except Exception as ex:  # noqa: BLE001
        print(f"     [graph 失败] {type(ex).__name__}: {ex}", flush=True)
        return None


def capture_io(model, x):
    """一次前向，拿到 backbone/neck/detect 的真实输入与可调用 wrapper。"""
    bb_in = [x]
    nk_in: list = []
    dt_in: list = []
    hs = []

    def pre_bb(m, inp):
        bb_in[0] = inp[0].detach()

    def pre_nk(m, inp):
        nk_in.clear()
        nk_in.extend(t.detach() for t in inp)

    def pre_dt(m, inp):
        # Detect 接收 list[Tensor]
        dt_in.clear()
        arg = inp[0]
        if isinstance(arg, (list, tuple)):
            dt_in.extend(t.detach() for t in arg)
        else:
            dt_in.append(arg.detach())

    hs.append(model.backbone.register_forward_pre_hook(pre_bb))
    hs.append(model.neck.register_forward_pre_hook(pre_nk))
    hs.append(model.detect.register_forward_pre_hook(pre_dt))
    with torch.no_grad():
        model(x)
    for h in hs:
        h.remove()

    # 静态输入缓冲（Graph capture 要求地址稳定；对子模块用 clone 出的固定缓冲）
    x_bb = bb_in[0].contiguous()
    x_nk = tuple(t.contiguous() for t in nk_in)
    x_dt = [t.contiguous() for t in dt_in]

    targets = {
        "whole": lambda: model(x),
        "backbone": lambda: model.backbone(x_bb),
        "neck": lambda: model.neck(*x_nk),
        "detect": lambda: model.detect(x_dt),
    }
    shapes = {
        "backbone_in": list(x_bb.shape),
        "neck_in": [list(t.shape) for t in x_nk],
        "detect_in": [list(t.shape) for t in x_dt],
    }
    return targets, shapes


def main() -> None:
    t0 = time.time()
    print(f"GPU：{torch.cuda.get_device_name(0)}；torch {torch.__version__}", flush=True)
    device = torch.device("cuda")
    x = torch.rand(1, 3, IMG, IMG, device=device)
    out: dict = {}

    for label, cfg in MODELS.items():
        print(f"\n== {label}", flush=True)
        model = ra.load_model({"model": cfg["model"], "run": cfg["run"]}).eval().to(device)
        targets, shapes = capture_io(model, x)
        row: dict = {"shapes": shapes, "modules": {}}
        for name, fn in targets.items():
            e_med, e_p90 = eager_ms(fn)
            g = graph_ms(fn)
            entry = {
                "eager_ms": e_med,
                "eager_p90_ms": e_p90,
                "graph_ms": None if g is None else g[0],
                "graph_p90_ms": None if g is None else g[1],
            }
            if g is not None and e_med > 0:
                entry["graph_over_eager"] = g[0] / e_med
            row["modules"][name] = entry
            g_str = "失败" if g is None else f"{g[0]:.3f} ms (P90 {g[1]:.3f})"
            print(
                f"   {name:<10} eager {e_med:7.3f} ms (P90 {e_p90:.3f}) | graph {g_str}",
                flush=True,
            )

        # Graph 占比（相对整网 Graph）
        gm = row["modules"]
        if gm["whole"]["graph_ms"] and all(gm[k]["graph_ms"] for k in ("backbone", "neck", "detect")):
            tw = gm["whole"]["graph_ms"]
            for k in ("backbone", "neck", "detect"):
                gm[k]["graph_share_of_whole"] = gm[k]["graph_ms"] / tw
            ssum = sum(gm[k]["graph_ms"] for k in ("backbone", "neck", "detect"))
            row["graph_sum_modules_ms"] = ssum
            row["graph_sum_over_whole"] = ssum / tw
            print(
                f"   [Graph 占比] backbone {100 * gm['backbone']['graph_share_of_whole']:.1f}% | "
                f"neck {100 * gm['neck']['graph_share_of_whole']:.1f}% | "
                f"detect {100 * gm['detect']['graph_share_of_whole']:.1f}% | "
                f"三者之和/整网 = {ssum / tw:.2f}",
                flush=True,
            )
        out[label] = row
        del model
        torch.cuda.empty_cache()

    # 跨模型对比表
    if all(k in out for k in MODELS):
        print("\n== 跨模型 Graph 对比（ms）", flush=True)
        print(f"   {'模块':<10} {'YOLOv5s':>10} {'SGC 无注意力':>12} {'倍数':>8} {'差值':>8}", flush=True)
        for name in ("whole", "backbone", "neck", "detect"):
            a = out["YOLOv5s"]["modules"][name]["graph_ms"]
            b = out["SGC 无注意力"]["modules"][name]["graph_ms"]
            if a is None or b is None:
                continue
            print(f"   {name:<10} {a:10.3f} {b:12.3f} {b / a:8.2f} {b - a:8.3f}", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    stamp = date.today().isoformat()
    path = OUT / f"module_graph_{stamp}.json"
    path.write_text(
        json.dumps(
            {
                "date": stamp,
                "iters": {"eager": EAGER_N, "graph": GRAPH_N, "warmup": WARM},
                "note": "graph≈GPU 执行；eager 含 CPU 发射。模块 Graph 之和不必等于整网。",
                "results": out,
                "seconds": round(time.time() - t0),
            },
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )
    print(f"\n已保存 {path}（总耗时 {time.time() - t0:.0f}s）", flush=True)


if __name__ == "__main__":
    main()
