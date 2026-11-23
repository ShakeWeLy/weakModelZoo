"""Backbone / Neck / Detect / 整网：CUDA Graph vs eager（GPU，batch 1，640×640，FP32，无后处理）。

目的：去掉 eager 算子发射开销后，得到各模块在 GPU 侧的真实耗时分布。
不根据旧的 eager 模块表做替换决策。

方法：
1. 一次整网前向，捕获 backbone / neck / detect 及其直接子模块的真实输入；
2. 对每个目标（whole / backbone / neck / detect / backbone.* / neck.*）：
   - eager 逐次同步（CUDA Event 中位数）
   - CUDA Graph 回放（CUDA Event 中位数）——接近纯 GPU 执行时间
3. 报告 Graph 占比、与 YOLOv5s 的倍数，并给出子模块 Graph 表。

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


def _freeze_args(inp) -> tuple:
    """把 hook 捕获的输入做成地址稳定的 contiguous 缓冲，供 Graph capture 使用。"""
    out = []
    for a in inp:
        if torch.is_tensor(a):
            out.append(a.detach().contiguous())
        elif isinstance(a, (list, tuple)):
            out.append(type(a)(t.detach().contiguous() if torch.is_tensor(t) else t for t in a))
        else:
            out.append(a)
    return tuple(out)


def _make_fn(mod, args):
    """闭包：用固定 args 调用模块。Detect 等接收 list 的模块也适用。"""
    return lambda m=mod, a=args: m(*a)


def capture_io(model, x):
    """一次前向，拿到 whole / backbone / neck / detect 及其直接子模块的真实输入。"""
    child_names = [f"backbone.{n}" for n, _ in model.backbone.named_children()] + [
        f"neck.{n}" for n, _ in model.neck.named_children()
    ]
    want = {"backbone", "neck", "detect", *child_names}
    captured: dict[str, tuple] = {}
    hs = []
    mods = dict(model.named_modules())
    for name in want:
        def pre(m, inp, name=name):
            captured[name] = tuple(inp)

        hs.append(mods[name].register_forward_pre_hook(pre))
    with torch.no_grad():
        model(x)
    for h in hs:
        h.remove()

    frozen = {n: _freeze_args(captured[n]) for n in want if n in captured}
    targets: dict[str, object] = {"whole": lambda: model(x)}
    for n, args in frozen.items():
        targets[n] = _make_fn(mods[n], args)

    shapes = {
        n: [
            list(a.shape) if torch.is_tensor(a) else
            [list(t.shape) if torch.is_tensor(t) else str(type(t)) for t in a]
            for a in args
        ]
        for n, args in frozen.items()
    }
    return targets, shapes, child_names


def main() -> None:
    t0 = time.time()
    print(f"GPU：{torch.cuda.get_device_name(0)}；torch {torch.__version__}", flush=True)
    device = torch.device("cuda")
    x = torch.rand(1, 3, IMG, IMG, device=device)
    out: dict = {}

    for label, cfg in MODELS.items():
        print(f"\n== {label}", flush=True)
        model = ra.load_model({"model": cfg["model"], "run": cfg["run"]}).eval().to(device)
        targets, shapes, child_names = capture_io(model, x)
        row: dict = {"shapes": shapes, "modules": {}, "children": child_names}
        # 先 top，再子模块；打印时对齐到最长名字
        order = ["whole", "backbone", "neck", "detect", *child_names]
        w = max(len(n) for n in order)
        for name in order:
            fn = targets[name]
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
                f"   {name:<{w}} eager {e_med:7.3f} ms (P90 {e_p90:.3f}) | graph {g_str}",
                flush=True,
            )

        # Graph 占比（相对整网 Graph）
        gm = row["modules"]
        if gm["whole"]["graph_ms"] and all(gm[k]["graph_ms"] for k in ("backbone", "neck", "detect")):
            tw = gm["whole"]["graph_ms"]
            for k in ("backbone", "neck", "detect", *child_names):
                if gm[k]["graph_ms"] is not None:
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

    # 跨模型对比表（top + 公共子模块）
    if all(k in out for k in MODELS):
        yolo_c = set(out["YOLOv5s"]["children"])
        sgc_c = set(out["SGC 无注意力"]["children"])
        common = [n for n in out["YOLOv5s"]["children"] if n in sgc_c]
        only_sgc = [n for n in out["SGC 无注意力"]["children"] if n not in yolo_c]
        print("\n== 跨模型 Graph 对比（ms）", flush=True)
        print(f"   {'模块':<22} {'YOLOv5s':>10} {'SGC 无注意力':>12} {'倍数':>8} {'差值':>8}", flush=True)
        for name in ("whole", "backbone", "neck", "detect", *common):
            a = out["YOLOv5s"]["modules"][name]["graph_ms"]
            b = out["SGC 无注意力"]["modules"][name]["graph_ms"]
            if a is None or b is None:
                continue
            print(f"   {name:<22} {a:10.3f} {b:12.3f} {b / a:8.2f} {b - a:8.3f}", flush=True)
        for name in only_sgc:
            b = out["SGC 无注意力"]["modules"][name]["graph_ms"]
            if b is None:
                continue
            print(f"   {name:<22} {'—':>10} {b:12.3f} {'—':>8} {'—':>8}", flush=True)

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
