"""检验两件事（GPU，batch 1，640×640，eager，无需训练）。

part 1：整网是 GPU 受限还是 CPU 发射（launch）受限
  - eager 逐次同步墙钟（CUDA Event 中位数）：与 module_timing 口径一致
  - eager 流水线吞吐：连续发射 N 次前向后统一同步，取 总时间/N
  - CUDA Graph 回放：去掉 Python/算子发射开销后的纯 GPU 执行时间（中位数，及流水线吞吐）
  - profiler 中 self_cpu_time 之和与 self_device_time 之和的对比

part 2：copy 归因（不依赖调用栈）
  - 对每个模块类型的每个实例，用其真实输入单独前向，统计"顶层" copy 次数（去掉 contiguous→clone→copy_ 的嵌套）
  - 同时统计顶层 cat 次数，与 Copy 对照

用法（仓库根目录）：python experiments/detection/project2_2611/no_train_analysis/launch_and_copy_check.py
输出：outputs/launch_copy/launch_copy_2026-11-20.json
"""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile

_PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJ))
sys.path.insert(0, str(_PROJ / "analysis_scripts"))
import redundancy_analysis as ra  # noqa: E402

OUT = Path(__file__).resolve().parent / "outputs" / "launch_copy"
IMG = 640
WARM = 20
EAGER_N = 100
PIPE_N = 100
GRAPH_N = 200
COPY_KEYS = {"aten::copy_", "aten::contiguous", "aten::clone", "aten::_to_copy"}
MODELS = {
    "YOLOv5s": {"model": "yolov5s", "run": "2026-10-30_002_YOLOv5s_V3",
                "types": ["BaseConv", "C3", "Bottleneck", "SPPF", "Detect"]},
    "SGC 无注意力": {"model": "sgc_yolov5s_c3_sgdown_gbot_dw5",
                     "run": "2026-11-17_015_SGC_c3_sgdown_gbot_dw5_s_noDFC_cbamNone_seed42_V1",
                     "types": ["SGhostConv", "GSConv", "GSBottleneck", "VOVGCSPYolo", "SDGhost", "Bottleneck",
                               "Detect"]},
}


def med(v):
    return float(np.median(v))


def eager_sync(model, x) -> float:
    ts = []
    with torch.no_grad():
        for _ in range(EAGER_N):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            model(x)
            e.record()
            torch.cuda.synchronize()
            ts.append(s.elapsed_time(e))
    return med(ts)


def eager_pipe(model, x) -> float:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(PIPE_N):
            model(x)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / PIPE_N


def graph_check(model, x) -> tuple[float, float] | None:
    try:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s), torch.no_grad():
            for _ in range(5):
                model(x)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.no_grad(), torch.cuda.graph(g):
            model(x)
        for _ in range(WARM):
            g.replay()
        torch.cuda.synchronize()
        ts = []
        for _ in range(GRAPH_N):
            a = torch.cuda.Event(enable_timing=True)
            b = torch.cuda.Event(enable_timing=True)
            a.record()
            g.replay()
            b.record()
            torch.cuda.synchronize()
            ts.append(a.elapsed_time(b))
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(PIPE_N):
            g.replay()
        torch.cuda.synchronize()
        pipe = (time.perf_counter() - t0) * 1000.0 / PIPE_N
        return med(ts), pipe
    except Exception as ex:  # noqa: BLE001
        print(f"   [graph] 失败：{type(ex).__name__}: {ex}", flush=True)
        return None


def cpu_vs_device(model, x) -> tuple[float, float]:
    with torch.no_grad():
        for _ in range(WARM):
            model(x)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        with torch.no_grad():
            for _ in range(10):
                model(x)
        torch.cuda.synchronize()
    cpu = dev = 0.0
    for e in prof.events():
        cpu += float(getattr(e, "self_cpu_time_total", 0) or 0)
        dev += float(getattr(e, "self_device_time_total", 0) or 0)
    return cpu / 10 / 1000.0, dev / 10 / 1000.0


def top_level(events, keys):
    out = []
    for e in events:
        if e.key not in keys:
            continue
        p = e.cpu_parent
        nested = False
        while p is not None:
            if p.key in keys:
                nested = True
                break
            p = p.cpu_parent
        if not nested:
            out.append(e)
    return out


def per_type_copies(model, x, types) -> dict:
    captured: list = []
    mods = list(model.named_modules())
    hs = []
    for name, m in mods:
        if type(m).__name__ in types:
            def pre(mod, inp, name=name):
                captured.append((name, type(mod).__name__, mod, inp))
            hs.append(m.register_forward_pre_hook(pre))
    with torch.no_grad():
        model(x)
    for h in hs:
        h.remove()
    res = {}
    for T in types:
        insts = [(n, m, i) for n, t, m, i in captured if t == T]
        if not insts:
            continue
        copies = 0.0
        cats = 0.0
        for n, m, i in insts:
            with torch.no_grad():
                for _ in range(3):
                    m(*i)
            torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CPU]) as prof:
                with torch.no_grad():
                    m(*i)
                torch.cuda.synchronize()
            evs = prof.events()
            copies += len(top_level(evs, COPY_KEYS))
            cats += sum(1 for e in top_level(evs, {"aten::cat"}))
        res[T] = {"instances": len(insts), "top_copy_per_instance": copies / len(insts),
                  "top_cat_per_instance": cats / len(insts)}
        print(f"   [{T}] 实例 {len(insts)}：顶层 copy/实例 {copies / len(insts):.2f}，顶层 cat/实例 {cats / len(insts):.2f}",
              flush=True)
    return res


def main() -> None:
    t0 = time.time()
    device = torch.device("cuda")
    x = torch.rand(1, 3, IMG, IMG, device=device)
    out = {}
    for label, cfg in MODELS.items():
        print(f"\n== {label}", flush=True)
        model = ra.load_model({"model": cfg["model"], "run": cfg["run"]}).eval().to(device)
        r = {}
        r["eager_sync_ms"] = eager_sync(model, x)
        r["eager_pipe_ms"] = eager_pipe(model, x)
        g = graph_check(model, x)
        if g is not None:
            r["graph_replay_sync_ms"], r["graph_pipe_ms"] = g
        cpu, dev = cpu_vs_device(model, x)
        r["profiler_self_cpu_ms"] = cpu
        r["profiler_self_device_ms"] = dev
        print(f"   eager 逐次同步 {r['eager_sync_ms']:.2f} ms | eager 流水线 {r['eager_pipe_ms']:.2f} ms | "
              f"Graph 回放 {r.get('graph_replay_sync_ms', float('nan')):.2f} ms | "
              f"Graph 流水线 {r.get('graph_pipe_ms', float('nan')):.2f} ms", flush=True)
        print(f"   profiler self_cpu 和 {cpu:.2f} ms/前向 | self_device 和 {dev:.2f} ms/前向", flush=True)
        r["per_type_copies"] = per_type_copies(model, x, cfg["types"])
        out[label] = r
        del model
        torch.cuda.empty_cache()

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "launch_copy_2026-11-20.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n已保存 {OUT / 'launch_copy_2026-11-20.json'}（总耗时 {time.time() - t0:.0f}s）", flush=True)


if __name__ == "__main__":
    main()
