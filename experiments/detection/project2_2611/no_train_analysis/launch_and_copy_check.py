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
# ProfilerActivity：采集 CPU / CUDA 活动；profile：采集上下文管理器
from torch.profiler import ProfilerActivity, profile

# 仓库根目录（本文件位于 .../no_train_analysis/ 下，parents[1] 即项目根）
_PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJ))
sys.path.insert(0, str(_PROJ / "analysis_scripts"))
import redundancy_analysis as ra  # noqa: E402

OUT = Path(__file__).resolve().parent / "outputs" / "launch_copy"
IMG = 640
WARM = 20          # 预热次数（让频率、缓存、cuDNN 算法进入稳定状态）
EAGER_N = 100      # eager 逐次同步计时的迭代次数
PIPE_N = 100       # 流水线吞吐计时的迭代次数
GRAPH_N = 200      # CUDA Graph 回放计时迭代次数
# 视为"copy 类"的 aten 顶层算子名
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
    """中位数，抵抗偶发抖动（其他进程占 GPU、时钟降频等）。"""
    return float(np.median(v))


def eager_sync(model, x) -> float:
    """eager 逐次同步墙钟：每次前向单独用 CUDA Event 计时并 synchronize。

    特点：每次前向后 CPU 都等待 GPU 完成，CPU 发射与 GPU 执行串行化，
    因此测到的是「单次前向的端到端延迟」，包含 GPU kernel 时间 + CPU 发射开销
    + 每次同步的固定开销。与 module_timing.py 的口径一致。
    """
    ts = []
    with torch.no_grad():
        for _ in range(EAGER_N):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()                  # 在默认流插入起始时间戳
            model(x)                    # 异步提交前向 kernel，CPU 立即返回
            e.record()                  # 在流上插入结束时间戳（排在前向 kernel 之后）
            torch.cuda.synchronize()    # 等待本轮到 e 为止全部完成
            ts.append(s.elapsed_time(e))   # 毫秒
    return med(ts)


def eager_pipe(model, x) -> float:
    """eager 流水线吞吐：连续发射 PIPE_N 次前向，只在最后同步一次，取总时间/N。

    特点：CPU 不必每次等待 GPU，发射与执行可重叠。若 GPU 受限，结果 ≈ eager_sync；
    若 CPU 发射受限，流水线吞吐会明显低于单次延迟（因为 CPU 成为瓶颈，GPU 有空隙）。
    用 time.perf_counter() 测墙钟（包含 CPU 发射时间），除以次数得到「每次前向摊薄时间」。
    """
    torch.cuda.synchronize()           # 清空流，计时边界干净
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(PIPE_N):
            model(x)                   # 连续异步提交，不等待
    torch.cuda.synchronize()           # 等所有 PIPE_N 次前向执行完
    return (time.perf_counter() - t0) * 1000.0 / PIPE_N   # 转毫秒/次


def graph_check(model, x) -> tuple[float, float] | None:
    """CUDA Graph 回放：把整网前向捕获成一张图，回放时跳过 Python/算子发射开销。

    返回 (逐次同步回放中位数 ms, 流水线回放 ms/次)；失败返回 None。
    CUDA Graph 把一系列 kernel 及其依赖固化，回放只需一次 API 调用，
    因此测到的是「纯 GPU 执行时间」（几乎不含 CPU 发射开销），
    与 eager 对比即可判断整网是 GPU 受限还是 CPU launch 受限。
    """
    try:
        # ---- 捕获前的准备：在独立 stream 上预热，避免捕获到分配器等一次性操作 ----
        s = torch.cuda.Stream()
        # 让新 stream 等待当前 stream 的已有工作，保证依赖顺序正确
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s), torch.no_grad():
            for _ in range(5):         # 在侧流上预热，触发 cuDNN/缓存分配
                model(x)
        # 让当前 stream 等待侧流完成，之后在默认 stream 上捕获
        torch.cuda.current_stream().wait_stream(s)

        # ---- 捕获：进入 torch.cuda.graph 上下文后，所有 GPU 操作被记录而非执行 ----
        g = torch.cuda.CUDAGraph()
        with torch.no_grad(), torch.cuda.graph(g):
            model(x)                   # 此次前向被"录制"进图，不真正执行

        # ---- 预热回放：让图回放进入稳定状态 ----
        for _ in range(WARM):
            g.replay()                 # 回放整张图（一次 API 调用触发所有 kernel）
        torch.cuda.synchronize()

        # ---- 逐次同步计时：每次回放后用 Event 计时并同步 ----
        ts = []
        for _ in range(GRAPH_N):
            a = torch.cuda.Event(enable_timing=True)
            b = torch.cuda.Event(enable_timing=True)
            a.record()
            g.replay()
            b.record()
            torch.cuda.synchronize()
            ts.append(a.elapsed_time(b))

        # ---- 流水线吞吐：连续回放 PIPE_N 次，最后统一同步 ----
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(PIPE_N):
            g.replay()
        torch.cuda.synchronize()
        pipe = (time.perf_counter() - t0) * 1000.0 / PIPE_N
        return med(ts), pipe
    except Exception as ex:  # noqa: BLE001  （某些模型/算子不支持 graph 捕获）
        print(f"   [graph] 失败：{type(ex).__name__}: {ex}", flush=True)
        return None


def cpu_vs_device(model, x) -> tuple[float, float]:
    """用 profiler 汇总每次前向的 self CPU 时间与 self device 时间（毫秒）。

    - self_cpu_time_total：算子自身在 CPU 上的耗时（含算子分发、内核参数准备等），
      反映 CPU 发射侧的工作量。
    - self_device_time_total：算子自身在设备（GPU）上的耗时，反映 GPU 执行侧。
    两者对比可判断瓶颈在 CPU 发射还是 GPU 执行。
    """
    with torch.no_grad():
        for _ in range(WARM):
            model(x)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        with torch.no_grad():
            for _ in range(10):
                model(x)
        torch.cuda.synchronize()       # 确保所有 CUDA kernel 被记录
    cpu = dev = 0.0
    for e in prof.events():            # 原始事件树，逐事件累加 self 时间
        cpu += float(getattr(e, "self_cpu_time_total", 0) or 0)
        dev += float(getattr(e, "self_device_time_total", 0) or 0)
    return cpu / 10 / 1000.0, dev / 10 / 1000.0   # 微秒 -> 毫秒/次


def top_level(events, keys):
    """从事件树中挑出"顶层"目标算子，去掉嵌套（如 contiguous→clone→copy_）。

    做法：沿 cpu_parent 向上找，若祖先也在 keys 中，则当前事件是嵌套的、跳过。
    """
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
    """Part 2：按模块类型统计每个实例前向中的顶层 copy 次数与顶层 cat 次数。

    不依赖调用栈：先用 hook 捕获每个实例在真实整网前向中的输入，
    再对每个实例单独前向并 profile，统计顶层 copy / cat 的个数。
    用"顶层"口径避免 contiguous→clone→copy_ 这类嵌套重复计数。
    """
    captured: list = []                # (模块名, 类型名, 模块对象, 输入 tuple)
    mods = list(model.named_modules())
    hs = []
    for name, m in mods:
        if type(m).__name__ in types:
            # 默认参数 name=name 绑定当前名称，避免闭包延迟绑定
            def pre(mod, inp, name=name):
                captured.append((name, type(mod).__name__, mod, inp))
            hs.append(m.register_forward_pre_hook(pre))
    with torch.no_grad():
        model(x)                       # 跑一次整网，触发所有 hook
    for h in hs:
        h.remove()                     # 移除 hook，避免影响后续 profile

    res = {}
    for T in types:
        insts = [(n, m, i) for n, t, m, i in captured if t == T]
        if not insts:
            continue
        copies = 0.0
        cats = 0.0
        for n, m, i in insts:
            # 先热身 3 次：触发该实例内部的 cuDNN 算法选择、缓存分配
            with torch.no_grad():
                for _ in range(3):
                    m(*i)
            torch.cuda.synchronize()
            # 只采集 CPU 活动即可统计算子调用次数（不需要 CUDA kernel 时间）
            with profile(activities=[ProfilerActivity.CPU]) as prof:
                with torch.no_grad():
                    m(*i)
                torch.cuda.synchronize()
            evs = prof.events()
            copies += len(top_level(evs, COPY_KEYS))              # 顶层 copy 个数
            cats += sum(1 for e in top_level(evs, {"aten::cat"}))  # 顶层 cat 个数
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
        # 加载模型 -> eval() 推理模式 -> 搬到 GPU
        model = ra.load_model({"model": cfg["model"], "run": cfg["run"]}).eval().to(device)
        r = {}
        # part 1：四种口径对比，判断 GPU 受限 vs CPU launch 受限
        r["eager_sync_ms"] = eager_sync(model, x)       # 逐次同步延迟
        r["eager_pipe_ms"] = eager_pipe(model, x)       # 流水线吞吐
        g = graph_check(model, x)                       # CUDA Graph 纯 GPU 时间
        if g is not None:
            r["graph_replay_sync_ms"], r["graph_pipe_ms"] = g
        cpu, dev = cpu_vs_device(model, x)              # profiler CPU/GPU self 时间
        r["profiler_self_cpu_ms"] = cpu
        r["profiler_self_device_ms"] = dev
        print(f"   eager 逐次同步 {r['eager_sync_ms']:.2f} ms | eager 流水线 {r['eager_pipe_ms']:.2f} ms | "
              f"Graph 回放 {r.get('graph_replay_sync_ms', float('nan')):.2f} ms | "
              f"Graph 流水线 {r.get('graph_pipe_ms', float('nan')):.2f} ms", flush=True)
        print(f"   profiler self_cpu 和 {cpu:.2f} ms/前向 | self_device 和 {dev:.2f} ms/前向", flush=True)
        # part 2：按模块类型统计顶层 copy / cat
        r["per_type_copies"] = per_type_copies(model, x, cfg["types"])
        out[label] = r
        # 删除模型并清空缓存，避免模型间显存干扰
        del model
        torch.cuda.empty_cache()

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "launch_copy_2026-11-20.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n已保存 {OUT / 'launch_copy_2026-11-20.json'}（总耗时 {time.time() - t0:.0f}s）", flush=True)


if __name__ == "__main__":
    main()