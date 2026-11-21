"""在已保存的 TRT 引擎上检查两件事（不重新构建引擎）：
1. 默认流 vs 独立 CUDA 流执行的延迟（TRT 对默认流会额外同步）；
2. SGC FP16 引擎的逐层耗时（IProfiler），用于定位剩余耗时。

用法（仓库根目录）：python experiments/detection/project2_2611/no_train_analysis/trt_stream_layer_check.py
输出：outputs/trt/trt_stream_layer_2026-11-21.json
"""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import tensorrt as trt
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from trt_deploy_bench import TRTRunner  # noqa: E402

OUT = _HERE / "outputs" / "trt"
N = 300


def median_ms_on(stream: torch.cuda.Stream | None, fn) -> float:
    ts = []
    for _ in range(N):
        s_ = torch.cuda.Event(enable_timing=True)
        e_ = torch.cuda.Event(enable_timing=True)
        st = stream if stream is not None else torch.cuda.current_stream()
        with torch.cuda.stream(st):
            s_.record(st)
            fn()
            e_.record(st)
        st.synchronize()
        ts.append(s_.elapsed_time(e_))
    return float(np.median(ts))


def pipe_ms_on(stream: torch.cuda.Stream | None, fn) -> float:
    st = stream if stream is not None else torch.cuda.current_stream()
    st.synchronize()
    t0 = time.perf_counter()
    with torch.cuda.stream(st):
        for _ in range(N):
            fn()
    st.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / N


def categorize(name: str) -> str:
    """按 TRT 层名中的关键词归类（TRT 会把相邻算子融合成一层，名字中保留了原始模块路径）。"""
    n = name
    if "Reformatting" in n or "CopyNode" in n:
        return "格式转换/拷贝（Reformatting）"
    if "shuffle" in n:
        return "ChannelShuffle（Reshape/Transpose）"
    if "Conv" in n:
        return "卷积（标准/深度/分组，TRT 名称不区分）"
    if "Sigmoid" in n or "PWN" in n:
        return "逐元素（SiLU/Mul/Add 等，PWN 融合）"
    if "Concat" in n or "cat" in n.lower():
        return "Concat"
    if "Resize" in n or "Upsample" in n:
        return "上采样"
    if "Pool" in n:
        return "池化"
    return "其它"


class LayerProf(trt.IProfiler):
    def __init__(self) -> None:
        super().__init__()
        self.t: dict[str, float] = defaultdict(float)
        self.cnt: dict[str, int] = defaultdict(int)

    def report_layer_time(self, layer_name: str, ms: float) -> None:
        self.t[layer_name] += ms
        self.cnt[layer_name] += 1


def main() -> None:
    result: dict = {}
    side = torch.cuda.Stream()
    for name in ["sgc_fp16", "sgc_fp32", "yolov5s_fp16", "yolov5s_fp32"]:
        plan = (OUT / f"{name}.engine").read_bytes()
        runner = TRTRunner(plan)
        runner.set_input(torch.rand(1, 3, 640, 640, device="cuda"))
        for _ in range(30):
            runner.run()
        torch.cuda.synchronize()
        r = {
            "default_stream_sync_ms": median_ms_on(None, runner.run),
            "default_stream_pipe_ms": pipe_ms_on(None, runner.run),
            "side_stream_sync_ms": median_ms_on(side, runner.run),
            "side_stream_pipe_ms": pipe_ms_on(side, runner.run),
        }
        result[name] = r
        print(f"{name:<14} 默认流：逐次 {r['default_stream_sync_ms']:.2f} / 流水线 {r['default_stream_pipe_ms']:.2f} ms"
              f" | 独立流：逐次 {r['side_stream_sync_ms']:.2f} / 流水线 {r['side_stream_pipe_ms']:.2f} ms", flush=True)
        del runner
        torch.cuda.empty_cache()

    # 逐层耗时（SGC FP16）
    plan = (OUT / "sgc_fp16.engine").read_bytes()
    runner = TRTRunner(plan)
    runner.set_input(torch.rand(1, 3, 640, 640, device="cuda"))
    prof = LayerProf()
    runner.ctx.profiler = prof
    for _ in range(30):
        runner.run()
    torch.cuda.synchronize()
    prof.t.clear()
    prof.cnt.clear()
    for _ in range(50):
        runner.run()
    torch.cuda.synchronize()
    per_layer = {k: prof.t[k] / prof.cnt[k] for k in prof.t}
    total = sum(per_layer.values())
    top = sorted(per_layer.items(), key=lambda kv: -kv[1])
    layers = [{"layer": k, "ms": v, "share": v / total, "cat": categorize(k)} for k, v in top]
    cat_ms: dict[str, float] = defaultdict(float)
    cat_n: dict[str, int] = defaultdict(int)
    for item in layers:
        cat_ms[item["cat"]] += item["ms"]
        cat_n[item["cat"]] += 1
    cats = [{"cat": c, "ms": cat_ms[c], "share": cat_ms[c] / total, "n_layers": cat_n[c]}
            for c in sorted(cat_ms, key=lambda c: -cat_ms[c])]
    result["sgc_fp16_layers"] = {"total_layer_ms": total, "n_layers": len(layers), "categories": cats,
                                 "all": layers}
    print(f"\nSGC FP16 逐层合计 {total:.2f} ms，共 {len(layers)} 层；按类别：")
    for c in cats:
        print(f"  {c['ms']:6.3f} ms  {c['share'] * 100:5.1f}%  {c['n_layers']:4d} 层  {c['cat']}")
    print("耗时 Top 10 层：")
    for item in layers[:10]:
        print(f"  {item['ms']:7.3f} ms  {item['share'] * 100:5.1f}%  {item['layer'][:100]}")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "trt_stream_layer_2026-11-21.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n已保存", OUT / "trt_stream_layer_2026-11-21.json")


if __name__ == "__main__":
    main()
