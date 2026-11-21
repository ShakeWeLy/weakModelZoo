"""ChannelShuffle TRT 消融：latency、引擎结构、逐层耗时（同一进程、同一时间窗、交替测量）。

注意：无 Shuffle 版本不是等价模型，结果仅用于估计 Shuffle 的执行成本上限，不代表检测精度或新模型性能。

用法（仓库根目录）：python experiments/detection/project2_2611/no_train_analysis/shuffle_ablation_bench.py
输出：outputs/shuffle_ablation/bench_2026-11-21.json
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
from trt_stream_layer_check import LayerProf, categorize  # noqa: E402

OUT = _HERE / "outputs" / "shuffle_ablation"
TRT_DIR = _HERE / "outputs" / "trt"
ENGINES = {
    "原始 SGC FP16": TRT_DIR / "sgc_fp16.engine",
    "去 Shuffle FP16（消融）": OUT / "sgc_noshuffle_fp16.engine",
}
TRIALS = 3
WARM = 30
EAGER_N = 200
PIPE_N = 200
PROF_N = 100


def sync_median(fn) -> float:
    ts = []
    for _ in range(EAGER_N):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return float(np.median(ts))


def pipe_ms(fn) -> float:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(PIPE_N):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / PIPE_N


def engine_structure(plan: bytes) -> dict:
    logger = trt.Logger(trt.Logger.ERROR)
    eng = trt.Runtime(logger).deserialize_cuda_engine(plan)
    info = json.loads(eng.create_engine_inspector().get_engine_information(trt.LayerInformationFormat.JSON))
    names = [x if isinstance(x, str) else x.get("Name", "") for x in info["Layers"]]
    cats: dict[str, int] = defaultdict(int)
    for n in names:
        cats[categorize(n)] += 1
    return {"n_layers": len(names), "n_io_bindings": len(info.get("Bindings", [])),
            "category_layer_count": dict(sorted(cats.items(), key=lambda kv: -kv[1])),
            "n_shuffle_named": sum("shuffle" in n for n in names),
            "n_reformat_named": sum(("Reformatting" in n or "CopyNode" in n) for n in names)}


def layer_profile(runner: TRTRunner) -> dict:
    prof = LayerProf()
    runner.ctx.profiler = prof
    for _ in range(20):
        runner.run()
    torch.cuda.synchronize()
    prof.t.clear()
    prof.cnt.clear()
    for _ in range(PROF_N):
        runner.run()
    torch.cuda.synchronize()
    per = {k: prof.t[k] / prof.cnt[k] for k in prof.t}
    cat_ms: dict[str, float] = defaultdict(float)
    cat_n: dict[str, int] = defaultdict(int)
    for k, v in per.items():
        c = categorize(k)
        cat_ms[c] += v
        cat_n[c] += 1
    total = sum(per.values())
    shuffle_ms = sum(v for k, v in per.items() if "shuffle" in k)
    reformat_ms = sum(v for k, v in per.items() if "Reformatting" in k or "CopyNode" in k)
    return {"layer_sum_ms": total, "shuffle_named_ms": shuffle_ms, "reformat_ms": reformat_ms,
            "shuffle_plus_reformat_ms": shuffle_ms + reformat_ms,
            "category_ms": {c: cat_ms[c] for c in sorted(cat_ms, key=lambda c: -cat_ms[c])},
            "category_layers": dict(cat_n)}


def main() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    x = torch.rand(1, 3, 640, 640, device="cuda")
    runners, res = {}, {}
    for label, path in ENGINES.items():
        plan = path.read_bytes()
        res[label] = {"engine_file": path.name, "engine_MB": path.stat().st_size / 1e6,
                      "structure": engine_structure(plan), "latency": {"eager_sync_ms": [], "eager_pipe_ms": []}}
        r = TRTRunner(plan)
        r.set_input(x)
        for _ in range(WARM):
            r.run()
        torch.cuda.synchronize()
        runners[label] = r

    # 交替测量，降低漂移影响
    for trial in range(TRIALS):
        for label, r in runners.items():
            s_ms = sync_median(r.run)
            p_ms = pipe_ms(r.run)
            res[label]["latency"]["eager_sync_ms"].append(s_ms)
            res[label]["latency"]["eager_pipe_ms"].append(p_ms)
            print(f"[trial{trial + 1}] {label:<22} 逐次同步 {s_ms:.2f} | 流水线 {p_ms:.2f} ms", flush=True)

    for label, r in runners.items():
        res[label]["layer_profile"] = layer_profile(r)
        lp = res[label]["layer_profile"]
        print(f"{label}：逐层合计 {lp['layer_sum_ms']:.2f} ms | shuffle 层 {lp['shuffle_named_ms']:.3f} ms | "
              f"Reformatting {lp['reformat_ms']:.3f} ms", flush=True)

    for label in res:
        lat = res[label]["latency"]
        lat["eager_sync_median_ms"] = float(np.median(lat["eager_sync_ms"]))
        lat["eager_pipe_median_ms"] = float(np.median(lat["eager_pipe_ms"]))
        s = res[label]["structure"]
        print(f"{label:<22} 层数 {s['n_layers']} | shuffle 命名层 {s['n_shuffle_named']} | "
              f"reformat 命名层 {s['n_reformat_named']} | 引擎 {res[label]['engine_MB']:.1f} MB | "
              f"逐次同步中位数 {lat['eager_sync_median_ms']:.2f} ms")

    (OUT / "bench_2026-11-21.json").write_text(json.dumps({"results": res, "seconds": round(time.time() - t0)},
                                                          ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"已保存 {OUT / 'bench_2026-11-21.json'}（总耗时 {time.time() - t0:.0f}s）")


if __name__ == "__main__":
    main()
