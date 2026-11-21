"""推理期 BN 折叠的数值校验、算子计数与速度对比（GPU，batch 1，640×640，eager 与 CUDA Graph）。

对每个模型比较两个版本：原版与 BN 折叠版（deepcopy 后折叠，不改动原模型）。
- 数值：同一固定随机输入，输出张量的最大绝对误差；
- 算子：一次前向中 aten::batch_norm 调用次数与 aten 算子调用总数（CPU 侧 profiler 计数）；
- 速度：eager 逐次同步、eager 流水线吞吐、CUDA Graph 回放，3 次试验取中位数，四个变体交替测量。

用法（仓库根目录）：python experiments/detection/project2_2611/no_train_analysis/bn_fold_bench.py
输出：outputs/bn_fold/bn_fold_2026-11-20.json
"""

from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile

_PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJ))
sys.path.insert(0, str(_PROJ / "analysis_scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import redundancy_analysis as ra  # noqa: E402
from bn_fold import count_bn, fold_bn_inplace  # noqa: E402

OUT = Path(__file__).resolve().parent / "outputs" / "bn_fold"
TRIALS = 3
WARM = 30
EAGER_N = 200
PIPE_N = 200
GRAPH_N = 200
MODELS = {
    "SGC 无注意力": {"model": "sgc_yolov5s_c3_sgdown_gbot_dw5",
                     "run": "2026-11-17_015_SGC_c3_sgdown_gbot_dw5_s_noDFC_cbamNone_seed42_V1"},
    "YOLOv5s": {"model": "yolov5s", "run": "2026-10-30_002_YOLOv5s_V3"},
}


def flat(o) -> list[torch.Tensor]:
    if torch.is_tensor(o):
        return [o]
    if isinstance(o, (list, tuple)):
        return [t for i in o for t in flat(i)]
    if isinstance(o, dict):
        return [t for v in o.values() for t in flat(v)]
    return []


def count_ops(model, x) -> tuple[float, float]:
    """返回 (aten::batch_norm 调用次数, aten 算子调用总数)，每次前向。"""
    with torch.no_grad():
        for _ in range(3):
            model(x)
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        with torch.no_grad():
            model(x)
    bn = 0.0
    total = 0.0
    for ka in prof.key_averages():
        if not ka.key.startswith("aten::"):
            continue
        total += ka.count
        if ka.key == "aten::batch_norm":
            bn += ka.count
    return bn, total


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
    return float(np.median(ts))


def eager_pipe(model, x) -> float:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(PIPE_N):
            model(x)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / PIPE_N


def graph_replay(model, x) -> float:
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
    return float(np.median(ts))


def measure(model, x) -> dict:
    with torch.no_grad():
        for _ in range(WARM):
            model(x)
    torch.cuda.synchronize()
    return {"eager_sync_ms": eager_sync(model, x),
            "eager_pipe_ms": eager_pipe(model, x),
            "graph_ms": graph_replay(model, x)}


def main() -> None:
    t0 = time.time()
    device = torch.device("cuda")
    torch.manual_seed(0)
    x = torch.rand(1, 3, 640, 640, device=device)
    OUT.mkdir(parents=True, exist_ok=True)

    variants: dict[str, torch.nn.Module] = {}
    report: dict[str, dict] = {}
    for label, cfg in MODELS.items():
        orig = ra.load_model(cfg).eval().to(device)
        fused = copy.deepcopy(orig).eval()
        n_fold = fold_bn_inplace(fused)
        fused = fused.to(device)
        bn_before, ops_before = count_ops(orig, x)
        bn_after, ops_after = count_ops(fused, x)
        with torch.no_grad():
            o_orig = [t.float().cpu() for t in flat(orig(x))]
            o_fused = [t.float().cpu() for t in flat(fused(x))]
        max_abs = max(float((a - b).abs().max()) for a, b in zip(o_orig, o_fused))
        max_val = max(float(a.abs().max()) for a in o_orig)
        report[label] = {
            "folded_blocks": n_fold,
            "BatchNorm2d_modules_before": count_bn(orig),
            "BatchNorm2d_modules_after": count_bn(fused),
            "aten_batch_norm_calls_before": bn_before,
            "aten_batch_norm_calls_after": bn_after,
            "aten_total_calls_before": ops_before,
            "aten_total_calls_after": ops_after,
            "output_max_abs_err": max_abs,
            "output_max_abs_value": max_val,
        }
        print(f"[{label}] 折叠 {n_fold} 个 block；BatchNorm2d {count_bn(orig)} → {count_bn(fused)}；"
              f"aten::batch_norm {bn_before:.0f} → {bn_after:.0f}；算子调用 {ops_before:.0f} → {ops_after:.0f}；"
              f"输出最大误差 {max_abs:.3e}（输出最大值 {max_val:.2f}）", flush=True)
        variants[f"{label} 原版"] = orig
        variants[f"{label} BN折叠"] = fused

    rows: dict[str, list[dict]] = {k: [] for k in variants}
    for trial in range(TRIALS):
        for name, model in variants.items():
            r = measure(model, x)
            rows[name].append(r)
            print(f"[trial{trial + 1}] {name:<18} eager逐次同步 {r['eager_sync_ms']:6.2f} | "
                  f"eager流水线 {r['eager_pipe_ms']:6.2f} | Graph {r['graph_ms']:5.2f} ms", flush=True)

    summary = {}
    for name, rs in rows.items():
        summary[name] = {k: {"median": float(np.median([r[k] for r in rs])),
                             "trials": [r[k] for r in rs]}
                         for k in ["eager_sync_ms", "eager_pipe_ms", "graph_ms"]}
    out = {"report": report, "speed": summary, "seconds": round(time.time() - t0)}
    (OUT / "bn_fold_2026-11-20.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n已保存 {OUT / 'bn_fold_2026-11-20.json'}（总耗时 {time.time() - t0:.0f}s）")


if __name__ == "__main__":
    main()
