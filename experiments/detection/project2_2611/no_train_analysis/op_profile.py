"""算子级 Profiling（GPU，batch 1，640×640，随机输入，eager）。

1. 整网：torch.profiler 统计每次前向的算子类别耗时（按 self device time，避免父子算子重复计）与调用次数。
2. 按模块类型：对每个类型的所有实例，用该实例在整网前向中捕获到的真实输入，单独 profile 它们的前向，
   得到该类型每次整网前向中的算子类别耗时与调用次数。组合模块（如 VOVGCSPYolo、C3、SDGhost）的结果
   包含其子模块（inclusive），与子模块类型的结果不应直接相加。

注意：CUDA 时间是 kernel 执行时间之和，不含 CPU 调度空隙；“整网 kernel 时间之和”与“整网墙钟时间”之差即为空隙估计。

用法（仓库根目录）：python experiments/detection/project2_2611/no_train_analysis/op_profile.py
输出：outputs/op_profile/op_profile_2026-11-20.json
"""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

_PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJ))
sys.path.insert(0, str(_PROJ / "analysis_scripts"))
import redundancy_analysis as ra  # noqa: E402

OUT = Path(__file__).resolve().parent / "outputs" / "op_profile"
IMG = 640
ITERS = 30
WARM = 10
MODELS = {
    "YOLOv5s": {"model": "yolov5s", "run": "2026-10-30_002_YOLOv5s_V3",
                "types": ["BaseConv", "C3", "Bottleneck"]},
    "SGC 无注意力": {"model": "sgc_yolov5s_c3_sgdown_gbot_dw5",
                     "run": "2026-11-17_015_SGC_c3_sgdown_gbot_dw5_s_noDFC_cbamNone_seed42_V1",
                     "types": ["SGhostConv", "Bottleneck", "GSConv", "VOVGCSPYolo", "SDGhost"]},
}
CATS = ["标准卷积 Conv", "深度卷积 DWConv", "BN", "激活", "Concat", "Add/Mul", "Copy/Contiguous",
        "内存分配 empty", "池化/上采样/Mean", "形状/视图", "其它"]


def classify(key: str) -> str:
    k = key.replace("aten::", "")
    if k.startswith(("empty", "resize_")):
        return "内存分配 empty"
    if "_conv_depthwise" in k:
        return "深度卷积 DWConv"
    if "convolution" in k or "conv2d" in k:
        return "标准卷积 Conv"
    if "batch_norm" in k:
        return "BN"
    if k in ("silu", "silu_", "sigmoid", "sigmoid_", "relu", "relu_", "hardswish") or k.startswith(("silu", "sigmoid")):
        return "激活"
    if k in ("cat", "stack"):
        return "Concat"
    if k.startswith(("add", "mul", "sub", "div")) and not k.startswith(("addmm", "mul_out")):
        return "Add/Mul"
    if "copy_" in k or "clone" in k or "contiguous" in k:
        return "Copy/Contiguous"
    if any(s in k for s in ("pool", "upsample", "mean")):
        return "池化/上采样/Mean"
    if any(s in k for s in ("view", "reshape", "transpose", "permute", "slice", "split", "as_strided",
                            "select", "narrow", "unsqueeze", "squeeze", "flatten", "expand", "chunk", "unbind")):
        return "形状/视图"
    return "其它"


def self_device_us(e) -> float:
    v = getattr(e, "self_device_time_total", None)
    if v is None:
        v = getattr(e, "self_cuda_time_total", 0.0)
    return float(v or 0.0)


def summarize(evs, iters: int) -> dict:
    cat_ms: dict[str, float] = defaultdict(float)
    cat_cnt: dict[str, float] = defaultdict(float)
    ops = []
    for e in evs:
        cat = classify(e.key)
        t = self_device_us(e) / iters / 1000.0
        c = e.count / iters
        cat_ms[cat] += t
        cat_cnt[cat] += c
        ops.append({"op": e.key, "ms_per_fwd": t, "calls_per_fwd": c, "cat": cat})
    ops.sort(key=lambda d: -d["ms_per_fwd"])
    return {"cat_ms": {c: cat_ms.get(c, 0.0) for c in CATS},
            "cat_calls": {c: cat_cnt.get(c, 0.0) for c in CATS},
            "kernel_ms_total": float(sum(cat_ms.values())),
            "top_ops": ops[:15]}


def run_profile(fn, iters: int):
    with torch.no_grad():
        for _ in range(WARM):
            fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        with torch.no_grad():
            for _ in range(iters):
                fn()
        torch.cuda.synchronize()
    return prof.key_averages()


def main() -> None:
    t0 = time.time()
    device = torch.device("cuda")
    x = torch.rand(1, 3, IMG, IMG, device=device)
    out: dict[str, dict] = {}
    for label, cfg in MODELS.items():
        model = ra.load_model({"model": cfg["model"], "run": cfg["run"]}).eval().to(device)

        # 1) 整网算子类别
        whole = summarize(run_profile(lambda: model(x), ITERS), ITERS)

        # 2) 捕获每个目标类型实例的真实输入（一次整网前向）
        captured: list = []
        handles = []
        types = set(cfg["types"])
        for name, mod in model.named_modules():
            if type(mod).__name__ in types:
                def pre(m, inp, name=name):
                    captured.append((name, type(m).__name__, m, inp[0].detach()))
                handles.append(mod.register_forward_pre_hook(pre))
        with torch.no_grad():
            model(x)
        for h in handles:
            h.remove()

        per_type = {}
        for T in cfg["types"]:
            insts = [(n, m, xi) for n, t, m, xi in captured if t == T]
            if not insts:
                continue

            def fn_type(insts=insts):
                for _, m, xi in insts:
                    m(xi)

            s = summarize(run_profile(fn_type, ITERS), ITERS)
            # 每次整网前向中，该类型的实例数与每个实例的 cat / copy 调用次数
            n_inst = len(insts)
            s["n_instances"] = n_inst
            s["cat_calls_per_instance"] = s["cat_calls"]["Concat"] / n_inst
            s["copy_calls_per_instance"] = s["cat_calls"]["Copy/Contiguous"] / n_inst
            s["dw_ms_per_instance"] = s["cat_ms"]["深度卷积 DWConv"] / n_inst
            per_type[T] = s
            print(f"[{label}] {T}：实例 {n_inst}，kernel 合计 {s['kernel_ms_total']:.3f} ms/前向，"
                  f"Concat {s['cat_calls']['Concat']:.1f} 次，Copy {s['cat_calls']['Copy/Contiguous']:.1f} 次",
                  flush=True)

        out[label] = {"whole": whole, "per_type": per_type}
        print(f"\n== {label} 整网 kernel 合计 {whole['kernel_ms_total']:.3f} ms/前向")
        for c in CATS:
            if whole["cat_calls"][c] > 0 or whole["cat_ms"][c] > 0:
                print(f"   {c:<16} {whole['cat_ms'][c]:8.3f} ms   调用 {whole['cat_calls'][c]:7.1f} 次")
        print("   耗时最大的算子：")
        for d in whole["top_ops"][:10]:
            print(f"     {d['ms_per_fwd']:7.3f} ms  {d['calls_per_fwd']:6.1f} 次  [{d['cat']}] {d['op']}")
        del model
        torch.cuda.empty_cache()

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "op_profile_2026-11-20.json").write_text(
        json.dumps({"iters": ITERS, "results": out, "seconds": round(time.time() - t0)},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n已保存 {OUT / 'op_profile_2026-11-20.json'}（总耗时 {time.time() - t0:.0f}s）")


if __name__ == "__main__":
    main()
