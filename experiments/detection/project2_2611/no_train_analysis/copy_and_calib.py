"""Copy 来源归因 + Profiler 计时口径校验（GPU，batch 1，640×640，eager，无需训练）。

part A：copy 来源归因
  - 对整网前向做一次 profile（record_shapes + with_stack），只统计"顶层" copy 类算子
    （aten::copy_ / contiguous / clone / _to_copy，若父算子也是 copy 则跳过，避免重复计数）。
  - 每次调用的调用栈取第一个属于本仓库模型代码的 Python 帧（跳过 torch 与分析脚本），
    得到 file:line:func，按位置统计每次前向的次数与估计搬运字节数（输入 numel × 4）。

part B：计时口径校验
  - 整网与若干"两模型结构相同"的子模块：
      墙钟（CUDA Event，中位数）
      kernel 原始时间之和（profiler 中 device_type=CUDA 的 kernel 事件持续时间之和）
      op_profile.py 使用的口径（所有 CPU/CUDA 算子的 self_device_time 之和）
  - 比较三者，判断"2.6 倍放大"来自 kernel 本身，还是来自统计口径。

用法（仓库根目录）：python experiments/detection/project2_2611/no_train_analysis/copy_and_calib.py
输出：outputs/copy_attr/copy_attr_2026-11-20.json，outputs/copy_attr/calib_2026-11-20.json
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

OUT = Path(__file__).resolve().parent / "outputs" / "copy_attr"
IMG = 640
WARM = 5
ITERS = 10
CALIB_ITERS = 30
CALIB_WALL = 100
COPY_KEYS = {"aten::copy_", "aten::contiguous", "aten::clone", "aten::_to_copy"}
MODELS = {
    "YOLOv5s": {"model": "yolov5s", "run": "2026-10-30_002_YOLOv5s_V3"},
    "SGC 无注意力": {"model": "sgc_yolov5s_c3_sgdown_gbot_dw5",
                     "run": "2026-11-17_015_SGC_c3_sgdown_gbot_dw5_s_noDFC_cbamNone_seed42_V1"},
}
# 两模型结构相同的子模块（用于计时口径校验）
CALIB_MODULES = ["backbone.stem", "backbone.stage5", "neck.up", "neck.reduce5", "detect"]


def is_model_frame(s: str) -> bool:
    if "weakModelZoo" not in s or "site-packages" in s or "\\torch\\" in s or "/torch/" in s:
        return False
    return "no_train_analysis" not in s and "analysis_scripts" not in s


def top_level_copy_events(events):
    out = []
    for e in events:
        if e.device_type != torch.autograd.DeviceType.CPU or e.key not in COPY_KEYS:
            continue
        p = e.cpu_parent
        nested = False
        while p is not None:
            if p.key in COPY_KEYS:
                nested = True
                break
            p = p.cpu_parent
        if not nested:
            out.append(e)
    return out


def copy_attribution(model, x) -> dict:
    with torch.no_grad():
        for _ in range(WARM):
            model(x)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=True, with_stack=True) as prof:
        with torch.no_grad():
            for _ in range(ITERS):
                model(x)
        torch.cuda.synchronize()
    evs = prof.events()
    copies = top_level_copy_events(evs)
    loc_cnt: dict[str, float] = defaultdict(float)
    loc_bytes: dict[str, float] = defaultdict(float)
    loc_kind: dict[str, dict] = defaultdict(lambda: defaultdict(float))
    no_frame = 0.0
    for e in copies:
        frames = [s for s in (e.stack or []) if is_model_frame(s)]
        key = " <- ".join(f.split("\\")[-1].split("/")[-1] for f in frames[:2]) if frames else "(无模型帧)"
        if not frames:
            no_frame += 1
        loc_cnt[key] += 1.0 / ITERS
        shp = e.input_shapes[0] if e.input_shapes else []
        numel = 1
        for d in shp:
            numel *= d
        loc_bytes[key] += numel * 4.0 / ITERS
        loc_kind[key][e.key] += 1.0 / ITERS
    total_bytes = sum(loc_bytes.values())
    rows = []
    for k in sorted(loc_cnt, key=lambda k: -loc_cnt[k]):
        rows.append({"location": k, "calls_per_fwd": loc_cnt[k],
                     "MB_per_fwd_est": loc_bytes[k] / 1e6,
                     "kinds": dict(loc_kind[k])})
    return {"top_level_copy_per_fwd": len(copies) / ITERS,
            "without_model_frame_per_fwd": no_frame / ITERS,
            "est_MB_per_fwd_total": total_bytes / 1e6,
            "rows": rows}


def run_kernel_sum(fn, iters: int) -> tuple[float, float]:
    """返回 (kernel 原始时间之和 ms/次, 算子 self_device 之和 ms/次)。"""
    with torch.no_grad():
        for _ in range(WARM):
            fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        with torch.no_grad():
            for _ in range(iters):
                fn()
        torch.cuda.synchronize()
    evs = prof.events()
    raw_us = 0.0
    for e in evs:
        if e.device_type == torch.autograd.DeviceType.CUDA:
            raw_us += e.time_range.elapsed_us()
    self_us = 0.0
    for e in evs:
        v = getattr(e, "self_device_time_total", None)
        if v is None:
            v = getattr(e, "self_cuda_time_total", 0.0)
        self_us += float(v or 0.0)
    return raw_us / iters / 1000.0, self_us / iters / 1000.0


def wall_ms(fn) -> float:
    with torch.no_grad():
        for _ in range(WARM):
            fn()
    torch.cuda.synchronize()
    ts = []
    with torch.no_grad():
        for _ in range(CALIB_WALL):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            fn()
            e.record()
            torch.cuda.synchronize()
            ts.append(s.elapsed_time(e))
    return float(np.median(ts))


def capture_inputs(model, x, names: list[str]) -> dict:
    mods = dict(model.named_modules())
    cap: dict = {}
    hs = []
    for n in names:
        def pre(m, inp, n=n):
            cap[n] = inp
        hs.append(mods[n].register_forward_pre_hook(pre))
    with torch.no_grad():
        model(x)
    for h in hs:
        h.remove()
    return cap


def main() -> None:
    t0 = time.time()
    device = torch.device("cuda")
    print(f"GPU：{torch.cuda.get_device_name(0)}；torch {torch.__version__}", flush=True)
    x = torch.rand(1, 3, IMG, IMG, device=device)
    attr_out, calib_out = {}, {}
    for label, cfg in MODELS.items():
        model = ra.load_model({"model": cfg["model"], "run": cfg["run"]}).eval().to(device)

        # ---- part A：copy 来源 ----
        a = copy_attribution(model, x)
        attr_out[label] = a
        print(f"\n== {label}：顶层 copy {a['top_level_copy_per_fwd']:.1f} 次/前向，"
              f"无模型帧 {a['without_model_frame_per_fwd']:.1f}，估计搬运 {a['est_MB_per_fwd_total']:.1f} MB/前向", flush=True)
        for r in a["rows"][:40]:
            print(f"   {r['calls_per_fwd']:6.1f} 次  {r['MB_per_fwd_est']:8.2f} MB  {r['location']}  {r['kinds']}", flush=True)

        # ---- part B：计时口径 ----
        calib = {}
        whole_fn = lambda: model(x)  # noqa: E731
        raw, selfsum = run_kernel_sum(whole_fn, CALIB_ITERS)
        wall = wall_ms(whole_fn)
        calib["whole"] = {"wall_ms": wall, "kernel_raw_ms": raw, "self_sum_ms": selfsum}
        print(f"   [整网] 墙钟 {wall:.2f} ms | kernel 原始和 {raw:.2f} ms（×{raw / wall:.2f}） | "
              f"self_sum {selfsum:.2f} ms（×{selfsum / wall:.2f}）", flush=True)
        cap = capture_inputs(model, x, CALIB_MODULES)
        mods = dict(model.named_modules())
        for n in CALIB_MODULES:
            if n not in cap:
                continue
            inp = cap[n]
            fn = (lambda m=mods[n], i=inp: m(*i))
            w = wall_ms(fn)
            r_, s_ = run_kernel_sum(fn, CALIB_ITERS)
            calib[n] = {"wall_ms": w, "kernel_raw_ms": r_, "self_sum_ms": s_}
            print(f"   [{n}] 墙钟 {w:.3f} ms | kernel 原始和 {r_:.3f}（×{r_ / w:.2f}） | "
                  f"self_sum {s_:.3f}（×{s_ / w:.2f}）", flush=True)
        calib_out[label] = calib
        del model
        torch.cuda.empty_cache()

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "copy_attr_2026-11-20.json").write_text(
        json.dumps(attr_out, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "calib_2026-11-20.json").write_text(
        json.dumps(calib_out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n已保存到 {OUT}（总耗时 {time.time() - t0:.0f}s）")


if __name__ == "__main__":
    main()
