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
# ProfilerActivity：采集 CPU / CUDA 活动；profile：采集上下文管理器
from torch.profiler import ProfilerActivity, profile

# 仓库根目录（本文件位于 .../no_train_analysis/ 下，parents[1] 即项目根）
_PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJ))
sys.path.insert(0, str(_PROJ / "analysis_scripts"))
import redundancy_analysis as ra  # noqa: E402

OUT = Path(__file__).resolve().parent / "outputs" / "copy_attr"
IMG = 640
WARM = 5                # 预热次数（profiler 开销大，这里比纯计时脚本少）
ITERS = 10              # part A profile 的迭代次数
CALIB_ITERS = 30        # part B 中 profiler 计时的迭代次数
CALIB_WALL = 100        # part B 中 CUDA Event 墙钟计时的迭代次数
# 视为"copy 类"的 aten 顶层算子名
COPY_KEYS = {"aten::copy_", "aten::contiguous", "aten::clone", "aten::_to_copy"}
MODELS = {
    "YOLOv5s": {"model": "yolov5s", "run": "2026-10-30_002_YOLOv5s_V3"},
    "SGC 无注意力": {"model": "sgc_yolov5s_c3_sgdown_gbot_dw5",
                     "run": "2026-11-17_015_SGC_c3_sgdown_gbot_dw5_s_noDFC_cbamNone_seed42_V1"},
}
# 两模型结构相同的子模块（用于计时口径校验，避免结构差异干扰对比）
CALIB_MODULES = ["backbone.stem", "backbone.stage5", "neck.up", "neck.reduce5", "detect"]


def is_model_frame(s: str) -> bool:
    """判断调用栈帧是否属于本仓库的模型代码。

    目的：从 profiler 的 Python 调用栈中筛出"用户代码"帧，跳过 torch 内部、
    site-packages 以及本分析脚本自身（否则归因会落到分析脚本上）。
    """
    if "weakModelZoo" not in s or "site-packages" in s or "\\torch\\" in s or "/torch/" in s:
        return False
    return "no_train_analysis" not in s and "analysis_scripts" not in s


def top_level_copy_events(events):
    """从 profiler 事件树中挑出"顶层" copy 算子，避免父子重复计数。

    profiler 的事件是树形结构：例如 aten::contiguous 内部可能调用 aten::copy_，
    两者都在 COPY_KEYS 里。若都统计会重复。做法：沿 cpu_parent 向上找，
    只要祖先里有 copy 类算子，就认为当前事件是嵌套的、跳过。
    """
    out = []
    for e in events:
        # 只看 CPU 侧 aten 算子（CUDA 侧是 kernel 名，不在 COPY_KEYS 中）
        if e.device_type != torch.autograd.DeviceType.CPU or e.key not in COPY_KEYS:
            continue
        # 向上遍历父事件，检测是否存在 copy 类祖先
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
    """Part A：统计每次前向中顶层 copy 算子的来源位置、次数与估计搬运字节数。"""
    with torch.no_grad():
        for _ in range(WARM):
            model(x)
    torch.cuda.synchronize()   # 预热完成，保证 profile 边界干净

    # record_shapes=True：记录输入张量 shape，用于估计搬运字节数
    # with_stack=True：记录 Python 调用栈，用于归因到具体代码位置
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=True, with_stack=True) as prof:
        with torch.no_grad():
            for _ in range(ITERS):
                model(x)
        torch.cuda.synchronize()   # 确保所有 CUDA kernel 被记录

    evs = prof.events()                       # 返回原始（未聚合）事件列表，保留树结构
    copies = top_level_copy_events(evs)       # 只保留顶层 copy，避免重复
    loc_cnt: dict[str, float] = defaultdict(float)                 # 位置 -> 次数/前向
    loc_bytes: dict[str, float] = defaultdict(float)               # 位置 -> 字节/前向
    loc_kind: dict[str, dict] = defaultdict(lambda: defaultdict(float))  # 位置 -> 各算子次数
    no_frame = 0.0                                                 # 无模型帧的 copy 次数

    for e in copies:
        # 从调用栈中筛出模型代码帧；e.stack 是字符串列表（with_stack=True 时存在）
        frames = [s for s in (e.stack or []) if is_model_frame(s)]
        # 取前两个模型帧拼成位置标签（只保留文件名，去掉目录）
        key = " <- ".join(f.split("\\")[-1].split("/")[-1] for f in frames[:2]) if frames else "(无模型帧)"
        if not frames:
            no_frame += 1
        loc_cnt[key] += 1.0 / ITERS            # 归一化到每次前向
        # 估计搬运字节：输入张量 numel × 4 字节（FP32）
        shp = e.input_shapes[0] if e.input_shapes else []
        numel = 1
        for d in shp:
            numel *= d
        loc_bytes[key] += numel * 4.0 / ITERS
        loc_kind[key][e.key] += 1.0 / ITERS    # 记录该位置下各 copy 算子的次数

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
    """Part B：返回 (kernel 原始时间之和 ms/次, 算子 self_device 之和 ms/次)。

    两种口径：
    - raw：遍历 profiler 原始事件，只累加 device_type=CUDA 的 kernel 事件
          的 time_range.elapsed_us()，即"GPU 上实际执行的 kernel 时长之和"。
    - self_sum：所有 CPU/CUDA 算子的 self_device_time_total 之和，
          即 op_profile.py 使用的口径（含 CPU 侧 aten 算子的设备时间字段）。
    两者之差可揭示"统计口径放大"：self_sum 会把同一段 GPU 时间在多个层级重复计入。
    """
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
    # 口径 1：只累加 CUDA kernel 事件的持续时间（原始时间）
    raw_us = 0.0
    for e in evs:
        if e.device_type == torch.autograd.DeviceType.CUDA:
            # time_range 是该事件的起止区间；elapsed_us() 返回区间时长（微秒）
            raw_us += e.time_range.elapsed_us()

    # 口径 2：累加所有算子的 self_device_time（op_profile.py 的口径）
    self_us = 0.0
    for e in evs:
        v = getattr(e, "self_device_time_total", None)
        if v is None:
            v = getattr(e, "self_cuda_time_total", 0.0)
        self_us += float(v or 0.0)

    return raw_us / iters / 1000.0, self_us / iters / 1000.0


def wall_ms(fn) -> float:
    """用 CUDA Event 测墙钟时间中位数（毫秒）。

    Event.record() 在默认流插入时间戳；synchronize() 后 elapsed_time 有效。
    墙钟包含 kernel 执行时间 + CPU 调度空隙 + 流间等待，是与 kernel 时间对比的基准。
    """
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
            torch.cuda.synchronize()   # 等待本轮到 e 为止的所有工作完成
            ts.append(s.elapsed_time(e))   # 毫秒
    return float(np.median(ts))


def capture_inputs(model, x, names: list[str]) -> dict:
    """用 forward pre-hook 捕获指定模块在真实整网前向中收到的输入。

    返回 dict：模块名 -> 输入 tuple。用于后续单独调用该模块时喂入真实 shape/stride。
    """
    mods = dict(model.named_modules())
    cap: dict = {}
    hs = []
    for n in names:
        # 默认参数 n=n 绑定当前名称，避免闭包延迟绑定
        def pre(m, inp, n=n):
            cap[n] = inp
        hs.append(mods[n].register_forward_pre_hook(pre))
    with torch.no_grad():
        model(x)                  # 跑一次整网，触发所有 hook
    for h in hs:
        h.remove()                # 移除 hook，避免影响后续计时
    return cap


def main() -> None:
    t0 = time.time()
    device = torch.device("cuda")
    print(f"GPU：{torch.cuda.get_device_name(0)}；torch {torch.__version__}", flush=True)
    x = torch.rand(1, 3, IMG, IMG, device=device)
    attr_out, calib_out = {}, {}
    for label, cfg in MODELS.items():
        model = ra.load_model({"model": cfg["model"], "run": cfg["run"]}).eval().to(device)

        # ---- part A：copy 来源归因 ----
        a = copy_attribution(model, x)
        attr_out[label] = a
        print(f"\n== {label}：顶层 copy {a['top_level_copy_per_fwd']:.1f} 次/前向，"
              f"无模型帧 {a['without_model_frame_per_fwd']:.1f}，估计搬运 {a['est_MB_per_fwd_total']:.1f} MB/前向", flush=True)
        for r in a["rows"][:40]:
            print(f"   {r['calls_per_fwd']:6.1f} 次  {r['MB_per_fwd_est']:8.2f} MB  {r['location']}  {r['kinds']}", flush=True)

        # ---- part B：计时口径校验 ----
        calib = {}
        whole_fn = lambda: model(x)  # noqa: E731
        # 整网：三种口径对比
        raw, selfsum = run_kernel_sum(whole_fn, CALIB_ITERS)
        wall = wall_ms(whole_fn)
        calib["whole"] = {"wall_ms": wall, "kernel_raw_ms": raw, "self_sum_ms": selfsum}
        print(f"   [整网] 墙钟 {wall:.2f} ms | kernel 原始和 {raw:.2f} ms（×{raw / wall:.2f}） | "
              f"self_sum {selfsum:.2f} ms（×{selfsum / wall:.2f}）", flush=True)

        # 子模块：先用真实输入填充 cap，再逐模块三种口径计时
        cap = capture_inputs(model, x, CALIB_MODULES)
        mods = dict(model.named_modules())
        for n in CALIB_MODULES:
            if n not in cap:
                continue
            inp = cap[n]
            # 默认参数绑定模块与输入，避免闭包延迟绑定
            fn = (lambda m=mods[n], i=inp: m(*i))
            w = wall_ms(fn)
            r_, s_ = run_kernel_sum(fn, CALIB_ITERS)
            calib[n] = {"wall_ms": w, "kernel_raw_ms": r_, "self_sum_ms": s_}
            print(f"   [{n}] 墙钟 {w:.3f} ms | kernel 原始和 {r_:.3f}（×{r_ / w:.2f}） | "
                  f"self_sum {s_:.3f}（×{s_ / w:.2f}）", flush=True)
        calib_out[label] = calib

        del model
        torch.cuda.empty_cache()   # 清空缓存分配器，避免模型间显存干扰

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "copy_attr_2026-11-20.json").write_text(
        json.dumps(attr_out, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "calib_2026-11-20.json").write_text(
        json.dumps(calib_out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n已保存到 {OUT}（总耗时 {time.time() - t0:.0f}s）")


if __name__ == "__main__":
    main()