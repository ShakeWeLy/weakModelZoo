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
# ProfilerActivity：指定 profiler 采集哪些活动（CPU 侧算子调度 / CUDA 侧 kernel）
# profile：上下文管理器，进入后开始采集，退出时收集事件
from torch.profiler import ProfilerActivity, profile

# 仓库根目录（本文件位于 .../no_train_analysis/ 下，parents[1] 即项目根）
_PROJ = Path(__file__).resolve().parents[1]
# 将项目根和 analysis_scripts 加入模块搜索路径，便于 import 仓库内工具
sys.path.insert(0, str(_PROJ))
sys.path.insert(0, str(_PROJ / "analysis_scripts"))
import redundancy_analysis as ra  # noqa: E402  （需先设置 sys.path，故放在此处 import）

OUT = Path(__file__).resolve().parent / "outputs" / "op_profile"
IMG = 640      # 输入分辨率 640×640
ITERS = 30     # profile 迭代次数（profiler 开销大，取较小值；结果按次数归一化）
WARM = 10      # 预热次数（让 GPU 频率、cuDNN 算法选择、缓存进入稳定状态）
MODELS = {
    # 模型标签 -> 配置：模型名、训练 run 目录、要按模块类型细分的类型名列表
    "YOLOv5s": {"model": "yolov5s", "run": "2026-10-30_002_YOLOv5s_V3",
                "types": ["BaseConv", "C3", "Bottleneck"]},
    "SGC 无注意力": {"model": "sgc_yolov5s_c3_sgdown_gbot_dw5",
                     "run": "2026-11-17_015_SGC_c3_sgdown_gbot_dw5_s_noDFC_cbamNone_seed42_V1",
                     "types": ["SGhostConv", "Bottleneck", "GSConv", "VOVGCSPYolo", "SDGhost"]},
}
# 算子分类的固定顺序（输出时保证各类别都在，缺失填 0）
CATS = ["标准卷积 Conv", "深度卷积 DWConv", "BN", "激活", "Concat", "Add/Mul", "Copy/Contiguous",
        "内存分配 empty", "池化/上采样/Mean", "形状/视图", "其它"]


def classify(key: str) -> str:
    """根据 aten 算子名（如 'aten::convolution'）归入语义类别。

    profiler 的 e.key 形如 'aten::xxx' 或 kernel 名（如 'Memset (Device)'），
    这里去掉 'aten::' 前缀后按关键字匹配。注意顺序很重要：
    先判 empty/resize_ 避免被后续规则误捕，再判 depthwise 再判普通卷积。
    """
    k = key.replace("aten::", "")
    if k.startswith(("empty", "resize_")):
        return "内存分配 empty"          # 张量分配/重置，常见于 cat/contiguous 前
    if "_conv_depthwise" in k:
        return "深度卷积 DWConv"          # cuDNN 的 depthwise 专用 kernel
    if "convolution" in k or "conv2d" in k:
        return "标准卷积 Conv"            # 普通卷积（含分组卷积）
    if "batch_norm" in k:
        return "BN"
    if k in ("silu", "silu_", "sigmoid", "sigmoid_", "relu", "relu_", "hardswish") or k.startswith(("silu", "sigmoid")):
        return "激活"
    if k in ("cat", "stack"):
        return "Concat"                  # 通道拼接，YOLO 的 neck 里很多
    # 排除 addmm/mul_out（属于矩阵乘，不是逐元素运算）
    if k.startswith(("add", "mul", "sub", "div")) and not k.startswith(("addmm", "mul_out")):
        return "Add/Mul"
    if "copy_" in k or "clone" in k or "contiguous" in k:
        return "Copy/Contiguous"         # 显存搬运，常由 cat/permute 触发
    if any(s in k for s in ("pool", "upsample", "mean")):
        return "池化/上采样/Mean"
    if any(s in k for s in ("view", "reshape", "transpose", "permute", "slice", "split", "as_strided",
                            "select", "narrow", "unsqueeze", "squeeze", "flatten", "expand", "chunk", "unbind")):
        return "形状/视图"                # 通常零拷贝，只改元数据
    return "其它"


def self_device_us(e) -> float:
    """读取单个事件的 self device 时间（微秒）。

    self_device_time_total 是 PyTorch 新版字段（CPU/GPU 通用命名）；
    老版本只有 self_cuda_time_total。用 self（独占）而非 total（含子算子）
    是为了避免父子算子重复计数：例如 convolution 的 total 已包含其内部 kernel，
    若再加子项会重复。
    """
    v = getattr(e, "self_device_time_total", None)
    if v is None:
        v = getattr(e, "self_cuda_time_total", 0.0)
    return float(v or 0.0)


def summarize(evs, iters: int) -> dict:
    """把 profiler 的事件列表按类别汇总，并归一化到"每次前向"。

    参数 evs：profile.key_averages() 返回的事件平均值列表（每次事件一行）。
    参数 iters：profile 期间执行的前向次数，用于把总量除以次数得到单次值。
    """
    cat_ms: dict[str, float] = defaultdict(float)   # 类别 -> 每次前向耗时（ms）
    cat_cnt: dict[str, float] = defaultdict(float)  # 类别 -> 每次前向调用次数
    ops = []
    for e in evs:
        cat = classify(e.key)
        # self_device_us 单位为微秒；/iters 得到单次前向；/1000 转毫秒
        t = self_device_us(e) / iters / 1000.0
        c = e.count / iters          # e.count 是 iters 次前向的总调用次数
        cat_ms[cat] += t
        cat_cnt[cat] += c
        ops.append({"op": e.key, "ms_per_fwd": t, "calls_per_fwd": c, "cat": cat})
    ops.sort(key=lambda d: -d["ms_per_fwd"])   # 按耗时降序，取 top 算子
    # 用 CATS 固定顺序输出，缺失类别补 0，便于对比
    return {"cat_ms": {c: cat_ms.get(c, 0.0) for c in CATS},
            "cat_calls": {c: cat_cnt.get(c, 0.0) for c in CATS},
            "kernel_ms_total": float(sum(cat_ms.values())),  # 所有 kernel 时间之和
            "top_ops": ops[:15]}


def run_profile(fn, iters: int):
    """对可调用 fn 做 iters 次前向的 profiler 采集，返回 key_averages()。

    - 先预热 WARM 次并 synchronize，避免把首轮 cuDNN 算法搜索等开销算进去。
    - profile 上下文同时采集 CPU 与 CUDA 活动：
      CPU 活动用于看到 aten 算子调度，CUDA 活动用于拿到真实 kernel 执行时间。
    - 在 profile 块内最后 synchronize，确保所有 CUDA kernel 都被记录到。
    """
    with torch.no_grad():
        for _ in range(WARM):
            fn()
    torch.cuda.synchronize()     # 等待预热 kernel 完成，计时边界清晰

    # activities 指定采集类别；默认还会采集内存/栈等，这里只要 CPU+CUDA
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        with torch.no_grad():
            for _ in range(iters):
                fn()
        # 必须同步，否则 profiler 退出时部分 CUDA kernel 可能尚未执行完而漏记
        torch.cuda.synchronize()

    # key_averages() 按算子名聚合，返回每个算子的平均耗时/调用次数等统计
    return prof.key_averages()


def main() -> None:
    t0 = time.time()
    device = torch.device("cuda")
    # 随机输入：batch=1, 3 通道, 640×640，直接放 GPU
    x = torch.rand(1, 3, IMG, IMG, device=device)
    out: dict[str, dict] = {}
    for label, cfg in MODELS.items():
        # 加载模型 -> eval() 推理模式 -> 搬到 GPU
        model = ra.load_model({"model": cfg["model"], "run": cfg["run"]}).eval().to(device)

        # 1) 整网算子类别：对完整前向做 profile，按类别汇总
        whole = summarize(run_profile(lambda: model(x), ITERS), ITERS)

        # 2) 捕获每个目标类型实例的真实输入（一次整网前向）
        #    目的：后续单独 profile 某类型实例时，用它们在真实网络中收到的输入，
        #    保证 shape/stride 与实际一致，避免用随机输入引入偏差。
        captured: list = []      # 元素：(模块全名, 类型名, 模块对象, 输入张量)
        handles = []
        types = set(cfg["types"])
        for name, mod in model.named_modules():
            if type(mod).__name__ in types:
                # 前向 pre-hook：拿到模块的真实输入 inp[0]，detach 去掉计算图引用
                def pre(m, inp, name=name):
                    captured.append((name, type(m).__name__, m, inp[0].detach()))
                handles.append(mod.register_forward_pre_hook(pre))
        with torch.no_grad():
            model(x)             # 跑一次整网，触发所有 hook，填满 captured
        for h in handles:
            h.remove()               # 移除 hook，避免影响后续 profile

        per_type = {}
        for T in cfg["types"]:
            # 取出该类型的所有实例及其捕获输入
            insts = [(n, m, xi) for n, t, m, xi in captured if t == T]
            if not insts:
                continue

            # 闭包：对该类型的所有实例各跑一次前向（默认参数绑定 insts，避免延迟绑定）
            def fn_type(insts=insts):
                for _, m, xi in insts:
                    m(xi)

            s = summarize(run_profile(fn_type, ITERS), ITERS)
            # 归一化到"每个实例/每次整网前向"的指标
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

        # 删除模型引用并清空 CUDA 缓存，避免两个模型同时驻留影响显存/计时
        del model
        torch.cuda.empty_cache()

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "op_profile_2026-11-20.json").write_text(
        json.dumps({"iters": ITERS, "results": out, "seconds": round(time.time() - t0)},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n已保存 {OUT / 'op_profile_2026-11-20.json'}（总耗时 {time.time() - t0:.0f}s）")


if __name__ == "__main__":
    main()