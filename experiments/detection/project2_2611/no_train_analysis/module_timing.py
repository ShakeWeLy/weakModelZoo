"""模块级计时（GPU，batch 1，640×640，随机输入，PyTorch eager，FP32，无后处理）。

方法：
- 整网总耗时：CUDA Event 包住一次完整前向，中位数（不挂 hook，避免干扰）。
- 模块耗时：在 backbone / neck / detect 及其直接子模块上挂 forward pre/post hook，
  用 CUDA Event 记录每次前向的起止时间，中位数。hook 本身有开销，因此模块时间之和
  不保证等于整网时间（注意：模块独立计时受调度、缓存、算子融合影响）。

用法（仓库根目录）：python experiments/detection/project2_2611/no_train_analysis/module_timing.py
输出：outputs/module_timing/module_timing_2026-11-20.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

# 仓库根目录（本文件位于 .../no_train_analysis/ 下，parents[1] 即项目根）
_PROJ = Path(__file__).resolve().parents[1]
# 将项目根目录和 analysis_scripts 加入模块搜索路径，便于 import 仓库内工具
sys.path.insert(0, str(_PROJ))
sys.path.insert(0, str(_PROJ / "analysis_scripts"))
import redundancy_analysis as ra  # noqa: E402  （需先设置 sys.path，故放在此处 import）

OUT = Path(__file__).resolve().parent / "outputs" / "module_timing"
IMG = 640      # 输入分辨率 640×640
ITERS = 200    # 计时迭代次数
WARM = 30      # 预热次数（让 GPU 频率、缓存、cuDNN 算法选择进入稳定状态）
MODELS = {
    # 模型标签 -> 模型配置（模型名 + 训练 run 目录），供 ra.load_model 加载权重
    "YOLOv5s": {"model": "yolov5s", "run": "2026-10-30_002_YOLOv5s_V3"},
    "SGC 无注意力": {"model": "sgc_yolov5s_c3_sgdown_gbot_dw5",
                     "run": "2026-11-17_015_SGC_c3_sgdown_gbot_dw5_s_noDFC_cbamNone_seed42_V1"},
}


def total_ms(model, x) -> tuple[float, float]:
    """测量整网一次前向的耗时，返回 (中位数, P90)，单位毫秒。

    使用 CUDA Event 计时：Event 记录在 GPU 流上，elapsed_time 返回两个 Event
    之间在 GPU 上实际执行的毫秒数，比 CPU 端 time.time() 更准确（不受异步提交影响）。
    """
    with torch.no_grad():          # 关闭 autograd，避免构建计算图带来的额外开销/显存
        for _ in range(WARM):      # 预热：触发 cuDNN benchmark、JIT、缓存分配器等
            model(x)
    # 阻塞 CPU 直到 GPU 上所有先前提交的 kernel 执行完毕，保证预热真正完成
    torch.cuda.synchronize()

    ts = []
    with torch.no_grad():
        for _ in range(ITERS):
            # enable_timing=True：允许 Event 记录时间戳（有少量开销，但可接受）
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()             # 在默认 CUDA 流上插入起始时间戳
            model(x)               # 前向是异步提交的，CPU 立即返回
            e.record()             # 在流上插入结束时间戳（排在所有前向 kernel 之后）
            # 同步等待流上所有工作（含 e）完成，之后 elapsed_time 才有效
            torch.cuda.synchronize()
            # 返回 s 与 e 之间经过的 GPU 时间（毫秒）
            ts.append(s.elapsed_time(e))

    a = np.array(ts)
    # 取中位数抵抗偶发抖动（如其他进程占用 GPU、时钟降频）；P90 反映长尾
    return float(np.median(a)), float(np.percentile(a, 90))


def module_ms(model, x, names: list[str]) -> dict[str, float]:
    """测量指定模块（按名称）单次前向的耗时中位数，单位毫秒。

    实现：为每个模块注册 forward pre-hook 和 forward post-hook，
    在模块前向开始/结束时各记录一个 CUDA Event，事后计算差值。
    注意 hook 本身有开销，且模块独立计时忽略流水线重叠与算子融合，
    因此各模块时间之和与整网时间不完全一致。
    """
    # named_modules() 返回 (名称, 模块对象) 的迭代器，转成 dict 便于按名索引
    mods = dict(model.named_modules())

    with torch.no_grad():
        for _ in range(WARM):      # 预热，保证后续计时稳定
            model(x)
    torch.cuda.synchronize()       # 等待预热 kernel 全部完成

    cur: dict = {}                 # 记录每个模块"当前这次前向"的起始 Event
    rec: dict[str, list] = {n: [] for n in names}  # 每个模块收集 (start, end) Event 对
    handles = []

    for n in names:
        # 用默认参数 n=n 绑定当前名称，避免闭包捕获循环变量（否则所有 hook 都用最后一个 n）
        def pre(m, i, n=n):
            """forward pre-hook：模块前向即将执行时调用。

            参数：m=模块本身，i=输入 tuple。此处只记录 GPU 起始时间戳。
            """
            e = torch.cuda.Event(enable_timing=True)
            e.record()             # 在默认流上插入时间戳（位于该模块 kernel 之前）
            cur[n] = e

        def post(m, i, o, n=n):
            """forward hook：模块前向执行完毕后调用。

            参数：m=模块，i=输入，o=输出。记录结束时间戳并保存 Event 对。
            注意：hook 在 Python 侧被调用时 kernel 仍可能异步执行，
            Event 的时间戳是插入流中的逻辑位置，因此仍能正确度量 GPU 执行区间。
            """
            e = torch.cuda.Event(enable_timing=True)
            e.record()             # 在默认流上插入时间戳（位于该模块 kernel 之后）
            rec[n].append((cur.pop(n), e))  # pop 取出对应起始 Event 并配对

        # register_forward_pre_hook：注册前向之前的回调，返回 RemovableHandle
        handles.append(mods[n].register_forward_pre_hook(pre))
        # register_forward_hook：注册前向之后的回调
        handles.append(mods[n].register_forward_hook(post))

    with torch.no_grad():
        for _ in range(ITERS):     # 注意：这里不在每轮 synchronize，
            model(x)               # Event 对已入队，最后统一同步即可
    torch.cuda.synchronize()       # 等待所有 ITERS 次前向及其 Event 全部执行完

    for h in handles:              # 移除 hook，避免影响后续测量/推理
        h.remove()

    # elapsed_time 返回两个 Event 间 GPU 时间（毫秒）；取中位数
    return {n: float(np.median([s.elapsed_time(e) for s, e in rec[n]])) for n in names}


def main() -> None:
    t0 = time.time()
    # 打印 GPU 型号与 PyTorch 版本，便于结果复现与环境核对
    print(f"GPU：{torch.cuda.get_device_name(0)}；torch {torch.__version__}")
    device = torch.device("cuda")
    # 构造随机输入：batch=1, 3 通道, 640×640，直接放在 GPU 上
    x = torch.rand(1, 3, IMG, IMG, device=device)

    out: dict[str, dict] = {}
    for label, spec in MODELS.items():
        # 加载模型 -> eval() 切换到推理模式（关闭 Dropout、固定 BN 统计量）-> 搬到 GPU
        model = ra.load_model(spec).eval().to(device)

        tot_med, tot_p90 = total_ms(model, x)   # 整网计时

        top = ["backbone", "neck", "detect"]    # 三个顶层模块
        # 收集 backbone / neck 的直接子模块全名，用于细粒度计时
        sub = [f"backbone.{n}" for n, _ in model.backbone.named_children()] + \
              [f"neck.{n}" for n, _ in model.neck.named_children()]

        per = module_ms(model, x, top + sub)    # 顶层 + 子模块分别计时

        out[label] = {"total_ms": tot_med, "total_p90_ms": tot_p90, "modules_ms": per,
                      "backbone_children": [f"backbone.{n}" for n, _ in model.backbone.named_children()],
                      "neck_children": [f"neck.{n}" for n, _ in model.neck.named_children()]}

        s = sum(per[t] for t in top)
        print(f"\n== {label}：整网 {tot_med:.2f} ms（P90 {tot_p90:.2f}）；Backbone+Neck+Detect 之和 {s:.2f} ms")
        for t in top:
            print(f"   {t:<10} {per[t]:7.2f} ms  {100 * per[t] / tot_med:5.1f}%")
        for n in sub:
            print(f"     └ {n:<20} {per[n]:7.2f} ms  {100 * per[n] / tot_med:5.1f}%")

        # 删除模型引用并释放缓存显存，避免两个模型同时驻留 GPU 影响后续计时
        del model
        torch.cuda.empty_cache()   # 清空 PyTorch 的 CUDA 缓存分配器（归还显存给驱动）

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "module_timing_2026-11-20.json").write_text(
        json.dumps({"iters": ITERS, "warmup": WARM, "results": out, "seconds": round(time.time() - t0)},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n已保存 {OUT / 'module_timing_2026-11-20.json'}（总耗时 {time.time() - t0:.0f}s）")


if __name__ == "__main__":
    main()