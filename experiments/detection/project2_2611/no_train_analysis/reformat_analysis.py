"""剩余 Reformat kernel 的定位：把每个 Reformat 层映射到其张量、格式转换与消费者。

步骤：
1. 用 DETAILED profiling verbosity 构建置换折叠 SGC 的 FP16 引擎（使 inspector 输出每层的输入/输出格式）；
2. 导出 inspector 的逐层 JSON，筛选 Reformat 类层，记录其输入张量、输出张量、格式/精度与消费者；
3. 用 IProfiler 统计每个 Reformat 层的逐层耗时（同步开销使绝对值偏高，仅作相对参考）；
4. 按"网络 I/O / 内部布局 / 精度转换"归类。

用法（仓库根目录）：python experiments/detection/project2_2611/no_train_analysis/reformat_analysis.py
输出：outputs/reformat/…
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import tensorrt as trt
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from trt_deploy_bench import TRTRunner  # noqa: E402
from trt_stream_layer_check import LayerProf  # noqa: E402

OUT = _HERE / "outputs" / "reformat"
ONNX = _HERE / "outputs" / "shuffle_fold" / "sgc_fold.onnx"
ENGINE = OUT / "sgc_fold_fp16_detailed.engine"


def build_detailed(onnx_path: Path, plan_path: Path) -> bytes:
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(onnx_path.read_bytes()):
        errs = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        raise RuntimeError("ONNX 解析失败：\n" + "\n".join(errs))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
    config.set_flag(trt.BuilderFlag.FP16)
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    t0 = time.time()
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("引擎构建失败")
    plan_path.write_bytes(plan)
    print(f"DETAILED 引擎构建 {time.time() - t0:.0f}s，{plan_path.stat().st_size / 1e6:.1f} MB", flush=True)
    return bytes(plan)


def is_reformat(layer: dict) -> bool:
    """Reformat 类层 + TRT 的 copy 层（Concat 输入拷贝、输出拷贝等）。"""
    t = str(layer.get("LayerType", "")).lower()
    n = str(layer.get("Name", "")).lower()
    return ("reformat" in t or "reformat" in n or "copynode" in n
            or n.endswith(" copy") or "_copy_output" in n)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if ENGINE.exists():
        plan = ENGINE.read_bytes()
        print("使用已有 DETAILED 引擎", flush=True)
    else:
        plan = build_detailed(ONNX, ENGINE)

    logger = trt.Logger(trt.Logger.ERROR)
    eng = trt.Runtime(logger).deserialize_cuda_engine(plan)
    info = json.loads(eng.create_engine_inspector().get_engine_information(trt.LayerInformationFormat.JSON))
    layers = info["Layers"]
    (OUT / "layers_verbose.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"层数 {len(layers)}", flush=True)

    types = Counter(str(l.get("LayerType")) for l in layers)
    print("层类型统计：", dict(types.most_common()), flush=True)

    # 逐层耗时（IProfiler，50 次平均）
    runner = TRTRunner(plan)
    torch.manual_seed(0)
    runner.set_input(torch.rand(1, 3, 640, 640, device="cuda"))
    for _ in range(20):
        runner.run()
    torch.cuda.synchronize()
    prof = LayerProf()
    runner.ctx.profiler = prof
    for _ in range(10):
        runner.run()
    torch.cuda.synchronize()
    prof.t.clear()
    prof.cnt.clear()
    for _ in range(50):
        runner.run()
    torch.cuda.synchronize()
    t_layer = {k: prof.t[k] / prof.cnt[k] for k in prof.t}
    (OUT / "layer_times.json").write_text(json.dumps(t_layer, ensure_ascii=False, indent=1), encoding="utf-8")

    # Reformat 层：输入/输出张量与格式、ONNX 来源、耗时
    rows = []
    for l in layers:
        if not is_reformat(l):
            continue
        name = l["Name"]
        ins = [f"{x['Name']} [{x.get('Format/Datatype', '')}]" for x in l.get("Inputs", [])]
        outs = [f"{x['Name']} [{x.get('Format/Datatype', '')}]" for x in l.get("Outputs", [])]
        meta = l.get("Metadata", "")
        onnx_nodes = [s.replace("[ONNX Layer: ", "").rstrip("]") for s in meta.split("\u001f") if "ONNX Layer" in s]
        rows.append({"name": name, "type": l.get("LayerType"), "inputs": ins, "outputs": outs,
                     "onnx": onnx_nodes, "ms": t_layer.get(name, 0.0)})
    rows.sort(key=lambda r: -r["ms"])
    (OUT / "reformat_layers.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    total_ms = sum(r["ms"] for r in rows)
    print(f"\nReformat 层 {len(rows)} 个，逐层耗时合计 {total_ms:.3f} ms（IProfiler，偏高）", flush=True)
    for r in rows[:60]:
        print(f"{r['ms']:7.4f} ms | {r['name'][:60]:<60} | in {r['inputs']} -> out {r['outputs']} | onnx {r['onnx'][:3]}",
              flush=True)


if __name__ == "__main__":
    main()
