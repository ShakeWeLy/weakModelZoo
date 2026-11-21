"""置换折叠后的导出、ONNX 算子对比、TRT FP16 构建与速度对比。

对比对象：
- 原始 SGC FP16（sgc_fp16.engine）
- 无 Shuffle 消融上限（sgc_noshuffle_fp16.engine，非等价模型）
- 置换折叠 SGC FP16（本脚本构建，数学等价）

用法（仓库根目录）：python experiments/detection/project2_2611/no_train_analysis/shuffle_fold_bench.py
输出：outputs/shuffle_fold/fold_bench_2026-11-21.json，sgc_fold.onnx，sgc_fold_fp16.engine
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_PROJ = _HERE.parent
_ROOT = _PROJ.parents[2]
for p in (_ROOT, _PROJ, _PROJ / "analysis_scripts", _HERE):
    sys.path.insert(0, str(p))

from shuffle_fold import OUT, fold_graph, verify  # noqa: E402
from trt_deploy_bench import TRTRunner, build_engine  # noqa: E402
from shuffle_ablation_bench import engine_structure, layer_profile, pipe_ms, sync_median  # noqa: E402
from shuffle_ablation_build import op_counts  # noqa: E402

TRT_DIR = _HERE / "outputs" / "trt"
ABL_DIR = _HERE / "outputs" / "shuffle_ablation"
TRIALS = 3


def main() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    ref, new, report = fold_graph()
    v = verify(ref, new)
    print(f"数值校验：相对误差 {v['rel']:.2e}，剩余 ChannelShuffle {v['channel_shuffle_nodes_left']}", flush=True)

    # 1) 导出 ONNX（与 trt_deploy_bench 一致：opset 17，静态 batch 1）
    dev = torch.device("cuda")
    x = torch.rand(1, 3, 640, 640, device=dev)
    new = new.to(dev).eval()
    onnx_path = OUT / "sgc_fold.onnx"
    with torch.no_grad():
        n_out = len(new(x)) if isinstance(new(x), (list, tuple)) else 1
    torch.onnx.export(new, (x,), str(onnx_path), input_names=["images"],
                      output_names=[f"out{i}" for i in range(n_out)],
                      opset_version=17, dynamo=False, do_constant_folding=True)
    counts = {
        "original": op_counts(TRT_DIR / "sgc_folded.onnx"),
        "no_shuffle_upper_bound": op_counts(ABL_DIR / "sgc_noshuffle_folded.onnx"),
        "shuffle_folded": op_counts(onnx_path),
    }
    keys = ["Reshape", "Transpose", "Concat", "Gather", "Constant", "Conv", "Add", "Mul", "Sigmoid", "Resize", "Upsample"]
    print(f"{'op':<12}{'原始':>8}{'无Shuffle上限':>16}{'置换折叠':>10}")
    for k in keys:
        print(f"{k:<12}{counts['original'].get(k, 0):>8}{counts['no_shuffle_upper_bound'].get(k, 0):>16}"
              f"{counts['shuffle_folded'].get(k, 0):>10}", flush=True)

    # 2) 构建 TRT FP16 引擎
    plan_path = OUT / "sgc_fold_fp16.engine"
    if not plan_path.exists():
        build_engine(onnx_path, True, plan_path)
    plan = plan_path.read_bytes()

    # 3) 速度对比（交替测量）
    engines = {
        "原始 SGC FP16": (TRT_DIR / "sgc_fp16.engine").read_bytes(),
        "无 Shuffle 上限 FP16（非等价）": (ABL_DIR / "sgc_noshuffle_fp16.engine").read_bytes(),
        "置换折叠 SGC FP16（等价）": plan,
    }
    runners = {}
    for label, p in engines.items():
        r = TRTRunner(p)
        r.set_input(x)
        for _ in range(30):
            r.run()
        torch.cuda.synchronize()
        runners[label] = r
    lat = {k: {"sync": [], "pipe": []} for k in runners}
    for _ in range(TRIALS):
        for label, r in runners.items():
            lat[label]["sync"].append(sync_median(r.run))
            lat[label]["pipe"].append(pipe_ms(r.run))
    results = {}
    for label, r in runners.items():
        results[label] = {
            "latency_sync_median_ms": float(np.median(lat[label]["sync"])),
            "latency_sync_trials": lat[label]["sync"],
            "latency_pipe_median_ms": float(np.median(lat[label]["pipe"])),
            "latency_pipe_trials": lat[label]["pipe"],
            "structure": engine_structure(engines[label]),
            "layer_profile": layer_profile(r),
        }
        print(f"{label:<28} 逐次同步 {results[label]['latency_sync_median_ms']:.2f} | 流水线 "
              f"{results[label]['latency_pipe_median_ms']:.2f} ms | 层数 {results[label]['structure']['n_layers']} | "
              f"shuffle 命名层 {results[label]['structure']['n_shuffle_named']} | reformat 命名层 "
              f"{results[label]['structure']['n_reformat_named']}", flush=True)

    out = {"verify": v, "fold_report_summary": {k: report[k] for k in
                                               ["rounds", "n_shuffle_removed", "n_conv_weights_rewritten",
                                                "n_conv_output_perm_chosen"]},
           "onnx_op_counts": counts, "trt": results, "seconds": round(time.time() - t0)}
    (OUT / "fold_bench_2026-11-21.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"已保存 {OUT / 'fold_bench_2026-11-21.json'}（总耗时 {time.time() - t0:.0f}s）")


if __name__ == "__main__":
    main()
