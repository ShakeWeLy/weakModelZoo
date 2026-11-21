"""ChannelShuffle 消融（仅用于估计 Shuffle 对 TRT 执行成本的上限）。

注意：去掉 ChannelShuffle 后模型不再与原始模型等价，本脚本的输出不能用于判断检测精度，
也不能作为新模型结果；仅用于估计 Shuffle 的执行成本上限。

做法：
- 与原始 SGC 完全相同的 BN 折叠模型，导出 ONNX 时把所有 ChannelShuffle.forward 替换为恒等映射；
- 用与 sgc_fp16.engine 完全相同的构建配置（FP16、工作区 4 GB）构建引擎；
- 对两份 ONNX 统计算子类型计数，确认差异仅在 Reshape/Transpose/Shuffle 相关节点。

用法（仓库根目录）：python experiments/detection/project2_2611/no_train_analysis/shuffle_ablation_build.py
输出：outputs/shuffle_ablation/sgc_noshuffle_folded.onnx、sgc_noshuffle_fp16.engine、onnx_op_counts.json
"""

from __future__ import annotations

import copy
import json
import sys
import time
from collections import Counter
from pathlib import Path

import onnx
import torch

_HERE = Path(__file__).resolve().parent
_PROJ = _HERE.parent
_ROOT = _PROJ.parents[2]
for p in (_ROOT, _PROJ, _PROJ / "analysis_scripts", _HERE):
    sys.path.insert(0, str(p))

import redundancy_analysis as ra  # noqa: E402
from bn_fold import fold_bn_inplace  # noqa: E402
from trt_deploy_bench import build_engine, flat  # noqa: E402

OUT = _HERE / "outputs" / "shuffle_ablation"
TRT_DIR = _HERE / "outputs" / "trt"
SPEC = {"model": "sgc_yolov5s_c3_sgdown_gbot_dw5",
        "run": "2026-11-17_015_SGC_c3_sgdown_gbot_dw5_s_noDFC_cbamNone_seed42_V1"}


def export_identity_shuffle(model: torch.nn.Module, x: torch.Tensor, path: Path) -> int:
    with torch.no_grad():
        n_out = len(flat(model(x)))
    orig_forward = torch.nn.ChannelShuffle.forward
    torch.nn.ChannelShuffle.forward = lambda self, t: t   # 消融：去掉 ChannelShuffle
    try:
        torch.onnx.export(model, (x,), str(path), input_names=["images"],
                          output_names=[f"out{i}" for i in range(n_out)],
                          opset_version=17, dynamo=False, do_constant_folding=True)
    finally:
        torch.nn.ChannelShuffle.forward = orig_forward
    return n_out


def op_counts(path: Path) -> dict:
    m = onnx.load(str(path))
    return dict(Counter(n.op_type for n in m.graph.node))


def main() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    torch.manual_seed(0)
    x = torch.rand(1, 3, 640, 640, device=dev)

    orig = ra.load_model(SPEC).eval().to(dev)
    folded = copy.deepcopy(orig).eval()
    fold_bn_inplace(folded)
    onnx_ns = OUT / "sgc_noshuffle_folded.onnx"
    n_out = export_identity_shuffle(folded, x, onnx_ns)
    print(f"已导出无 Shuffle ONNX：{onnx_ns.name}（输出 {n_out} 个张量）", flush=True)

    counts = {
        "with_shuffle": op_counts(TRT_DIR / "sgc_folded.onnx"),
        "no_shuffle": op_counts(onnx_ns),
    }
    keys = sorted(set(counts["with_shuffle"]) | set(counts["no_shuffle"]))
    print(f"{'op':<22}{'with':>8}{'no_shuffle':>12}{'diff':>8}")
    diff_rows = {}
    for k in keys:
        a, b = counts["with_shuffle"].get(k, 0), counts["no_shuffle"].get(k, 0)
        diff_rows[k] = {"with_shuffle": a, "no_shuffle": b}
        if a != b:
            print(f"{k:<22}{a:>8}{b:>12}{b - a:>+8}")
    (OUT / "onnx_op_counts.json").write_text(
        json.dumps({"counts": counts, "diff": diff_rows}, ensure_ascii=False, indent=1), encoding="utf-8")

    plan_path = OUT / "sgc_noshuffle_fp16.engine"
    build_engine(onnx_ns, True, plan_path)
    print(f"完成，总耗时 {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
