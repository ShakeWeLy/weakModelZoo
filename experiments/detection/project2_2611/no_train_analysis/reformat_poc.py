"""Reformat 根因的概念验证（不改源码，只改导出图）。

两处改动，各自独立验证：
A) 只导出 out0（部署所需的预测），去掉训练用的原始特征图输出 out1/2/3；
B) Detect._grid 直接用 float32 生成网格（去掉 ONNX 中由 .float() 产生的 Cast→FLOAT 节点），
   检验 TRT 是否因此把解码路径从 FP32 改为 FP16。

数值检查：改动只影响导出图，PyTorch 中 out0 与折叠版逐元素比较。
之后构建 DETAILED FP16 引擎，与置换折叠版做同样的 Reformat 分类与逐层耗时对比。

用法（仓库根目录）：python experiments/detection/project2_2611/no_train_analysis/reformat_poc.py
输出：outputs/reformat/poc_*.onnx、poc_*.json；引擎 poc_lean_fp16_detailed.engine
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import onnx
import torch
import torch.nn as nn

_HERE = Path(__file__).resolve().parent
_PROJ = _HERE.parent
_ROOT = _PROJ.parents[2]
for p in (_ROOT, _PROJ, _PROJ / "analysis_scripts", _HERE):
    sys.path.insert(0, str(p))

from shuffle_fold import fold_graph  # noqa: E402
from src.models.detection.YOLO.yolov5 import Detect  # noqa: E402
from reformat_analysis import build_detailed  # noqa: E402
from reformat_classify import classify  # noqa: E402

OUT = _HERE / "outputs" / "reformat"
ONNX_LEAN = OUT / "poc_lean.onnx"
ENGINE_LEAN = OUT / "poc_lean_fp16_detailed.engine"


class OutOnly(nn.Module):
    """只返回 out0（检测结果），丢弃训练用的原始特征图。"""

    def __init__(self, gm: nn.Module) -> None:
        super().__init__()
        self.gm = gm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gm(x)[0]


def grid_float32(ny: int, nx: int, device: torch.device) -> torch.Tensor:
    """与 Detect._grid 数值相同，但直接以 float32 生成（不产生 Cast 节点）。"""
    yv, xv = torch.meshgrid(torch.arange(ny, device=device, dtype=torch.float32),
                            torch.arange(nx, device=device, dtype=torch.float32), indexing="ij")
    return torch.stack((xv, yv), dim=2).view(1, 1, ny, nx, 2)


def count_ops(path: Path) -> dict:
    m = onnx.load(str(path))
    c: dict[str, int] = {}
    for n in m.graph.node:
        c[n.op_type] = c.get(n.op_type, 0) + 1
    return c


def main() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    _, gm, _ = fold_graph()
    gm = gm.cpu().eval()
    x = torch.rand(1, 3, 640, 640)
    with torch.no_grad():
        ref_pred = gm(x)[0]
    lean = OutOnly(gm).eval()

    # 数值检查：只改输出，应完全相同
    with torch.no_grad():
        out_lean = lean(x)
    err_out = float((out_lean - ref_pred).abs().max())
    print(f"A) 只输出 out0：与折叠版逐元素最大误差 {err_out:.3e}", flush=True)

    # B) 网格改为 float32 生成
    orig_grid = Detect._grid
    Detect._grid = staticmethod(grid_float32)
    try:
        with torch.no_grad():
            out_grid = lean(x)
        err_grid = float((out_grid - ref_pred).abs().max())
        print(f"B) float32 网格：与折叠版最大误差 {err_grid:.3e}", flush=True)
        torch.onnx.export(lean, (x,), str(ONNX_LEAN), input_names=["images"], output_names=["out0"],
                          opset_version=17, dynamo=False, do_constant_folding=True)
    finally:
        Detect._grid = orig_grid

    ops_fold = count_ops(OUT.parent / "shuffle_fold" / "sgc_fold.onnx")
    ops_lean = count_ops(ONNX_LEAN)
    print("\nONNX 算子对比（折叠版 → 精简版）：")
    for k in ["Cast", "Transpose", "Reshape", "Concat", "Gather", "Constant", "Sigmoid"]:
        print(f"  {k:<10} {ops_fold.get(k, 0):>4} → {ops_lean.get(k, 0):>4}", flush=True)

    plan = build_detailed(ONNX_LEAN, ENGINE_LEAN) if not ENGINE_LEAN.exists() else ENGINE_LEAN.read_bytes()

    import tensorrt as trt
    logger = trt.Logger(trt.Logger.ERROR)
    eng = trt.Runtime(logger).deserialize_cuda_engine(plan)
    info = json.loads(eng.create_engine_inspector().get_engine_information(trt.LayerInformationFormat.JSON))
    layers = info["Layers"]
    n_float_layers = sum(1 for l in layers if l.get("Outputs") and
                         all(o.get("Format/Datatype", "").startswith("Float") for o in l.get("Outputs", [])))
    n_detect_float = sum(1 for l in layers if l.get("Outputs") and
                         all(o.get("Format/Datatype", "").startswith("Float") for o in l.get("Outputs", []))
                         and any("/detect/" in s for s in l.get("Metadata", "").split("\u001f")))
    print(f"\n精简引擎：层数 {len(layers)}；全 FP32 输出的层 {n_float_layers}（其中 detect 相关 {n_detect_float}）")
    (OUT / "poc_lean_layers.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")

    # 逐层耗时与分类（与 reformat_analysis 同样的方法）
    from trt_deploy_bench import TRTRunner
    from trt_stream_layer_check import LayerProf
    import torch as _t
    runner = TRTRunner(plan)
    _t.manual_seed(0)
    runner.set_input(_t.rand(1, 3, 640, 640, device="cuda"))
    for _ in range(20):
        runner.run()
    _t.cuda.synchronize()
    prof = LayerProf()
    runner.ctx.profiler = prof
    for _ in range(10):
        runner.run()
    _t.cuda.synchronize()
    prof.t.clear()
    prof.cnt.clear()
    for _ in range(50):
        runner.run()
    _t.cuda.synchronize()
    t_layer = {k: prof.t[k] / prof.cnt[k] for k in prof.t}

    cat_ms: dict[str, float] = {}
    cat_n: dict[str, int] = {}
    for l in layers:
        n = l["Name"]
        if not (("reformat" in n.lower()) or ("copynode" in n.lower()) or n.endswith(" copy") or "_copy_output" in n):
            continue
        c = classify(l)
        cat_ms[c] = cat_ms.get(c, 0.0) + t_layer.get(n, 0.0)
        cat_n[c] = cat_n.get(c, 0) + 1
    total = sum(cat_ms.values())
    print(f"\n精简引擎 copy/reformat 合计 {total:.3f} ms（IProfiler，偏高）")
    for c, v in sorted(cat_ms.items(), key=lambda kv: -kv[1]):
        print(f"  {c:<34} 层 {cat_n[c]:>3}  {v:.3f} ms  {100 * v / total if total else 0:5.1f}%")
    summary = {"onnx_ops_fold": ops_fold, "onnx_ops_lean": ops_lean, "err_out_only": err_out,
               "err_float_grid": err_grid, "engine_layers": len(layers), "float_output_layers": n_float_layers,
               "detect_float_output_layers": n_detect_float,
               "copy_reformat_total_ms": total,
               "copy_reformat_by_category": {c: {"layers": cat_n[c], "ms": cat_ms[c]} for c in cat_ms},
               "seconds": round(time.time() - t0)}
    (OUT / "poc_lean_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n已保存 {OUT / 'poc_lean_summary.json'}（总耗时 {time.time() - t0:.0f}s）")


if __name__ == "__main__":
    main()
