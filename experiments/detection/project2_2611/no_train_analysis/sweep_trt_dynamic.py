"""不同输入尺寸与 Batch Size 下的 TRT FP16 延迟扫描（动态形状引擎）。

用法（仓库根目录）：
  python experiments/detection/project2_2611/no_train_analysis/sweep_trt_dynamic.py <variant>
variant：
  sgc_original   BN 折叠的 SGC（未做置换折叠）
  sgc_perm       置换折叠的 SGC（与 shuffle_fold 等价）
  yolov5s        BN 折叠的 YOLOv5s（对照）

流程：导出动态 ONNX（batch / H / W 可变）→ 构建动态 FP16 引擎（profile：min 1×320² ，opt 4×640²，max 8×800²）
→ 对 {320,416,512,640,800} × batch {1,2,4,8} 逐一测量（逐次同步中位数、流水线吞吐）。
输出：outputs/sweep/<variant>_dynamic_fp16.engine，outputs/sweep/<variant>_trt.json
"""

from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import tensorrt as trt
import torch

_HERE = Path(__file__).resolve().parent
_PROJ = _HERE.parent
_ROOT = _PROJ.parents[2]
for p in (_ROOT, _PROJ, _PROJ / "analysis_scripts", _HERE):
    sys.path.insert(0, str(p))

import redundancy_analysis as ra  # noqa: E402
from bn_fold import fold_bn_inplace  # noqa: E402
from shuffle_fold import build, fold_graph  # noqa: E402

OUT = _HERE / "outputs" / "sweep"
SIZES = [320, 416, 512, 640, 800]
BATCHES = [1, 2, 4, 8]
N_SYNC = 50
N_PIPE = 50
DT = {trt.float32: torch.float32, trt.float16: torch.float16, trt.int32: torch.int32, trt.int64: torch.int64}


def load_variant(name: str) -> torch.nn.Module:
    if name == "sgc_original":
        _, _, model = build()          # build() 已完成 BN 折叠，返回的 model 为原始结构
        return model.eval()
    if name == "sgc_perm":
        _, gm, _ = fold_graph()
        return gm.eval()
    if name == "yolov5s":
        m = ra.load_model({"model": "yolov5s", "run": "2026-10-30_002_YOLOv5s_V3"}).eval()
        fold_bn_inplace(m)
        return m
    raise ValueError(name)


def export_dynamic(model: torch.nn.Module, path: Path) -> int:
    model = model.cpu().eval()
    x = torch.rand(1, 3, 640, 640)
    with torch.no_grad():
        n_out = len(model(x))
    dyn = {"images": {0: "batch", 2: "height", 3: "width"}}
    for i in range(n_out):
        dyn[f"out{i}"] = {0: "batch"}
    dyn["out0"][1] = "anchors"
    for i in range(1, n_out):
        dyn[f"out{i}"].update({2: f"h{i}", 3: f"w{i}"})
    # ONNX 不支持 aten::channel_shuffle：导出期间替换为等价的 view/transpose/reshape（与 trt_deploy_bench 相同）
    from trt_deploy_bench import _shuffle_export_forward
    orig_forward = torch.nn.ChannelShuffle.forward
    torch.nn.ChannelShuffle.forward = _shuffle_export_forward
    try:
        torch.onnx.export(model, (x,), str(path), input_names=["images"],
                          output_names=[f"out{i}" for i in range(n_out)],
                          dynamic_axes=dyn, opset_version=17, dynamo=False, do_constant_folding=True)
    finally:
        torch.nn.ChannelShuffle.forward = orig_forward
    return n_out


def build_dynamic(onnx_path: Path, plan_path: Path) -> bytes:
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
    prof = builder.create_optimization_profile()
    # YOLOv5s 的动态引擎在 max=8×800² 时 TRT 构建 LLVM 内存溢出，因此上界收紧为 8×640²（见报告说明）
    max_hw = 640 if "yolov5s" in onnx_path.name else 800
    prof.set_shape("images", (1, 3, 320, 320), (4, 3, 640, 640), (8, 3, max_hw, max_hw))
    config.add_optimization_profile(prof)
    t0 = time.time()
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("动态引擎构建失败")
    plan_path.write_bytes(plan)
    print(f"动态引擎构建 {time.time() - t0:.0f}s，{plan_path.stat().st_size / 1e6:.1f} MB", flush=True)
    return bytes(plan)


class DynRunner:
    def __init__(self, plan: bytes) -> None:
        logger = trt.Logger(trt.Logger.ERROR)
        self.eng = trt.Runtime(logger).deserialize_cuda_engine(plan)
        self.ctx = self.eng.create_execution_context()
        self.out_names = [self.eng.get_tensor_name(i) for i in range(self.eng.num_io_tensors)
                          if self.eng.get_tensor_mode(self.eng.get_tensor_name(i)) == trt.TensorIOMode.OUTPUT]

    def configure(self, b: int, h: int, w: int) -> None:
        self.ctx.set_input_shape("images", (b, 3, h, w))
        self.x = torch.rand(b, 3, h, w, device="cuda")
        self.ctx.set_tensor_address("images", self.x.data_ptr())
        self.outs = []
        for name in self.out_names:
            shp = tuple(self.ctx.get_tensor_shape(name))
            t = torch.empty(shp, dtype=DT[self.eng.get_tensor_dtype(name)], device="cuda")
            self.ctx.set_tensor_address(name, t.data_ptr())
            self.outs.append(t)

    def run(self) -> None:
        self.ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)


def main() -> None:
    variant = sys.argv[1]
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    torch.backends.cudnn.allow_tf32 = False
    onnx_path = OUT / f"{variant}_dynamic.onnx"
    plan_path = OUT / f"{variant}_dynamic_fp16.engine"
    if not plan_path.exists():
        model = load_variant(variant)
        n_out = export_dynamic(model, onnx_path)
        print(f"动态 ONNX 导出完成：{onnx_path.name}（输出 {n_out} 个）", flush=True)
        build_dynamic(onnx_path, plan_path)
    plan = plan_path.read_bytes()
    runner = DynRunner(plan)

    rows = []
    for s in SIZES:
        for b in BATCHES:
            runner.configure(b, s, s)
            for _ in range(10):
                runner.run()
            torch.cuda.synchronize()
            ts = []
            for _ in range(N_SYNC):
                a = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
                a.record()
                runner.run()
                e.record()
                torch.cuda.synchronize()
                ts.append(a.elapsed_time(e))
            sync_ms = float(np.median(ts))
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            for _ in range(N_PIPE):
                runner.run()
            torch.cuda.synchronize()
            pipe_ms = (time.perf_counter() - t1) * 1000.0 / N_PIPE
            rows.append({"size": s, "batch": b, "sync_ms": sync_ms, "pipe_ms": pipe_ms,
                         "pipe_img_per_s": b * 1000.0 / pipe_ms, "pipe_ms_per_img": pipe_ms / b})
            print(f"{variant:<13} {s}x{s} b{b}: 逐次同步 {sync_ms:7.2f} ms | 流水线 {pipe_ms:7.2f} ms "
                  f"({b * 1000.0 / pipe_ms:6.1f} img/s)", flush=True)
    (OUT / f"{variant}_trt.json").write_text(json.dumps({"variant": variant, "rows": rows,
                                                        "seconds": round(time.time() - t0)},
                                                       ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"已保存 {OUT / f'{variant}_trt.json'}（总耗时 {time.time() - t0:.0f}s）", flush=True)


if __name__ == "__main__":
    main()
