"""TensorRT 部署对比（GPU，batch 1，640×640，静态形状）。

流程（对 SGC 无注意力与 YOLOv5s 各做一遍）：
1. 原版 PyTorch（TF32 关闭，作为精度参考）；
2. BN 折叠版 → 导出 ONNX（opset 17，静态形状）；
3. 构建 TensorRT 引擎：FP32 与 FP16；
4. 速度：TRT 引擎 eager-style 逐次同步 / 流水线吞吐，并与 BN 折叠版 PyTorch 的 eager 与 CUDA Graph 同窗对比；
5. 精度：TRT 输出与参考的最大绝对误差，以及相对输出最大值的比例。

用法（仓库根目录）：python experiments/detection/project2_2611/no_train_analysis/trt_deploy_bench.py
输出：outputs/trt/trt_deploy_2026-11-21.json；引擎与 ONNX 保存在 outputs/trt/ 下
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
sys.path.insert(0, str(_PROJ))
sys.path.insert(0, str(_PROJ / "analysis_scripts"))
sys.path.insert(0, str(_HERE))
import redundancy_analysis as ra  # noqa: E402
from bn_fold import fold_bn_inplace  # noqa: E402
from bn_fold_bench import graph_replay  # noqa: E402

OUT = _HERE / "outputs" / "trt"
TRIALS = 3
WARM = 30
EAGER_N = 200
PIPE_N = 200
MODELS = {
    "SGC 无注意力": {"model": "sgc_yolov5s_c3_sgdown_gbot_dw5",
                     "run": "2026-11-17_015_SGC_c3_sgdown_gbot_dw5_s_noDFC_cbamNone_seed42_V1"},
    "YOLOv5s": {"model": "yolov5s", "run": "2026-10-30_002_YOLOv5s_V3"},
}
DT = {trt.float32: torch.float32, trt.float16: torch.float16, trt.int32: torch.int32, trt.int64: torch.int64,
      trt.bool: torch.bool}
TAG = {"SGC 无注意力": "sgc", "YOLOv5s": "yolov5s"}


def flat(o) -> list[torch.Tensor]:
    if torch.is_tensor(o):
        return [o]
    if isinstance(o, (list, tuple)):
        return [t for i in o for t in flat(i)]
    if isinstance(o, dict):
        return [t for v in o.values() for t in flat(v)]
    return []


def _shuffle_export_forward(self, t: torch.Tensor) -> torch.Tensor:
    """与 F.channel_shuffle 等价的 view→transpose→reshape 写法（ONNX 不支持 aten::channel_shuffle）。"""
    n, c, h, w = t.shape
    g = self.groups
    return t.view(n, g, c // g, h, w).transpose(1, 2).reshape(n, c, h, w)


def export_onnx(model: torch.nn.Module, x: torch.Tensor, path: Path) -> int:
    # 1) 校验替代实现与原生 ChannelShuffle 等价
    probe = torch.rand(1, 64, 8, 8, device=x.device)
    m = torch.nn.ChannelShuffle(4)
    diff = float((m(probe) - _shuffle_export_forward(m, probe)).abs().max())
    assert diff == 0.0, f"ChannelShuffle 替代实现不等价，误差 {diff}"
    with torch.no_grad():
        n_out = len(flat(model(x)))
    # 2) 仅在导出期间替换 ChannelShuffle.forward，导出后恢复
    orig_forward = torch.nn.ChannelShuffle.forward
    torch.nn.ChannelShuffle.forward = _shuffle_export_forward
    try:
        torch.onnx.export(model, (x,), str(path), input_names=["images"],
                          output_names=[f"out{i}" for i in range(n_out)],
                          opset_version=17, dynamo=False, do_constant_folding=True)
    finally:
        torch.nn.ChannelShuffle.forward = orig_forward
    return n_out


def build_engine(onnx_path: Path, fp16: bool, plan_path: Path) -> bytes:
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(onnx_path.read_bytes()):
        errs = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        raise RuntimeError("ONNX 解析失败：\n" + "\n".join(errs))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    t0 = time.time()
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("引擎构建失败")
    plan_path.write_bytes(plan)
    print(f"   构建 {'FP16' if fp16 else 'FP32'} 引擎 {time.time() - t0:.0f}s，{plan_path.stat().st_size / 1e6:.1f} MB", flush=True)
    return bytes(plan)


class TRTRunner:
    """预分配输入输出缓冲区，run() 只调用 execute_async_v3（无拷贝）。"""

    def __init__(self, plan: bytes) -> None:
        logger = trt.Logger(trt.Logger.WARNING)
        self.engine = trt.Runtime(logger).deserialize_cuda_engine(plan)
        self.ctx = self.engine.create_execution_context()
        self.inputs: list[tuple[str, torch.Tensor]] = []
        self.outputs: list[tuple[str, torch.Tensor]] = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(name))
            buf = torch.empty(shape, dtype=DT[self.engine.get_tensor_dtype(name)], device="cuda")
            self.ctx.set_tensor_address(name, buf.data_ptr())
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.inputs.append((name, buf))
            else:
                self.outputs.append((name, buf))

    def set_input(self, x: torch.Tensor) -> None:
        self.inputs[0][1].copy_(x)

    def run(self) -> None:
        self.ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)

    def outputs_cpu(self) -> list[torch.Tensor]:
        return [b.float().cpu() for _, b in self.outputs]


def sync_median(fn) -> float:
    ts = []
    for _ in range(EAGER_N):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return float(np.median(ts))


def pipe_ms(fn) -> float:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(PIPE_N):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / PIPE_N


def accuracy(ref: list[torch.Tensor], got: list[torch.Tensor]) -> dict:
    err = max(float((a - b).abs().max()) for a, b in zip(ref, got))
    val = max(float(a.abs().max()) for a in ref)
    return {"max_abs_err": err, "max_abs_value": val, "rel": err / val}


def main() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    torch.manual_seed(0)
    x = torch.rand(1, 3, 640, 640, device=device)
    result: dict[str, dict] = {}

    for label, cfg in MODELS.items():
        tag = TAG[label]
        print(f"\n== {label}", flush=True)
        # 精度参考：原版 PyTorch，TF32 关闭
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        orig = ra.load_model(cfg).eval().to(device)
        with torch.no_grad():
            ref = [t.float().cpu() for t in flat(orig(x))]
        torch.backends.cudnn.allow_tf32 = True  # 与前面速度测试保持默认设置
        torch.backends.cuda.matmul.allow_tf32 = True

        folded = copy.deepcopy(orig).eval()
        n_fold = fold_bn_inplace(folded)
        onnx_path = OUT / f"{tag}_folded.onnx"
        n_out = export_onnx(folded, x, onnx_path)
        print(f"   ONNX 导出：{onnx_path.name}（折叠 {n_fold} 个 block，输出 {n_out} 个张量）", flush=True)

        r: dict = {"folded_blocks": n_fold, "onnx": onnx_path.name}

        # 速度对照：BN 折叠版 PyTorch（eager 与 CUDA Graph）
        def torch_fn(m=folded):
            with torch.no_grad():
                m(x)
        r["pytorch_fold_eager_sync_ms"] = sync_median(torch_fn)
        r["pytorch_fold_eager_pipe_ms"] = pipe_ms(torch_fn)
        r["pytorch_fold_graph_ms"] = graph_replay(folded, x)
        print(f"   PyTorch 折叠版：eager {r['pytorch_fold_eager_sync_ms']:.2f} | "
              f"流水线 {r['pytorch_fold_eager_pipe_ms']:.2f} | Graph {r['pytorch_fold_graph_ms']:.2f} ms", flush=True)

        for prec, fp16 in [("fp32", False), ("fp16", True)]:
            plan_path = OUT / f"{tag}_{prec}.engine"
            plan = build_engine(onnx_path, fp16, plan_path)
            runner = TRTRunner(plan)
            runner.set_input(x)
            runner.run()
            torch.cuda.synchronize()
            acc = accuracy(ref, runner.outputs_cpu())
            for _ in range(WARM):
                runner.run()
            torch.cuda.synchronize()
            r[f"trt_{prec}_accuracy"] = acc
            r[f"trt_{prec}_eager_sync_ms"] = sync_median(runner.run)
            r[f"trt_{prec}_pipe_ms"] = pipe_ms(runner.run)
            print(f"   TRT {prec}：逐次同步 {r[f'trt_{prec}_eager_sync_ms']:.2f} | 流水线 "
                  f"{r[f'trt_{prec}_pipe_ms']:.2f} ms | 最大误差 {acc['max_abs_err']:.3e}"
                  f"（相对 {acc['rel']:.2e}）", flush=True)
            del runner

        result[label] = r
        del orig, folded
        torch.cuda.empty_cache()

    (OUT / "trt_deploy_2026-11-21.json").write_text(
        json.dumps({"results": result, "seconds": round(time.time() - t0)}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    print(f"\n已保存 {OUT / 'trt_deploy_2026-11-21.json'}（总耗时 {time.time() - t0:.0f}s）")


if __name__ == "__main__":
    main()
