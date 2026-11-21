"""Nsight Systems 采集目标：只在推理区间内打开 CUDA profiler（cudaProfilerStart/Stop）。

用法（由 shuffle_ablation_nsys.py 调用，也可手动执行）：
  nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop -t cuda -o <out> \
      python shuffle_ablation_nsys_driver.py <engine_path> <n_iters>
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from trt_deploy_bench import TRTRunner  # noqa: E402


def main() -> None:
    engine = Path(sys.argv[1])
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    runner = TRTRunner(engine.read_bytes())
    torch.manual_seed(0)
    runner.set_input(torch.rand(1, 3, 640, 640, device="cuda"))
    for _ in range(30):
        runner.run()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    for _ in range(n):
        runner.run()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
    print(f"done {engine.name} x{n}", flush=True)


if __name__ == "__main__":
    main()
