"""算子级剖析：为什么 FLOPs 更少的 Ghost 模型推理反而更慢（batch=1，640×640）。

输出每个模型的：叶子模块数、每次前向的算子调用数、耗时最大的算子（按每次前向的设备 / CPU 时间）。
不训练、不改权重。

用法（仓库根目录）：
    python experiments/detection/project2_2611/profile_ops.py
输出：profile_ops/ops.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import redundancy_analysis as ra  # noqa: E402
from diagnose_p4_cbam import MODELS  # noqa: E402

OUT = Path(__file__).resolve().parent / "profile_ops"
ITERS = 20


def leaf_modules(model: torch.nn.Module) -> int:
    return sum(1 for m in model.modules() if len(list(m.children())) == 0)


def time_of(e, device: torch.device) -> float:
    if device.type == "cuda":
        v = getattr(e, "self_device_time_total", None) or getattr(e, "self_cuda_time_total", 0.0)
    else:
        v = getattr(e, "self_cpu_time_total", 0.0)
    return float(v) / ITERS / 1000.0  # 每次前向，毫秒


def run(key: str, device: torch.device) -> dict:
    spec = MODELS[key]
    model = ra.load_model(spec).eval().to(device)
    x = torch.rand(1, 3, 640, 640, device=device)
    acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if device.type == "cuda" else [])
    with torch.no_grad():
        for _ in range(10):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        with profile(activities=acts) as prof:
            for _ in range(ITERS):
                model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
    evs = prof.key_averages()
    calls = sum(e.count for e in evs) / ITERS
    top = sorted(evs, key=lambda e: time_of(e, device), reverse=True)[:12]
    rows = [{"op": e.key, "calls_per_fwd": e.count / ITERS, "ms_per_fwd": time_of(e, device)} for e in top]
    return {"model": spec["label"], "device": device.type, "leaf_modules": leaf_modules(model),
            "ops_per_fwd": calls, "top": rows}


def main() -> None:
    devices = [torch.device("cuda")] if torch.cuda.is_available() else []
    devices.append(torch.device("cpu"))
    results = []
    for dev in devices:
        for key in ("yolo", "ours", "c3topo"):
            r = run(key, dev)
            results.append(r)
            print(f"\n[{r['device'].upper()}] {r['model']}：叶子模块 {r['leaf_modules']}，每次前向算子调用 {r['ops_per_fwd']:.0f} 次")
            for row in r["top"][:10]:
                print(f"    {row['ms_per_fwd']:7.2f} ms  {row['calls_per_fwd']:6.1f} 次  {row['op']}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "ops.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n已保存 {OUT / 'ops.json'}")


if __name__ == "__main__":
    main()
