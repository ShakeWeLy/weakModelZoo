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

_PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJ))
sys.path.insert(0, str(_PROJ / "analysis_scripts"))
import redundancy_analysis as ra  # noqa: E402

OUT = Path(__file__).resolve().parent / "outputs" / "module_timing"
IMG = 640
ITERS = 200
WARM = 30
MODELS = {
    "YOLOv5s": {"model": "yolov5s", "run": "2026-10-30_002_YOLOv5s_V3"},
    "SGC 无注意力": {"model": "sgc_yolov5s_c3_sgdown_gbot_dw5",
                     "run": "2026-11-17_015_SGC_c3_sgdown_gbot_dw5_s_noDFC_cbamNone_seed42_V1"},
}


def total_ms(model, x) -> tuple[float, float]:
    with torch.no_grad():
        for _ in range(WARM):
            model(x)
    torch.cuda.synchronize()
    ts = []
    with torch.no_grad():
        for _ in range(ITERS):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            model(x)
            e.record()
            torch.cuda.synchronize()
            ts.append(s.elapsed_time(e))
    a = np.array(ts)
    return float(np.median(a)), float(np.percentile(a, 90))


def module_ms(model, x, names: list[str]) -> dict[str, float]:
    mods = dict(model.named_modules())
    with torch.no_grad():
        for _ in range(WARM):
            model(x)
    torch.cuda.synchronize()
    cur: dict = {}
    rec: dict[str, list] = {n: [] for n in names}
    handles = []
    for n in names:
        def pre(m, i, n=n):
            e = torch.cuda.Event(enable_timing=True)
            e.record()
            cur[n] = e

        def post(m, i, o, n=n):
            e = torch.cuda.Event(enable_timing=True)
            e.record()
            rec[n].append((cur.pop(n), e))

        handles.append(mods[n].register_forward_pre_hook(pre))
        handles.append(mods[n].register_forward_hook(post))
    with torch.no_grad():
        for _ in range(ITERS):
            model(x)
    torch.cuda.synchronize()
    for h in handles:
        h.remove()
    return {n: float(np.median([s.elapsed_time(e) for s, e in rec[n]])) for n in names}


def main() -> None:
    t0 = time.time()
    print(f"GPU：{torch.cuda.get_device_name(0)}；torch {torch.__version__}")
    device = torch.device("cuda")
    x = torch.rand(1, 3, IMG, IMG, device=device)
    out: dict[str, dict] = {}
    for label, spec in MODELS.items():
        model = ra.load_model(spec).eval().to(device)
        tot_med, tot_p90 = total_ms(model, x)
        top = ["backbone", "neck", "detect"]
        sub = [f"backbone.{n}" for n, _ in model.backbone.named_children()] + \
              [f"neck.{n}" for n, _ in model.neck.named_children()]
        per = module_ms(model, x, top + sub)
        out[label] = {"total_ms": tot_med, "total_p90_ms": tot_p90, "modules_ms": per,
                      "backbone_children": [f"backbone.{n}" for n, _ in model.backbone.named_children()],
                      "neck_children": [f"neck.{n}" for n, _ in model.neck.named_children()]}
        s = sum(per[t] for t in top)
        print(f"\n== {label}：整网 {tot_med:.2f} ms（P90 {tot_p90:.2f}）；Backbone+Neck+Detect 之和 {s:.2f} ms")
        for t in top:
            print(f"   {t:<10} {per[t]:7.2f} ms  {100 * per[t] / tot_med:5.1f}%")
        for n in sub:
            print(f"     └ {n:<20} {per[n]:7.2f} ms  {100 * per[n] / tot_med:5.1f}%")
        del model
        torch.cuda.empty_cache()

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "module_timing_2026-11-20.json").write_text(
        json.dumps({"iters": ITERS, "warmup": WARM, "results": out, "seconds": round(time.time() - t0)},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n已保存 {OUT / 'module_timing_2026-11-20.json'}（总耗时 {time.time() - t0:.0f}s）")


if __name__ == "__main__":
    main()
