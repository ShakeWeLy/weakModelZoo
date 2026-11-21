"""配对 bootstrap：按图像重采样，估计各变体相对 PyTorch 原版的 mAP 差值及 95% 置信区间。

输入：trt_map_eval.py 保存的逐图统计量 outputs/map_eval/stats_{sgc,yolov5s}.pkl
输出：outputs/map_eval/map_bootstrap_2026-11-21.json
"""

from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_PROJ = _HERE.parent
_ROOT = _PROJ.parents[2]
for p in (_ROOT, _PROJ):
    sys.path.insert(0, str(p))
from trainer.evaluator import summarize_stats  # noqa: E402

OUT = _HERE / "outputs" / "map_eval"
B = 300
REF = "PyTorch 原版"
VARIANTS = ["PyTorch BN折叠", "TRT FP32", "TRT FP16"]


def main() -> None:
    t0 = time.time()
    rng = np.random.default_rng(0)
    out: dict[str, dict] = {}
    for tag, label in [("sgc", "SGC 无注意力"), ("yolov5s", "YOLOv5s")]:
        d = pickle.loads((OUT / f"stats_{tag}.pkl").read_bytes())
        stats, names = d["stats"], d["names"]
        n = len(stats[REF])
        print(f"\n== {label}：{n} 张图，B={B}", flush=True)
        diffs = {v: {"map50": [], "map": []} for v in VARIANTS}
        for b in range(B):
            idx = rng.integers(0, n, n)
            ref = summarize_stats([stats[REF][i] for i in idx], names)
            for v in VARIANTS:
                s = summarize_stats([stats[v][i] for i in idx], names)
                diffs[v]["map50"].append(s["map50"] - ref["map50"])
                diffs[v]["map"].append(s["map"] - ref["map"])
        full_ref = summarize_stats(stats[REF], names)
        res = {}
        for v in VARIANTS:
            full = summarize_stats(stats[v], names)
            d50 = np.array(diffs[v]["map50"])
            d95 = np.array(diffs[v]["map"])
            res[v] = {
                "delta_mAP50": full["map50"] - full_ref["map50"],
                "delta_mAP50_CI95": [float(np.percentile(d50, 2.5)), float(np.percentile(d50, 97.5))],
                "delta_mAP50_95": full["map"] - full_ref["map"],
                "delta_mAP50_95_CI95": [float(np.percentile(d95, 2.5)), float(np.percentile(d95, 97.5))],
                "frac_resamples_mAP50_95_below_ref": float((d95 < 0).mean()),
            }
            r = res[v]
            print(f"   {v:<14} ΔmAP50 {r['delta_mAP50']:+.4f} CI95 [{r['delta_mAP50_CI95'][0]:+.4f}, "
                  f"{r['delta_mAP50_CI95'][1]:+.4f}] | ΔmAP50-95 {r['delta_mAP50_95']:+.4f} CI95 "
                  f"[{r['delta_mAP50_95_CI95'][0]:+.4f}, {r['delta_mAP50_95_CI95'][1]:+.4f}]", flush=True)
        out[label] = res

    (OUT / "map_bootstrap_2026-11-21.json").write_text(
        json.dumps({"B": B, "results": out, "seconds": round(time.time() - t0)}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    print(f"\n已保存 {OUT / 'map_bootstrap_2026-11-21.json'}（总耗时 {time.time() - t0:.0f}s）")


if __name__ == "__main__":
    main()
