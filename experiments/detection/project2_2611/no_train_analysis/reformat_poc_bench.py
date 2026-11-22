"""Reformat 概念验证的速度对比：三个引擎在同一会话内交替测量（逐次同步与流水线）。

- 置换折叠 FP16（非 DETAILED，与前一轮一致）
- 置换折叠 FP16（DETAILED，用于层级分析的构建，作为 DETAILED 本身是否影响速度的对照）
- 精简 FP16（仅 out0 + float32 网格，DETAILED）

用法（仓库根目录）：python experiments/detection/project2_2611/no_train_analysis/reformat_poc_bench.py
输出：outputs/reformat/poc_bench.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from trt_deploy_bench import TRTRunner  # noqa: E402
from shuffle_ablation_bench import pipe_ms, sync_median  # noqa: E402

OUT = _HERE / "outputs" / "reformat"
ENGINES = {
    "置换折叠 FP16（非 DETAILED）": _HERE / "outputs" / "shuffle_fold" / "sgc_fold_fp16.engine",
    "置换折叠 FP16（DETAILED）": OUT / "sgc_fold_fp16_detailed.engine",
    "精简 FP16（out0 + float 网格，DETAILED）": OUT / "poc_lean_fp16_detailed.engine",
}
TRIALS = 3


def main() -> None:
    t0 = time.time()
    torch.manual_seed(0)
    x = torch.rand(1, 3, 640, 640, device="cuda")
    runners = {}
    for label, p in ENGINES.items():
        r = TRTRunner(p.read_bytes())
        r.set_input(x)
        for _ in range(30):
            r.run()
        torch.cuda.synchronize()
        runners[label] = r
    res = {k: {"sync": [], "pipe": []} for k in ENGINES}
    for _ in range(TRIALS):
        for label, r in runners.items():
            res[label]["sync"].append(sync_median(r.run))
            res[label]["pipe"].append(pipe_ms(r.run))
    out = {}
    for label in ENGINES:
        out[label] = {"sync_median_ms": float(np.median(res[label]["sync"])),
                      "sync_trials": res[label]["sync"],
                      "pipe_median_ms": float(np.median(res[label]["pipe"])),
                      "pipe_trials": res[label]["pipe"]}
        print(f"{label:<40} 逐次同步 {out[label]['sync_median_ms']:.3f} ms {[round(v, 3) for v in res[label]['sync']]}"
              f" | 流水线 {out[label]['pipe_median_ms']:.3f} ms", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "poc_bench.json").write_text(json.dumps({"results": out, "seconds": round(time.time() - t0)},
                                                   ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"已保存 {OUT / 'poc_bench.json'}")


if __name__ == "__main__":
    main()
