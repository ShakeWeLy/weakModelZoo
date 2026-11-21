"""ChannelShuffle 改为在 __init__ 中创建前后的对比（GPU，batch 1，640×640，eager 与 CUDA Graph）。

用法（仓库根目录）：
  python experiments/detection/project2_2611/no_train_analysis/shuffle_module_bench.py before
  （修改代码）
  python experiments/detection/project2_2611/no_train_analysis/shuffle_module_bench.py after

before：保存参考输出（固定随机输入）与 state_dict 键，并测速。
after ：检查输出是否与参考一致、state_dict 键是否不变，并测速。
YOLOv5s 不含 ChannelShuffle，作为对照（同一时间窗内的漂移参考）。
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

OUT = Path(__file__).resolve().parent / "outputs" / "shuffle_module"
REF = OUT / "ref_output.pt"
TRIALS = 3
WARM = 30
EAGER_N = 200
PIPE_N = 200
GRAPH_N = 200
MODELS = {
    "SGC 无注意力": {"model": "sgc_yolov5s_c3_sgdown_gbot_dw5",
                     "run": "2026-11-17_015_SGC_c3_sgdown_gbot_dw5_s_noDFC_cbamNone_seed42_V1"},
    "YOLOv5s（对照）": {"model": "yolov5s", "run": "2026-10-30_002_YOLOv5s_V3"},
}


def flat(o) -> list[torch.Tensor]:
    if torch.is_tensor(o):
        return [o]
    if isinstance(o, (list, tuple)):
        return [t for i in o for t in flat(i)]
    if isinstance(o, dict):
        return [t for v in o.values() for t in flat(v)]
    return []


def eager_sync(model, x) -> float:
    ts = []
    with torch.no_grad():
        for _ in range(EAGER_N):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            model(x)
            e.record()
            torch.cuda.synchronize()
            ts.append(s.elapsed_time(e))
    return float(np.median(ts))


def eager_pipe(model, x) -> float:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(PIPE_N):
            model(x)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / PIPE_N


def graph_replay(model, x) -> float:
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s), torch.no_grad():
        for _ in range(5):
            model(x)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.no_grad(), torch.cuda.graph(g):
        model(x)
    for _ in range(WARM):
        g.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(GRAPH_N):
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b))
    return float(np.median(ts))


def measure(model, x) -> dict:
    with torch.no_grad():
        for _ in range(WARM):
            model(x)
    torch.cuda.synchronize()
    return {"eager_sync_ms": eager_sync(model, x),
            "eager_pipe_ms": eager_pipe(model, x),
            "graph_ms": graph_replay(model, x)}


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in ("before", "after"):
        raise SystemExit("用法：shuffle_module_bench.py before|after")
    tag = sys.argv[1]
    device = torch.device("cuda")
    OUT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    x = torch.rand(1, 3, 640, 640, device=device)

    sgc = ra.load_model(MODELS["SGC 无注意力"]).eval().to(device)
    keys = sorted(sgc.state_dict().keys())
    with torch.no_grad():
        out = [t.detach().cpu() for t in flat(sgc(x))]
    if tag == "before":
        torch.save({"keys": keys, "out": out}, REF)
        print(f"已保存参考输出与 state_dict 键（{len(keys)} 个键，{len(out)} 个输出张量）→ {REF}")
    else:
        ref = torch.load(REF, weights_only=False)
        assert ref["keys"] == keys, "state_dict 键发生变化"
        assert len(ref["out"]) == len(out), "输出张量数量变化"
        diffs = [float((a - b).abs().max()) for a, b in zip(ref["out"], out)]
        print(f"state_dict 键一致：{len(keys)} 个；输出最大绝对误差 {max(diffs):.3e}")
    del sgc
    torch.cuda.empty_cache()

    res: dict[str, list[dict]] = {k: [] for k in MODELS}
    for trial in range(TRIALS):
        for label, cfg in MODELS.items():
            model = ra.load_model(cfg).eval().to(device)
            r = measure(model, x)
            res[label].append(r)
            print(f"[{tag} trial{trial + 1}] {label:<14} eager逐次同步 {r['eager_sync_ms']:6.2f} | "
                  f"eager流水线 {r['eager_pipe_ms']:6.2f} | Graph {r['graph_ms']:5.2f} ms", flush=True)
            del model
            torch.cuda.empty_cache()

    summary = {}
    for label, rows in res.items():
        summary[label] = {k: {"median_of_trials": float(np.median([r[k] for r in rows])),
                              "trials": [r[k] for r in rows]}
                          for k in ["eager_sync_ms", "eager_pipe_ms", "graph_ms"]}
    (OUT / f"bench_{tag}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"已保存 {OUT / f'bench_{tag}.json'}")


if __name__ == "__main__":
    main()
