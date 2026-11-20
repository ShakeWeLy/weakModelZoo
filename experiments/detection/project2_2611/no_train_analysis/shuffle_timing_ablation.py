"""仅用于计时的消融：把 ChannelShuffle 替换为恒等映射，测 GPU 与 eager 时间差（不影响训练，不改模型文件）。

注意：替换后模型输出不正确，只用于估计 shuffle 带来的 GPU 与发射开销。
用法（仓库根目录）：python experiments/detection/project2_2611/no_train_analysis/shuffle_timing_ablation.py
输出：outputs/launch_copy/shuffle_ablation_2026-11-20.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

_PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJ))
sys.path.insert(0, str(_PROJ / "analysis_scripts"))
import redundancy_analysis as ra  # noqa: E402

OUT = Path(__file__).resolve().parent / "outputs" / "launch_copy"
WARM = 20
EAGER_N = 100
GRAPH_N = 200
MODELS = {
    "YOLOv5s": {"model": "yolov5s", "run": "2026-10-30_002_YOLOv5s_V3"},
    "SGC 无注意力": {"model": "sgc_yolov5s_c3_sgdown_gbot_dw5",
                     "run": "2026-11-17_015_SGC_c3_sgdown_gbot_dw5_s_noDFC_cbamNone_seed42_V1"},
}


def eager(model, x) -> float:
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


def graph(model, x) -> float:
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


def main() -> None:
    device = torch.device("cuda")
    x = torch.rand(1, 3, 640, 640, device=device)
    res = {}
    orig_forward = torch.nn.ChannelShuffle.forward
    for label, cfg in MODELS.items():
        for tag in ["with_shuffle", "identity_shuffle"]:
            # 每次都显式设置，避免补丁残留到下一组实验
            if tag == "identity_shuffle":
                torch.nn.ChannelShuffle.forward = lambda self, t: t  # 仅本进程内，计时用
            else:
                torch.nn.ChannelShuffle.forward = orig_forward
            model = ra.load_model({"model": cfg["model"], "run": cfg["run"]}).eval().to(device)
            with torch.no_grad():
                for _ in range(WARM):
                    model(x)
            torch.cuda.synchronize()
            e_ms = eager(model, x)
            g_ms = graph(model, x)
            res[f"{label} | {tag}"] = {"eager_sync_ms": e_ms, "graph_replay_ms": g_ms}
            print(f"{label:<12} {tag:<17} eager {e_ms:7.2f} ms | graph {g_ms:6.2f} ms", flush=True)
            del model
            torch.cuda.empty_cache()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "shuffle_ablation_2026-11-20.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print("已保存", OUT / "shuffle_ablation_2026-11-20.json")


if __name__ == "__main__":
    main()
