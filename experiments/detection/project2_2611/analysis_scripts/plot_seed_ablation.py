"""汇总图：消融结果 vs 多 seed 噪声（test 集）。数据来自 size_ap/multi_2026-11-17.json。

左：整体 mAP50；右：<24² AP50（主指标）。本模型与 YOLOv5s 为 3 个 seed，消融为 seed 42 单次。
用法（仓库根目录）：python experiments/detection/project2_2611/plot_seed_ablation.py
输出：size_ap/seed_ablation_2026-11-17.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
RES = json.loads((HERE / "size_ap" / "multi_2026-11-17.json").read_text(encoding="utf-8"))["results"]
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

GROUPS = [
    (0, "本模型\n（3 seeds）", ["ours_s42(V1)", "ours_s43", "ours_s44"]),
    (1, "去CBAM", ["去CBAM(s42)"]),
    (2, "P3去SAM", ["P3去SAM(s42)"]),
    (3, "去DFC", ["去DFC(s42)"]),
    (4, "残差恒等", ["残差恒等(s42)"]),
    (5, "YOLOv5s\n（3 seeds）", ["YOLOv5s_s42(V3)", "YOLOv5s_s43", "YOLOv5s_s44"]),
]


def value(name: str, metric: str) -> float:
    r = RES[name]
    return r["overall"]["map50"] if metric == "map50" else r["buckets"]["<24²"]["ap50"]


def draw(ax, metric: str, title: str) -> None:
    ours = [value(n, metric) for n in GROUPS[0][2]]
    yolo = [value(n, metric) for n in GROUPS[5][2]]
    ours_mean, yolo_mean = float(np.mean(ours)), float(np.mean(yolo))
    ax.axhspan(min(ours), max(ours), color="#1f77b4", alpha=0.08, zorder=0)
    ax.axhline(ours_mean, color="#1f77b4", ls="--", lw=1, zorder=1)
    ax.axhline(yolo_mean, color="#2ca02c", ls=":", lw=1, zorder=1)
    jit = [-0.08, 0.0, 0.08]
    for x, label, names in GROUPS:
        vals = [value(n, metric) for n in names]
        if len(vals) == 1:
            ax.scatter([x], vals, s=70, color="#d62728", zorder=3)
            ax.annotate(f"{vals[0]:.3f}", (x, vals[0]), textcoords="offset points", xytext=(10, -3), fontsize=8)
        else:
            color = "#1f77b4" if x == 0 else "#2ca02c"
            ax.scatter([x + j for j in jit], vals, s=50, color=color, zorder=3)
            m = float(np.mean(vals))
            ax.plot([x - 0.22, x + 0.22], [m, m], color="black", lw=2, zorder=4)
            ax.annotate(f"均值 {m:.3f}", (x, m), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=8)
    ax.set_xticks([g[0] for g in GROUPS])
    ax.set_xticklabels([g[1] for g in GROUPS], fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.grid(axis="y", alpha=0.3)


def main() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    draw(axes[0], "map50", "整体 mAP50（test，1501 张）")
    draw(axes[1], "small", "<24² AP50（test，1362 个 GT，占 91%）")
    axes[0].set_ylabel("AP50")
    fig.suptitle("蓝色：本模型 3 seeds（阴影为范围，虚线为均值）；绿色：YOLOv5s 3 seeds（点线为均值）；"
                 "红色：消融（seed 42 单次）", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = HERE / "size_ap" / "seed_ablation_2026-11-17.png"
    fig.savefig(out, dpi=120)
    print(f"已保存 {out}")


if __name__ == "__main__":
    main()
