"""Ghost 本征占比（intrinsic_ratio）消融的通道冗余对比。

对比 本征占比 r = 0.5（基线，ratio=2） / 0.625 / 0.75 × seed 42/43/44 共 9 个 run，
统计 stage2~stage5 下采样模块（backbone.stageK.0，SGhostConv）的输出冗余。

与 redundancy_analysis.py 的口径一致（同一批测试图、同一套 |r| 统计），并额外给出：
  - 本征-幽灵的父组映射：幽灵分支 groups = gcd(本征, 幽灵)，第 j 个幽灵由第 g = j // out_pg 组的
    in_pg 个本征通道生成；r=0.5 时退化为深度卷积（每个幽灵只有一个父通道）。
  - copy_frac：与任一本征通道 |r|>0.9 的幽灵比例（“复制”比例）。
  - best_par：幽灵与所有本征通道的最大 |r| 的均值。
  - R2_group：幽灵被其父组线性回归解释的方差比例（单父时等于 r²）。
  - dup_gg：幽灵之间的近似重复比例。
注意：本征通道数随 r 变化，所以本征-only 指标（NN_int、mean_int、dup_int）与通道数耦合，只作参考；
全部通道指标在三个 r 下通道数相同，可直接比较。

输出：experiments/detection/project2_2611/redundancy_ir/
用法（仓库根目录）：
  python experiments/detection/project2_2611/redundancy_ir_analysis.py --num 20
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import redundancy_analysis as ra  # noqa: E402
from trainer.config import ROOT  # noqa: E402
from trainer.data import YoloDetDataset  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUT = Path(__file__).resolve().parent / "redundancy_ir"
STAGES = [2, 3, 4, 5]
NAMES = [f"backbone.stage{k}.0" for k in STAGES]
BASE = "sgc_yolov5s_c3_sgdown_gbot_dw5"

# 每个占比下的 3 个 seed：model = configs/models 下的 TOML 名，run = runs 下的目录名
RUNS = {
    0.5: [
        {"seed": 42, "model": BASE, "run": "2026-11-13_029_SGC_c3_sgdown_gbot_dw5_s_V1"},
        {"seed": 43, "model": f"{BASE}_seed43", "run": "2026-11-17_001_SGC_c3_sgdown_gbot_dw5_s_seed43_V1"},
        {"seed": 44, "model": f"{BASE}_seed44", "run": "2026-11-17_002_SGC_c3_sgdown_gbot_dw5_s_seed44_V1"},
    ],
    0.625: [
        {"seed": 42, "model": f"{BASE}_ir0625", "run": "2026-11-17_009_SGC_c3_sgdown_gbot_dw5_s_ir0625_seed42_V1"},
        {"seed": 43, "model": f"{BASE}_ir0625_seed43", "run": "2026-11-17_010_SGC_c3_sgdown_gbot_dw5_s_ir0625_seed43_V1"},
        {"seed": 44, "model": f"{BASE}_ir0625_seed44", "run": "2026-11-17_011_SGC_c3_sgdown_gbot_dw5_s_ir0625_seed44_V1"},
    ],
    0.75: [
        {"seed": 42, "model": f"{BASE}_ir075", "run": "2026-11-17_012_SGC_c3_sgdown_gbot_dw5_s_ir075_seed42_V1"},
        {"seed": 43, "model": f"{BASE}_ir075_seed43", "run": "2026-11-17_013_SGC_c3_sgdown_gbot_dw5_s_ir075_seed43_V1"},
        {"seed": 44, "model": f"{BASE}_ir075_seed44", "run": "2026-11-17_014_SGC_c3_sgdown_gbot_dw5_s_ir075_seed44_V1"},
    ],
}
# 原版 YOLOv5s（Conv↓，无 Ghost）3 seed 参照：与 record.md 的 YOLOv5s 三个 seed 一致
YOLO_RUNS = [
    {"label": "YOLOv5s", "seed": 42, "model": "yolov5s", "run": "2026-10-30_002_YOLOv5s_V3"},
    {"label": "YOLOv5s", "seed": 43, "model": "yolov5s_seed43", "run": "2026-11-17_003_YOLOv5s_seed43_V1"},
    {"label": "YOLOv5s", "seed": 44, "model": "yolov5s_seed44", "run": "2026-11-17_004_YOLOv5s_seed44_V1"},
]
METRICS = ["NN_all", "mean_all", "dup_all", "NN_int", "mean_int", "dup_int",
           "copy_frac", "best_par_abs", "R2_group", "dup_gg"]


def _nanmax_rows(M: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmax(M, axis=1)


def group_parents(m: nn.Module) -> tuple[int, int, list[list[int]]]:
    """返回 (本征通道数, 幽灵通道数, 每个幽灵的父通道列表)。"""
    cheap = next(x for x in m.cheap_conv.modules() if isinstance(x, nn.Conv2d))
    g = cheap.groups
    n_int, n_gh = int(m.identity_num), int(m.cheap_num)
    in_pg, out_pg = n_int // g, n_gh // g
    parents = []
    for j in range(n_gh):
        start = (j // out_pg) * in_pg
        parents.append(list(range(start, start + in_pg)))
    return n_int, n_gh, parents


def per_image_metrics(cap: dict, n: str, parents: list[list[int]], n_int: int, n_gh: int) -> dict:
    I = ra.flat(cap[f"{n}.conv"])          # 本征（洗牌前）
    G = ra.flat(cap[f"{n}.cheap_conv"])    # 幽灵（洗牌前）
    F = np.concatenate([I, G], axis=0)     # 与 SGhostConv 拼接后的通道集合（置换不改变统计）
    R = ra.corr_matrix(F)
    Ri = ra.corr_matrix(I)
    RG = np.abs(ra.corr_matrix(G))
    np.fill_diagonal(RG, np.nan)
    RIG = np.abs(R[n_int:, :n_int])        # 幽灵 × 本征
    best_par = _nanmax_rows(RIG)
    gg = _nanmax_rows(RG)
    P = I.shape[1]
    r2 = []
    for j, pj in enumerate(parents):
        y = G[j]
        X = np.column_stack([np.ones(P), I[pj].T])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        res = y - X @ beta
        sst = float(((y - y.mean()) ** 2).sum())
        r2.append(1.0 - float((res ** 2).sum()) / sst if sst > 1e-10 else np.nan)
    return {
        "NN_all": ra.nn_abs(R), "mean_all": ra.mean_abs_pair(R), "dup_all": ra.dup_frac(R),
        "NN_int": ra.nn_abs(Ri) if n_int > 1 else np.nan,
        "mean_int": ra.mean_abs_pair(Ri) if n_int > 1 else np.nan,
        "dup_int": ra.dup_frac(Ri) if n_int > 1 else np.nan,
        "copy_frac": float(np.nanmean(best_par > ra.DUP_THR)),
        "best_par_abs": float(np.nanmean(best_par)),
        "R2_group": float(np.nanmean(r2)),
        "dup_gg": float(np.nanmean(_nanmax_rows(RG) > ra.DUP_THR)) if n_gh > 1 else np.nan,
    }


def analyze_run(spec: dict, ds: YoloDetDataset, picks: list[int]) -> tuple[dict, nn.Module]:
    model = ra.load_model(spec)
    mods = dict(model.named_modules())
    info = {n: group_parents(mods[n]) for n in NAMES}
    need = [k for n in NAMES for k in (f"{n}.conv", f"{n}.cheap_conv")]
    res: dict[str, list[dict]] = {n: [] for n in NAMES}
    for idx in picks:
        img_t, _ = ds[idx]
        cap: dict = {}
        handles = [mods[k].register_forward_hook(ra.make_hook(cap, k)) for k in need]
        with torch.no_grad():
            model(img_t[None].float() / 255.0)
        for h in handles:
            h.remove()
        for n in NAMES:
            n_int, n_gh, parents = info[n]
            res[n].append(per_image_metrics(cap, n, parents, n_int, n_gh))
    rows = {}
    for n, st in zip(NAMES, STAGES):
        row = {k: float(np.nanmean([d[k] for d in res[n]])) for k in METRICS}
        row.update({"stage": st, "C": info[n][0] + info[n][1], "n_int": info[n][0], "n_gh": info[n][1]})
        rows[st] = row
    return rows, model


def corr_example(model: nn.Module, img_t: torch.Tensor, n: str) -> np.ndarray:
    mods = dict(model.named_modules())
    cap: dict = {}
    handles = [mods[k].register_forward_hook(ra.make_hook(cap, k)) for k in (f"{n}.conv", f"{n}.cheap_conv")]
    with torch.no_grad():
        model(img_t[None].float() / 255.0)
    for h in handles:
        h.remove()
    F = np.concatenate([ra.flat(cap[f"{n}.conv"]), ra.flat(cap[f"{n}.cheap_conv"])], axis=0)
    return np.abs(ra.corr_matrix(F))


def plot_metrics(all_rows: dict, out: Path, yolo: dict | None = None) -> None:
    ratios = sorted(RUNS)
    panels = [("NN_all", "NN|r|（全部通道，越高越冗余）"),
              ("dup_all", f"近似重复比例（|r|>{ra.DUP_THR}，全部通道）"),
              ("mean_all", "平均两两 |r|（全部通道）"),
              ("copy_frac", "幽灵“复制”比例（与某本征 |r|>0.9）"),
              ("R2_group", "幽灵被父组线性解释的方差 R²")]
    fig, axes = plt.subplots(len(panels), len(STAGES), figsize=(3.4 * len(STAGES), 2.6 * len(panels)),
                             sharex=True)
    for i, (key, title) in enumerate(panels):
        for j, st in enumerate(STAGES):
            ax = axes[i, j]
            means, stds = [], []
            for r in ratios:
                vals = [all_rows[(r, s)][st][key] for s in range(3)]
                ax.scatter([r] * len(vals), vals, s=14, alpha=0.5, color="#1f77b4")
                means.append(np.nanmean(vals))
                stds.append(np.nanstd(vals, ddof=1))
            ax.errorbar(ratios, means, yerr=stds, color="#d62728", marker="o", capsize=3, lw=1.5,
                        label="Ghost（本征占比 r）" if (i == 0 and j == 0) else None)
            if yolo is not None and key in yolo[st]:
                ref = yolo[st][key]
                ax.axhline(ref["mean"], color="gray", ls="--", lw=1.2,
                           label="原版 YOLOv5s（3 seed 均值）" if (i == 0 and j == 0) else None)
                ax.axhspan(ref["mean"] - ref["std"], ref["mean"] + ref["std"], color="gray", alpha=0.12)
            if i == 0 and j == 0:
                ax.legend(fontsize=7, loc="best")
            ax.set_xticks(ratios)
            ax.set_xticklabels([f"{r:g}" for r in ratios], fontsize=8)
            ax.grid(alpha=0.3)
            if i == 0:
                ax.set_title(f"stage{st}（C={all_rows[(ratios[0], 0)][st]['C']}）", fontsize=10)
            if j == 0:
                ax.set_ylabel(title, fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("本征占比 r（0.5 = 基线）", fontsize=8)
    fig.suptitle("Ghost 本征占比对下采样输出通道冗余的影响（每点 = 1 个 seed，线 = 均值 ± 标准差）", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    ra.save_fig(fig, out, 110)
    plt.close(fig)


def plot_heatmaps(mats: dict, out: Path, stage: int) -> None:
    ratios = sorted(mats)
    fig, axes = plt.subplots(1, len(ratios), figsize=(4.6 * len(ratios), 4.4))
    im = None
    for ax, r in zip(axes, ratios):
        M, n_int = mats[r]
        im = ax.imshow(M, vmin=0, vmax=1, cmap="magma", interpolation="nearest")
        ax.axhline(n_int - 0.5, color="cyan", lw=1)
        ax.axvline(n_int - 0.5, color="cyan", lw=1)
        ax.set_title(f"r={r:g}（本征 {n_int} | 幽灵 {M.shape[0] - n_int}）\n|r| 矩阵，青线为本征/幽灵分界", fontsize=9)
        ax.set_xlabel("通道（本征在前，幽灵在后）", fontsize=8)
        ax.set_ylabel("通道", fontsize=8)
    cax = fig.add_axes([0.93, 0.18, 0.012, 0.66])
    fig.colorbar(im, cax=cax, label="|r|")
    fig.suptitle(f"stage{stage} 下采样输出的通道相关矩阵（同一张测试图，seed 42）", fontsize=11)
    fig.tight_layout(rect=(0, 0, 0.92, 0.93))
    ra.save_fig(fig, out, 110)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--num", type=int, default=20, help="随机测试图数量（与 redundancy_analysis.py 相同的抽样）")
    ap.add_argument("--example", default="0100061", help="热力图示例图（不含扩展名）")
    ap.add_argument("--heat-stage", type=int, default=3)
    args = ap.parse_args()
    t0 = time.time()

    data_dir = ROOT / "data" / "Det_Fly"
    ds = YoloDetDataset(data_dir / "images" / "test", data_dir / "labels" / "test", 640, train=False)
    stems = [p.stem for p in ds.paths]
    rng = np.random.default_rng(0)
    picks = [int(i) for i in rng.choice(len(ds), size=args.num, replace=False)]
    OUT.mkdir(parents=True, exist_ok=True)

    all_rows: dict = {}
    heat: dict = {}
    for r in sorted(RUNS):
        for s, spec in enumerate(RUNS[r]):
            rows, model = analyze_run(spec, ds, picks)
            all_rows[(r, s)] = rows
            print(f"完成：r={r:g} seed{spec['seed']}（{time.time() - t0:.0f}s）", flush=True)
            if spec["seed"] == 42:
                img_t, _ = ds[stems.index(args.example)]
                n = NAMES[STAGES.index(args.heat_stage)]
                heat[r] = (corr_example(model, img_t, n), rows[args.heat_stage]["n_int"])

    # 原版 YOLOv5s 参照（同一批测试图、同一套口径，仅全通道指标）
    yolo_rows: dict = {}
    for s, spec in enumerate(YOLO_RUNS):
        rows, _ = ra.analyze_model(spec, ds, picks)
        yolo_rows[s] = {r["stage"]: r for r in rows}
        print(f"完成：原版 YOLOv5s seed{spec['seed']}（{time.time() - t0:.0f}s）", flush=True)
    yolo_sum: dict = {}
    for st in STAGES:
        yolo_sum[st] = {}
        for k in ["NN_all", "mean_all", "dup_all"]:
            vals = [yolo_rows[s][st][k] for s in range(3)]
            yolo_sum[st][k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1)), "vals": vals}
    print("\n原版 YOLOv5s（Conv↓，3 seed，全部通道）：")
    for st in STAGES:
        c = yolo_sum[st]
        print(f"  stage{st}: NN|r| {c['NN_all']['mean']:.3f}±{c['NN_all']['std']:.3f}  "
              f"dup% {c['dup_all']['mean']:.3f}±{c['dup_all']['std']:.3f}  "
              f"mean|r| {c['mean_all']['mean']:.3f}±{c['mean_all']['std']:.3f}")

    # 汇总：每个 r、每个 stage 的 3-seed 均值 ± 标准差
    summary = {}
    print(f"\n{'r':>6} {'stage':>5} {'C':>4} {'n_int':>5} | {'NN_all':>13} {'dup_all':>13} {'mean_all':>13} | "
          f"{'copy':>13} {'best_par':>13} {'R2_grp':>13} | {'dup_gg':>13}")
    for r in sorted(RUNS):
        summary[r] = {}
        for st in STAGES:
            cell = {}
            for k in METRICS + ["C", "n_int", "n_gh"]:
                vals = [all_rows[(r, s)][st][k] for s in range(3)]
                cell[k] = {"mean": float(np.nanmean(vals)), "std": float(np.nanstd(vals, ddof=1)), "vals": vals}
            summary[r][st] = cell

            def f(k):
                c = cell[k]
                return f"{c['mean']:.3f}±{c['std']:.3f}"
            print(f"{r:>6g} {st:>5} {int(cell['C']['mean']):>4} {int(cell['n_int']['mean']):>5} | "
                  f"{f('NN_all'):>13} {f('dup_all'):>13} {f('mean_all'):>13} | "
                  f"{f('copy_frac'):>13} {f('best_par_abs'):>13} {f('R2_group'):>13} | {f('dup_gg'):>13}")

    (OUT / "ir_redundancy.json").write_text(json.dumps({
        "picks": picks,
        "runs": {f"{r:g}_seed{spec['seed']}": {"spec": spec, "rows": {str(st): all_rows[(r, s)][st]
                 for st in STAGES}} for r in sorted(RUNS) for s, spec in enumerate(RUNS[r])},
        "summary": {f"{r:g}": {str(st): summary[r][st] for st in STAGES} for r in sorted(RUNS)},
        "yolov5s_ref": {str(st): yolo_sum[st] for st in STAGES},
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    plot_metrics(all_rows, OUT / "ir_metrics.png", yolo_sum)
    plot_heatmaps(heat, OUT / f"ir_corr_stage{args.heat_stage}.png", args.heat_stage)
    print(f"\n输出目录：{OUT}（总耗时 {time.time() - t0:.0f}s）")


if __name__ == "__main__":
    main()
