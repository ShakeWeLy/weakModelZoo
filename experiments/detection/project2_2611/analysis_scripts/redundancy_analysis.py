"""下采样层通道冗余对比：YOLOv5s 标准 Conv↓ vs c3topo_1（同为 Conv↓）vs c3_sgdown（SGhost↓）。

对 --num 张测试图（固定随机种子）统计 stage2~stage5 下采样模块（backbone.stageK.0）的输出。
所有指标均在【单张图内】计算（同一张图里通道之间的关系），再对图像取平均：
  NN|r|    每个通道与其它通道的最大 |Pearson r|，对通道取平均（依赖通道数，仅作参考）
  mean|r|  所有通道两两 |Pearson r| 的均值
  dup%     近似重复通道比例：与任一其它通道的 |r|>0.9 的通道占比
仅 SGhost 额外给出（只用本征通道 / 幽灵与父通道）：
  NN_int / mean_int / dup_int   只用本征通道计算（不含本征-幽灵配对）
  r²        幽灵与其本征父通道的 Pearson r² 均值：线性可解释的方差比例
  |ρ|       幽灵与父通道的 Spearman |ρ| 均值：接近 1 表示幽灵是父通道的单调函数（逐像素）
  mono%     |ρ|>0.99 的幽灵通道比例

示例图：选取 20 张图上平均 |r| 最大的通道对，并在标题中同时给出平均值与本图的 r。

输出目录：experiments/detection/project2_2611/redundancy_analysis/
用法（仓库根目录）：
  python experiments/detection/project2_2611/redundancy_analysis.py --num 20
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from trainer import RunRecorder, build_config, build_model, model_output_dir  # noqa: E402
from trainer.config import ROOT  # noqa: E402
from trainer.data import YoloDetDataset  # noqa: E402

from src.module.conv import GhostConv  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def save_fig(fig, out: Path, dpi: int) -> None:
    """先写临时文件再原子替换，并在 Windows 文件被短暂占用时重试。"""
    tmp = out.with_name(out.stem + ".tmp.png")
    for attempt in range(6):
        try:
            fig.savefig(tmp, dpi=dpi)
            os.replace(tmp, out)
            return
        except OSError:
            time.sleep(0.5 * (attempt + 1))
    raise OSError(f"无法写入 {out}")


STAGES = [2, 3, 4, 5]
NAMES = [f"backbone.stage{k}.0" for k in STAGES]
SPECS = [
    {"label": "YOLOv5s (Conv↓)", "model": "yolov5s", "run": "2026-10-30_002_YOLOv5s_V3"},
    {"label": "c3topo_1 (Conv↓)", "model": "sgc_yolov5s_c3topo_1", "run": "2026-11-12_021_SGC_c3topo_1_s_V1"},
    {"label": "c3_sgdown (SGhost↓)", "model": "sgc_yolov5s_c3_sgdown", "run": "2026-11-12_024_SGC_c3_sgdown_s_V1"},
]
OUT = Path(__file__).resolve().parent / "redundancy_analysis"
DUP_THR = 0.9


# ---------------------------------------------------------------- 统计工具（均在单张图内计算）
def flat(t: np.ndarray) -> np.ndarray:
    """(1, C, H, W) -> (C, P) float64。"""
    return t[0].reshape(t.shape[1], -1).astype(np.float64)


def corr_matrix(a: np.ndarray) -> np.ndarray:
    """(C, P) 的逐通道 Pearson 相关矩阵；常数通道记为 NaN。"""
    a = a - a.mean(axis=1, keepdims=True)
    s = np.sqrt((a * a).sum(axis=1))
    R = np.full((a.shape[0], a.shape[0]), np.nan)
    ok = np.where(s > 1e-10)[0]
    an = a[ok] / s[ok, None]
    R[np.ix_(ok, ok)] = an @ an.T
    return R


def nn_abs(R: np.ndarray) -> float:
    """每个通道与其它通道的最大 |r| 的均值。"""
    M = np.abs(R).copy()
    np.fill_diagonal(M, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        nn = np.nanmax(M, axis=1)
    return float(np.nanmean(nn))


def mean_abs_pair(R: np.ndarray) -> float:
    """所有有效通道两两 |r| 的均值（上三角）。"""
    valid = ~np.isnan(np.diag(R))
    M = np.abs(R[np.ix_(valid, valid)])
    iu = np.triu_indices(M.shape[0], 1)
    return float(np.nanmean(M[iu]))


def dup_frac(R: np.ndarray, thr: float = DUP_THR) -> float:
    """最相似邻居 |r| 超过阈值的通道占比。"""
    M = np.abs(R).copy()
    np.fill_diagonal(M, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        nn = np.nanmax(M, axis=1)
    valid = ~np.isnan(nn)
    return float(np.mean(nn[valid] > thr))


def pair_stats(intr: np.ndarray, ghost: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """逐通道（本征 i 与幽灵 i）的 Pearson r 与 Spearman ρ。输入均为 (C, P)。"""
    C = intr.shape[0]
    r = np.full(C, np.nan)
    rho = np.full(C, np.nan)
    for i in range(C):
        x, y = intr[i], ghost[i]
        if x.std() < 1e-10 or y.std() < 1e-10:
            continue
        r[i] = np.corrcoef(x, y)[0, 1]
        rho[i] = np.corrcoef(np.argsort(np.argsort(x)), np.argsort(np.argsort(y)))[0, 1]
    return r, rho


def _acc(M, cnt, R):
    """累积逐图 |R| 与有效计数，用于求平均 |r| 矩阵。"""
    if M is None:
        M = np.zeros(R.shape)
        cnt = np.zeros(R.shape)
    return M + np.nan_to_num(np.abs(R)), cnt + (~np.isnan(R))


def mbar(acc: tuple) -> np.ndarray:
    M, cnt = acc
    return np.where(cnt > 0, M / np.maximum(cnt, 1), np.nan)


def top_pair(R: np.ndarray) -> tuple[int, int, float]:
    """|R| 最大的一对通道（上三角），返回 (i, j, R_ij)。"""
    M = np.abs(R).copy()
    M[np.isnan(M)] = -1.0
    np.fill_diagonal(M, -1.0)
    i, j = np.unravel_index(np.argmax(np.triu(M, 1)), M.shape)
    return int(i), int(j), float(R[i, j])


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a.ravel()))
    rb = np.argsort(np.argsort(b.ravel()))
    return float(np.corrcoef(ra, rb)[0, 1])


# ---------------------------------------------------------------- 模型与钩子
def load_model(spec: dict) -> torch.nn.Module:
    run_dir = model_output_dir(build_config(spec["model"])) / spec["run"]
    rec = RunRecorder.open(run_dir)
    cfg = rec.load_config()
    ckpt = torch.load(rec.checkpoints_dir / "best.pth", map_location="cpu", weights_only=False)
    model = build_model(cfg, int(ckpt["nc"]))
    model.load_state_dict(ckpt["model"])
    return model.eval()


def make_hook(cap: dict, key: str):
    def _h(_m, _i, out):
        cap[key] = out.detach().float().cpu().numpy()

    return _h


def capture(model: torch.nn.Module, img_t: torch.Tensor, names: list[str]) -> dict:
    mods = dict(model.named_modules())
    cap: dict = {}
    handles = [mods[n].register_forward_hook(make_hook(cap, n)) for n in names]
    with torch.no_grad():
        model(img_t[None].float() / 255.0)
    for h in handles:
        h.remove()
    return cap


# ---------------------------------------------------------------- 主分析
def analyze_model(spec: dict, ds: YoloDetDataset, picks: list[int]) -> tuple[list[dict], dict]:
    model = load_model(spec)
    mods = dict(model.named_modules())
    ghost = {n: isinstance(mods[n], GhostConv) for n in NAMES}
    need = set(NAMES)
    for n in NAMES:
        if ghost[n]:
            need |= {f"{n}.conv", f"{n}.cheap_conv"}
    cap: dict = {}
    handles = [mods[k].register_forward_hook(make_hook(cap, k)) for k in sorted(need)]

    per = {n: {"nn": [], "mean": [], "dup": [], "nn_i": [], "mean_i": [], "dup_i": [],
               "r": [], "rho": []} for n in NAMES}
    acc = {n: {"all": None, "int": None} for n in NAMES}
    for idx in picks:
        img_t, _ = ds[idx]
        cap.clear()
        with torch.no_grad():
            model(img_t[None].float() / 255.0)
        for n in NAMES:
            R = corr_matrix(flat(cap[n]))
            p = per[n]
            p["nn"].append(nn_abs(R))
            p["mean"].append(mean_abs_pair(R))
            p["dup"].append(dup_frac(R))
            acc[n]["all"] = _acc(*(acc[n]["all"] or (None, None)), R)
            if ghost[n]:
                it = flat(cap[f"{n}.conv"])
                gh = flat(cap[f"{n}.cheap_conv"])
                Ri = corr_matrix(it)
                p["nn_i"].append(nn_abs(Ri))
                p["mean_i"].append(mean_abs_pair(Ri))
                p["dup_i"].append(dup_frac(Ri))
                acc[n]["int"] = _acc(*(acc[n]["int"] or (None, None)), Ri)
                r, rho = pair_stats(it, gh)
                p["r"].append(r)
                p["rho"].append(rho)
    for h in handles:
        h.remove()

    rows: list[dict] = []
    for n, st in zip(NAMES, STAGES):
        p = per[n]
        row = {"label": spec["label"], "stage": st, "C": int(acc[n]["all"][0].shape[0]),
               "NN_all": float(np.mean(p["nn"])), "mean_all": float(np.mean(p["mean"])),
               "dup_all": float(np.mean(p["dup"]))}
        if ghost[n]:
            rs = np.stack(p["r"])
            rhos = np.stack(p["rho"])
            row.update({
                "NN_int": float(np.mean(p["nn_i"])),
                "mean_int": float(np.mean(p["mean_i"])),
                "dup_int": float(np.mean(p["dup_i"])),
                "r2": float(np.nanmean(rs ** 2)),
                "abs_rho": float(np.nanmean(np.abs(rhos))),
                "mono": float(np.nanmean(np.abs(rhos) > 0.99)),
            })
        rows.append(row)

    s3 = NAMES[1]
    extra = {"model": model, "Mbar3": mbar(acc[s3]["all"])}
    if ghost[s3]:
        extra["Mbar3_int"] = mbar(acc[s3]["int"])
        r3_mean = np.nanmean(np.stack(per[s3]["r"]), axis=0)
        med = np.nanmedian(r3_mean)
        extra["typical_i"] = int(np.nanargmin(np.abs(r3_mean - med)))
    return rows, extra


def example_panels(extras: dict, ds: YoloDetDataset, stems: list[str], example: str) -> list[dict]:
    img_t, _ = ds[stems.index(example)]
    n = NAMES[1]
    panels = []

    ex = extras[SPECS[0]["label"]]
    a, b, avg = top_pair(ex["Mbar3"])
    A, B = capture(ex["model"], img_t, [n])[n][0][a], capture(ex["model"], img_t, [n])[n][0][b]
    panels.append({"row": f"YOLOv5s Conv↓ 最相关的一对 ch{a}/ch{b}",
                   "A": A, "B": B, "la": f"Conv↓ ch{a}", "lb": f"Conv↓ ch{b}",
                   "xl": f"ch{a} 激活", "yl": f"ch{b} 激活",
                   "r": pearson(A, B), "rho": spearman(A, B), "avg": avg})

    ex = extras[SPECS[2]["label"]]
    cap = capture(ex["model"], img_t, [n, f"{n}.conv", f"{n}.cheap_conv"])
    intr, ghost = cap[f"{n}.conv"][0], cap[f"{n}.cheap_conv"][0]
    i = ex["typical_i"]
    A, B = intr[i], ghost[i]
    panels.append({"row": f"c3_sgdown 幽灵-父通道（典型）ch{i}",
                   "A": A, "B": B, "la": f"本征 ch{i}", "lb": f"幽灵 ch{i}",
                   "xl": "本征激活（父通道）", "yl": "幽灵激活",
                   "r": pearson(A, B), "rho": spearman(A, B), "avg": None})

    p, q, avg = top_pair(ex["Mbar3_int"])
    A, B = intr[p], intr[q]
    panels.append({"row": f"c3_sgdown 本征最相关的一对 ch{p}/ch{q}",
                   "A": A, "B": B, "la": f"本征 ch{p}", "lb": f"本征 ch{q}",
                   "xl": f"本征 ch{p}", "yl": f"本征 ch{q}",
                   "r": pearson(A, B), "rho": spearman(A, B), "avg": avg})
    return panels


def plot_examples(panels: list[dict], out: Path) -> None:
    fig, axes = plt.subplots(len(panels), 3, figsize=(11, 3.6 * len(panels)))
    rng = np.random.default_rng(0)
    for r, p in enumerate(panels):
        for c, key, title in [(0, "A", p["la"]), (1, "B", p["lb"])]:
            arr = p[key]
            lo, hi = np.percentile(arr, 1), np.percentile(arr, 99.5)
            axes[r, c].imshow(np.clip((arr - lo) / max(hi - lo, 1e-8), 0, 1), cmap="viridis",
                              interpolation="nearest")
            axes[r, c].set_title(title, fontsize=9)
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
        axes[r, 0].set_ylabel(p["row"], fontsize=9)
        a, b = p["A"].ravel(), p["B"].ravel()
        sel = rng.choice(a.size, size=min(4000, a.size), replace=False)
        axes[r, 2].scatter(a[sel], b[sel], s=3, alpha=0.4)
        axes[r, 2].set_xlabel(p["xl"], fontsize=8)
        axes[r, 2].set_ylabel(p["yl"], fontsize=8)
        title = f"本图 Pearson r={p['r']:.3f}, Spearman ρ={p['rho']:.3f}"
        if p["avg"] is not None:
            title += f"\n20 图平均 |r|={p['avg']:.3f}"
        axes[r, 2].set_title(title, fontsize=9)
        axes[r, 2].grid(alpha=0.3)
    fig.suptitle("Stage3 下采样输出：通道对示例（左、中为两通道特征图，右为逐像素散点）", fontsize=11)
    fig.tight_layout()
    save_fig(fig, out, 110)
    plt.close(fig)


def plot_bars(results: dict, out: Path) -> None:
    series = []
    for spec in SPECS[:2]:
        series.append((spec["label"], results[spec["label"]], "all"))
    c3 = results[SPECS[2]["label"]]
    series.append(("c3_sgdown 全部通道", c3, "all"))
    series.append(("c3_sgdown 仅本征通道", c3, "int"))
    colors = ["#2ca02c", "#1f77b4", "#d62728", "#ff9896"]
    x = np.arange(len(STAGES))
    w = 0.2
    metrics = [("NN", "NN|r|（最相似邻居，越高越冗余）"),
               ("mean", "平均两两 |r|（逐图平均）"),
               ("dup", f"近似重复通道比例（|r|>{DUP_THR} 的通道占比）")]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for j, (lab, rows, kind) in enumerate(series):
        off = (j - (len(series) - 1) / 2) * w
        for ax, (m, _) in zip(axes, metrics):
            key = {"NN": "NN_all" if kind == "all" else "NN_int",
                   "mean": "mean_all" if kind == "all" else "mean_int",
                   "dup": "dup_all" if kind == "all" else "dup_int"}[m]
            ax.bar(x + off, [r[key] for r in rows], w, label=lab, color=colors[j])
    for ax, (_, title) in zip(axes, metrics):
        ax.set_xticks(x)
        ax.set_xticklabels([f"stage{s}" for s in STAGES])
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    save_fig(fig, out, 120)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--num", type=int, default=20, help="随机测试图数量")
    ap.add_argument("--example", default="0100061", help="示意图使用的测试图文件名（不含扩展名）")
    args = ap.parse_args()

    data_dir = ROOT / "data" / "Det_Fly"
    ds = YoloDetDataset(data_dir / "images" / "test", data_dir / "labels" / "test", 640, train=False)
    stems = [p.stem for p in ds.paths]
    rng = np.random.default_rng(0)
    picks = [int(i) for i in rng.choice(len(ds), size=args.num, replace=False)]
    OUT.mkdir(parents=True, exist_ok=True)

    results: dict[str, list[dict]] = {}
    extras: dict[str, dict] = {}
    for spec in SPECS:
        rows, extra = analyze_model(spec, ds, picks)
        results[spec["label"]] = rows
        extras[spec["label"]] = extra
        print(f"完成：{spec['label']}（{len(picks)} 张测试图）", flush=True)

    print(f"\n{'stage':>5} {'模型':<20} {'C':>4} | {'NN|r|':>6} {'mean|r|':>8} {'dup%':>6} | "
          f"{'NN_i':>6} {'mean_i':>7} {'dup%_i':>7} | {'r²':>5} {'|ρ|':>6} {'mono%':>6}")
    for label, rows in results.items():
        for r in rows:
            line = (f"{r['stage']:>5} {label:<20} {r['C']:>4} | {r['NN_all']:>6.3f} {r['mean_all']:>8.3f} "
                    f"{100 * r['dup_all']:>5.1f}% | ")
            if "NN_int" in r:
                line += (f"{r['NN_int']:>6.3f} {r['mean_int']:>7.3f} {100 * r['dup_int']:>6.1f}% | "
                         f"{r['r2']:>5.3f} {r['abs_rho']:>6.3f} {100 * r['mono']:>5.1f}%")
            print(line)

    (OUT / "redundancy_stats.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    plot_bars(results, OUT / "redundancy_stats.png")
    panels = example_panels(extras, ds, stems, args.example)
    plot_examples(panels, OUT / "redundancy_examples.png")
    print("\n输出目录:", OUT)


if __name__ == "__main__":
    main()
