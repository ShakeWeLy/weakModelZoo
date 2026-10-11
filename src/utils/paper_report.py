"""论文格式的结果表与图：UNet / TransUNet / Swin-UNet（weakUnet_Synapse 测试集）。

数据来源（默认路径，可通过参数覆盖）：
- TransUNet：TransUNet/test_log/test_log_TU_Synapse224/*.txt（3D 体积级，取最后一次运行）
- Swin-UNet：Swin-Unet/test_log/test_log_/best_model.pth.txt（3D 体积级，取最后一次运行）
- UNet：experiments/segment/project1_2610/UNet/runs/2026-10-08_001_UNet_V1/predictions/test/organ_metrics.csv
        与 paper_metrics.csv（切片级均值，与前两者口径不同）

用法（项目根目录）：
    python src/utils/paper_report.py
    python src/utils/paper_report.py --out_dir outputs/paper_report

输出：
    tables.tex / tables.md / metrics.csv，以及 fig_dice_per_organ.{png,pdf}、fig_hd95_per_organ.{png,pdf}
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]

ORGANS = [
    "spleen",
    "right kidney",
    "left kidney",
    "gallbladder",
    "liver",
    "stomach",
    "aorta",
    "pancreas",
]

DEFAULT_TRANSUNET_LOG = (
    ROOT / "TransUNet/test_log/test_log_TU_Synapse224/TU_pretrain_R50-ViT-B_16_skip3_epo150_bs16_224.txt"
)
DEFAULT_SWIN_LOG = ROOT / "Swin-Unet/test_log/test_log_/best_model.pth.txt"
DEFAULT_UNET_DIR = ROOT / "experiments/segment/project1_2610/UNet/runs/2026-10-08_001_UNet_V1"

MEAN_CLASS_RE = re.compile(r"Mean class (\d+) mean_dice ([\d.]+) mean_hd95 ([\d.]+)")
OVERALL_RE = re.compile(r"Testing performance in best val model: mean_dice : ([\d.]+) mean_hd95 : ([\d.]+)")

PALETTE = {"UNet": "#9e9e9e", "TransUNet": "#1f77b4", "Swin-UNet": "#ff7f0e"}


@dataclass
class ModelResult:
    name: str
    protocol: str  # 评估口径
    dice: np.ndarray  # 每器官 Dice (%)，顺序同 ORGANS
    hd95: np.ndarray  # 每器官 HD95 (px)
    overall_dice: float  # 宏平均 Dice (%)
    overall_hd95: float  # 宏平均 HD95 (px)
    source: str


def parse_volume_log(path: Path, name: str) -> ModelResult:
    """解析 test.py 日志。同一文件可能含多次运行，取最后一次（最后 8 个类别行 + 最后一条总结）。"""
    text = path.read_text(encoding="utf-8", errors="ignore")
    class_rows = MEAN_CLASS_RE.findall(text)
    overall_rows = OVERALL_RE.findall(text)
    if len(class_rows) < len(ORGANS) or not overall_rows:
        raise ValueError(f"日志中未找到完整的测试结果: {path}")
    last = class_rows[-len(ORGANS):]
    ids = [int(c[0]) for c in last]
    if ids != list(range(1, len(ORGANS) + 1)):
        raise ValueError(f"类别顺序异常 {ids}: {path}")
    dice = np.array([float(c[1]) * 100 for c in last])
    hd95 = np.array([float(c[2]) for c in last])
    overall_dice, overall_hd95 = (float(v) for v in overall_rows[-1])
    return ModelResult(
        name=name,
        protocol="3D 体积级",
        dice=dice,
        hd95=hd95,
        overall_dice=overall_dice * 100,
        overall_hd95=overall_hd95,
        source=str(path.relative_to(ROOT)),
    )


def parse_unet(run_dir: Path, name: str = "UNet") -> ModelResult:
    organ_csv = run_dir / "predictions/test/organ_metrics.csv"
    paper_csv = run_dir / "paper_metrics.csv"
    with organ_csv.open(encoding="utf-8-sig", newline="") as f:
        rows = {r["organ"].replace("_", " "): r for r in csv.DictReader(f)}
    missing = [o for o in ORGANS if o not in rows]
    if missing:
        raise ValueError(f"organ_metrics.csv 缺少器官: {missing}")
    dice = np.array([float(rows[o]["dice_percent"]) for o in ORGANS])
    hd95 = np.array([float(rows[o]["hd95_mean"]) for o in ORGANS])

    with paper_csv.open(encoding="utf-8-sig", newline="") as f:
        test_row = next(r for r in csv.DictReader(f) if r["split"] == "test")
    return ModelResult(
        name=name,
        protocol="切片级均值",
        dice=dice,
        hd95=hd95,
        overall_dice=float(test_row["Dice (%)"]),
        overall_hd95=float(test_row["HD95 (px)"]),
        source=str(organ_csv.relative_to(ROOT)),
    )


def _bold(text: str, flag: bool, fmt: str) -> str:
    return fmt.format(text) if not flag else "\\textbf{" + fmt.format(text) + "}"


def build_tables(results: list[ModelResult]) -> tuple[str, str]:
    """返回 (LaTeX 字符串, Markdown 字符串)。每个器官行内最优值加粗（Dice 最大、HD95 最小）。"""
    names = [r.name for r in results]
    # ---- 表 1：整体指标 ----
    best_dice = max(r.overall_dice for r in results)
    best_hd = min(r.overall_hd95 for r in results)
    tex = []
    tex.append("% 表 1：整体指标（宏平均，不含背景）")
    tex.append("\\begin{table}[t]\n\\centering\n\\caption{Overall segmentation results on the weakUnet\\_Synapse test set. "
               "Dice and HD95 are macro-averaged over the eight organs. "
               "UNet is evaluated at the slice level, whereas TransUNet and Swin-UNet are evaluated on 3D volumes; "
               "the two protocols are not directly comparable.}")
    tex.append("\\label{tab:overall}\n\\begin{tabular}{lccl}\n\\toprule")
    tex.append("Method & Dice (\\%) $\\uparrow$ & HD95 (px) $\\downarrow$ & Protocol \\\\\n\\midrule")
    for r in results:
        tex.append(
            f"{r.name} & {_bold(f'{r.overall_dice:.2f}', r.overall_dice == best_dice, '{}')} & "
            f"{_bold(f'{r.overall_hd95:.2f}', r.overall_hd95 == best_hd, '{}')} & {r.protocol} \\\\"
        )
    tex.append("\\bottomrule\n\\end{tabular}\n\\end{table}")

    md = ["| Method | Dice (%) ↑ | HD95 (px) ↓ | Protocol |", "| --- | --- | --- | --- |"]
    for r in results:
        d = f"**{r.overall_dice:.2f}**" if r.overall_dice == best_dice else f"{r.overall_dice:.2f}"
        h = f"**{r.overall_hd95:.2f}**" if r.overall_hd95 == best_hd else f"{r.overall_hd95:.2f}"
        md.append(f"| {r.name} | {d} | {h} | {r.protocol} |")

    # ---- 表 2：逐器官 ----
    cols = []
    for r in results:
        cols.append(("Dice", r))
        cols.append(("HD95", r))
    col_spec = "l" + "cc" * len(results)
    header_tex = " & ".join(["Organ"] + [f"\\multicolumn{{2}}{{c}}{{{r.name}}}" for r in results]) + " \\\\"
    sub_tex = " & ".join([""] + ["Dice (\\%) & HD95 (px)" for _ in results]) + " \\\\"
    tex.append("% 表 2：逐器官指标")
    tex.append("\\begin{table*}[t]\n\\centering\n\\caption{Per-organ Dice (\\%) and HD95 (px) on the weakUnet\\_Synapse test set. "
               "Best value per organ in bold. UNet uses slice-level averaging; TransUNet and Swin-UNet use 3D volume-level averaging.}")
    tex.append("\\label{tab:per_organ}\n\\begin{tabular}{" + col_spec + "}\n\\toprule")
    tex.append(header_tex)
    tex.append("\\cmidrule(lr){2-3}" + "".join(
        f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(1, len(results))))
    tex.append(sub_tex + "\\midrule")

    md_header = ["| Organ | " + " | ".join(f"{r.name} Dice (%) | {r.name} HD95 (px)" for r in results) + " |"]
    md_header.append("| --- | " + " | ".join("--- | ---" for _ in results) + " |")
    md.append("")
    md.extend(md_header)

    for i, organ in enumerate(ORGANS):
        dice_vals = [r.dice[i] for r in results]
        hd_vals = [r.hd95[i] for r in results]
        best_d, best_h = max(dice_vals), min(hd_vals)
        cells_tex = [organ]
        cells_md = [organ]
        for r in results:
            d, h = r.dice[i], r.hd95[i]
            dtxt = f"{d:.2f}"
            htxt = f"{h:.2f}"
            cells_tex.append(
                (f"\\textbf{{{dtxt}}}" if d == best_d else dtxt) + " & " +
                (f"\\textbf{{{htxt}}}" if h == best_h else htxt)
            )
            cells_md.append((f"**{dtxt}**" if d == best_d else dtxt) + " | " +
                            (f"**{htxt}**" if h == best_h else htxt))
        tex.append(" & ".join(cells_tex) + " \\\\")
        md.append("| " + " | ".join(cells_md) + " |")

    # 宏平均行
    mean_cells_tex = ["Mean"]
    mean_cells_md = ["**Mean**"]
    mean_dice = [r.overall_dice for r in results]
    mean_hd = [r.overall_hd95 for r in results]
    for r in results:
        mean_cells_tex.append(f"{r.overall_dice:.2f} & {r.overall_hd95:.2f}")
        mean_cells_md.append(f"{r.overall_dice:.2f} | {r.overall_hd95:.2f}")
    tex.append("\\midrule")
    tex.append(" & ".join(mean_cells_tex) + " \\\\")
    tex.append("\\bottomrule\n\\end{tabular}\n\\end{table*}")
    md.append("| " + " | ".join(mean_cells_md) + " |")

    return "\n\n".join(tex) + "\n", "\n".join(md) + "\n"


def write_metrics_csv(results: list[ModelResult], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "protocol", "organ", "dice_percent", "hd95_px", "source"])
        for r in results:
            for i, organ in enumerate(ORGANS):
                w.writerow([r.name, r.protocol, organ, f"{r.dice[i]:.4f}", f"{r.hd95[i]:.4f}", r.source])
            w.writerow([r.name, r.protocol, "MEAN", f"{r.overall_dice:.4f}", f"{r.overall_hd95:.4f}", r.source])


def _style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.bbox": "tight",
    })


def plot_grouped(results: list[ModelResult], attr: str, ylabel: str, out_stem: Path, higher_better: bool) -> None:
    _style()
    n_models = len(results)
    x = np.arange(len(ORGANS))
    width = 0.8 / n_models
    fig, ax = plt.subplots(figsize=(7.0, 2.8))
    for j, r in enumerate(results):
        vals = getattr(r, attr)
        ax.bar(x + (j - (n_models - 1) / 2) * width, vals, width,
               label=r.name, color=PALETTE.get(r.name, None), edgecolor="black", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([o.title() for o in ORGANS], rotation=0)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 100 if attr == "dice" else None)
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.7)
    ax.legend(frameon=False, ncol=n_models, loc="upper center", bbox_to_anchor=(0.5, 1.15))
    fig.tight_layout()
    fig.savefig(out_stem.with_suffix(".png"), dpi=300)
    fig.savefig(out_stem.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--transunet_log", type=Path, default=DEFAULT_TRANSUNET_LOG)
    parser.add_argument("--swin_log", type=Path, default=DEFAULT_SWIN_LOG)
    parser.add_argument("--unet_dir", type=Path, default=DEFAULT_UNET_DIR)
    parser.add_argument("--out_dir", type=Path, default=ROOT / "outputs/paper_report")
    args = parser.parse_args()

    results = [
        parse_unet(args.unet_dir),
        parse_volume_log(args.transunet_log, "TransUNet"),
        parse_volume_log(args.swin_log, "Swin-UNet"),
    ]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tex, md = build_tables(results)
    (args.out_dir / "tables.tex").write_text(tex, encoding="utf-8")
    (args.out_dir / "tables.md").write_text(md, encoding="utf-8")
    write_metrics_csv(results, args.out_dir / "metrics.csv")
    plot_grouped(results, "dice", "Dice (%)", args.out_dir / "fig_dice_per_organ", higher_better=True)
    plot_grouped(results, "hd95", "HD95 (px)", args.out_dir / "fig_hd95_per_organ", higher_better=False)

    print(md)
    print(f"已输出到: {args.out_dir}")


if __name__ == "__main__":
    main()
