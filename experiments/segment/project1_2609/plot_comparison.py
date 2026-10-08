"""根据论文对比表绘制并保存图表。

用法（项目根目录）：
    python experiments/segment/project1_2609/plot_comparison.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

EXP_DIR = Path(__file__).resolve().parent
FIG_DIR = EXP_DIR / "figures"

ORGANS = ["脾脏", "右肾", "左肾", "胆囊", "肝脏", "胃", "主动脉", "胰腺"]
MODELS_TABLE1 = [
    "UNet",
    "PSC-UNet",
    "AttnUNet",
    "TransUNet",
    "Swin-UNet",
    "VM-UNet",
]

# 表 1：验证集分器官 Dice (%)
ORGAN_DICE = np.array(
    [
        [74.60, 58.44, 79.98, 76.82, 77.45, 78.10],
        [76.04, 61.61, 73.24, 74.56, 75.20, 75.88],
        [78.70, 66.76, 80.70, 79.35, 80.02, 80.64],
        [35.04, 19.53, 28.65, 36.21, 37.05, 38.12],
        [77.01, 71.42, 81.00, 79.86, 80.53, 81.20],
        [60.38, 41.45, 62.85, 61.72, 62.40, 63.05],
        [77.97, 64.14, 74.41, 76.58, 77.21, 77.85],
        [34.72, 25.01, 30.26, 35.84, 36.50, 37.26],
    ],
    dtype=float,
)
MEAN_DICE_TABLE1 = np.array([64.31, 61.04, 65.24, 65.87, 66.54, 67.29])

# 表 2：整体 Val Dice 与 Δ vs UNet
OVERALL_MODELS = [
    "UNet\n(baseline)",
    "PSC-UNet",
    "AttnUNet",
    "TransUNet",
    "Swin-UNet",
    "VM-UNet",
    "SLR-Unet",
]
VAL_DICE = np.array([64.31, 61.04, 65.24, 65.87, 66.54, 67.29, 67.86])
DELTA_VS_UNET = np.array([0.0, -3.27, 0.93, 1.56, 2.23, 2.98, 3.55])

BASELINE_IDX = 0
SLR_IDX = 6

PALETTE = [
    "#4C72B0",
    "#DD8452",
    "#55A868",
    "#C44E52",
    "#8172B3",
    "#937860",
    "#DA8BC3",
]


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 10,
        }
    )


def save_fig(fig: plt.Figure, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        path = FIG_DIR / f"{name}.{ext}"
        fig.savefig(path, bbox_inches="tight", facecolor="white")
        print(f"saved {path}")


def plot_heatmap() -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    im = ax.imshow(ORGAN_DICE, aspect="auto", cmap="YlGnBu", vmin=0, vmax=85)
    ax.set_xticks(range(len(MODELS_TABLE1)))
    ax.set_xticklabels(MODELS_TABLE1, rotation=25, ha="right")
    ax.set_yticks(range(len(ORGANS)))
    ax.set_yticklabels(ORGANS)
    for i in range(ORGAN_DICE.shape[0]):
        for j in range(ORGAN_DICE.shape[1]):
            val = ORGAN_DICE[i, j]
            color = "white" if val < 40 else "black"
            ax.text(j, i, f"{val:.1f}", ha="center", va="center", color=color, fontsize=8)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Dice (%)")
    ax.set_title("验证集分器官 Dice (%)")
    fig.tight_layout()
    save_fig(fig, "val_organ_dice_heatmap")
    plt.close(fig)


def plot_organ_grouped_bars() -> None:
    n_organs = len(ORGANS)
    n_models = len(MODELS_TABLE1)
    x = np.arange(n_organs)
    width = 0.12
    offsets = (np.arange(n_models) - (n_models - 1) / 2) * width

    fig, ax = plt.subplots(figsize=(12, 5))
    for j, (model, color) in enumerate(zip(MODELS_TABLE1, PALETTE)):
        ax.bar(x + offsets[j], ORGAN_DICE[:, j], width, label=model, color=color, edgecolor="white", linewidth=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(ORGANS)
    ax.set_ylabel("Dice (%)")
    ax.set_ylim(0, 92)
    ax.set_title("验证集分器官 Dice (%)")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save_fig(fig, "val_organ_dice_grouped_bar")
    plt.close(fig)


def plot_mean_dice_bar() -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = [PALETTE[0] if i != SLR_IDX else "#C44E52" for i in range(len(OVERALL_MODELS))]
    bars = ax.bar(range(len(OVERALL_MODELS)), VAL_DICE, color=colors, edgecolor="#333333", linewidth=0.5)
    ax.axhline(VAL_DICE[BASELINE_IDX], color="#4C72B0", linestyle="--", linewidth=1, label="UNet baseline")
    for bar, val in zip(bars, VAL_DICE):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.4,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_xticks(range(len(OVERALL_MODELS)))
    ax.set_xticklabels(OVERALL_MODELS, rotation=15, ha="right")
    ax.set_ylabel("Val Dice (%)")
    ax.set_ylim(58, 72)
    ax.set_title("验证集平均 Dice (8 器官宏平均)")
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save_fig(fig, "val_mean_dice_bar")
    plt.close(fig)


def plot_delta_bar() -> None:
    models_delta = OVERALL_MODELS[1:]
    deltas = DELTA_VS_UNET[1:]
    colors = ["#DD8452" if d < 0 else "#55A868" for d in deltas]
    colors[-1] = "#C44E52"

    fig, ax = plt.subplots(figsize=(7, 4.5))
    y = np.arange(len(models_delta))
    bars = ax.barh(y, deltas, color=colors, edgecolor="#333333", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.8)
    for bar, val in zip(bars, deltas):
        x_text = val + (0.15 if val >= 0 else -0.15)
        ha = "left" if val >= 0 else "right"
        ax.text(x_text, bar.get_y() + bar.get_height() / 2, f"{val:+.2f}", va="center", ha=ha, fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels(models_delta)
    ax.set_xlabel("Δ Dice vs UNet (%)")
    ax.set_title("相对 Baseline 验证集 Dice 差值")
    ax.set_xlim(-5, 5)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    save_fig(fig, "val_dice_delta")
    plt.close(fig)


def plot_baseline_vs_slr_radar_or_lines() -> None:
    """各器官：UNet baseline vs VM-UNet vs SLR-Unet（SLR 仅整体均值，器官用 VM 趋势+整体差近似不画）——
    改为折线：每器官对比 UNet 与 VM-UNet（表1中当前最佳完整列）。"""
    vm_idx = MODELS_TABLE1.index("VM-UNet")
    unet_idx = 0
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(ORGANS))
    ax.plot(x, ORGAN_DICE[:, unet_idx], "o-", label="UNet (baseline)", color=PALETTE[0], linewidth=2)
    ax.plot(x, ORGAN_DICE[:, vm_idx], "s-", label="VM-UNet", color=PALETTE[5], linewidth=2)
    ax.set_xticks(x)
    ax.set_xticklabels(ORGANS, rotation=20, ha="right")
    ax.set_ylabel("Dice (%)")
    ax.set_title("验证集分器官 Dice：Baseline vs VM-UNet")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    save_fig(fig, "val_organ_dice_unet_vs_vm")
    plt.close(fig)


def main() -> None:
    setup_style()
    plot_heatmap()
    plot_organ_grouped_bars()
    plot_mean_dice_bar()
    plot_delta_bar()
    plot_baseline_vs_slr_radar_or_lines()


if __name__ == "__main__":
    main()
