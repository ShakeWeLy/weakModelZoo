"""检测结果可视化：框叠加 + 官方风格指标图（PR / F1 / P / R / 混淆矩阵 / results）。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np


def render_detections(
    image_bgr: np.ndarray,
    gt_xyxy: np.ndarray,
    pred: np.ndarray,
    names: Sequence[str],
    vis_conf: float = 0.25,
) -> np.ndarray:
    """Args:
        image_bgr: (H, W, 3) uint8。
        gt_xyxy: (M, 4) 真值框（像素）。
        pred: (N, 6) -> x1, y1, x2, y2, conf, cls（像素）。
        vis_conf: 只画置信度 >= vis_conf 的预测。
    """
    out = image_bgr.copy()
    for x1, y1, x2, y2 in gt_xyxy:
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    for x1, y1, x2, y2, conf, cls in pred:
        if conf < vis_conf:
            continue
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        name = names[int(cls)] if int(cls) < len(names) else str(int(cls))
        cv2.putText(out, f"{name} {conf:.2f}", (int(x1), max(int(y1) - 4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    return out


def _class_colors(n: int) -> list[tuple[float, float, float]]:
    cmap = plt.get_cmap("tab10" if n <= 10 else "tab20")
    return [cmap(i % cmap.N) for i in range(n)]


def plot_pr_curve(curves: Mapping[str, Any], save_path: Path) -> Path:
    """PR 曲线（IoU=0.5），风格对齐 Ultralytics。"""
    names = list(curves["names"])
    px, py, ap50 = curves["px"], curves["py"], curves["ap50"]
    colors = _class_colors(len(names))
    fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=200)
    for i, name in enumerate(names):
        ax.plot(px, py[i], lw=1, color=colors[i], label=f"{name} {ap50[i]:.3f}")
    if len(names):
        ax.plot(px, py.mean(0), lw=3, color="blue", label=f"all classes {ap50.mean():.3f} mAP@0.5")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("PR Curve")
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_path = Path(save_path)
    fig.savefig(save_path)
    plt.close(fig)
    return save_path


def plot_mc_curve(
    x: np.ndarray,
    y: np.ndarray,
    names: Sequence[str],
    save_path: Path,
    *,
    xlabel: str = "Confidence",
    ylabel: str = "Metric",
    title: str = "Curve",
) -> Path:
    """多类别 vs 置信度曲线（P / R / F1）。"""
    colors = _class_colors(len(names))
    fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=200)
    for i, name in enumerate(names):
        ax.plot(x, y[i], lw=1, color=colors[i], label=f"{name}")
    if len(names):
        y_mean = y.mean(0)
        best = float(y_mean.max())
        best_x = float(x[int(y_mean.argmax())])
        ax.plot(x, y_mean, lw=3, color="blue", label=f"all classes {best:.3f} at {best_x:.3f}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_path = Path(save_path)
    fig.savefig(save_path)
    plt.close(fig)
    return save_path


def plot_confusion_matrix(
    mat: np.ndarray,
    names: Sequence[str],
    save_path: Path,
    *,
    normalize: bool = True,
) -> Path:
    """混淆矩阵（含 background），行=真值，列=预测。"""
    labels = list(names) + ["background"]
    data = mat.astype(np.float64)
    if normalize:
        row_sum = data.sum(1, keepdims=True)
        data = np.divide(data, row_sum, out=np.zeros_like(data), where=row_sum > 0)

    fig, ax = plt.subplots(1, 1, figsize=(max(6, len(labels)), max(5, len(labels) * 0.8)), dpi=200)
    im = ax.imshow(data, interpolation="nearest", cmap="Blues", vmin=0, vmax=1 if normalize else None)
    fig.colorbar(im, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix" + (" (Normalized)" if normalize else ""))
    thresh = (data.max() / 2) if data.size else 0.5
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            txt = f"{val:.2f}" if normalize else f"{int(mat[i, j])}"
            ax.text(j, i, txt, ha="center", va="center",
                    color="white" if val > thresh else "black", fontsize=8)
    fig.tight_layout()
    save_path = Path(save_path)
    fig.savefig(save_path)
    plt.close(fig)
    return save_path


def plot_metric_curves(curves: Mapping[str, Any], confusion: np.ndarray | None, out_dir: Path) -> list[Path]:
    """一次性写出 PR / F1 / P / R / 混淆矩阵（官方同款命名）。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    names = list(curves["names"])
    saved: list[Path] = [
        plot_pr_curve(curves, out_dir / "PR_curve.png"),
        plot_mc_curve(curves["conf_curve"], curves["f1_curve"], names, out_dir / "F1_curve.png",
                      ylabel="F1", title="F1 Curve"),
        plot_mc_curve(curves["conf_curve"], curves["p_curve"], names, out_dir / "P_curve.png",
                      ylabel="Precision", title="Precision Curve"),
        plot_mc_curve(curves["conf_curve"], curves["r_curve"], names, out_dir / "R_curve.png",
                      ylabel="Recall", title="Recall Curve"),
    ]
    if confusion is not None:
        saved.append(plot_confusion_matrix(confusion, names, out_dir / "confusion_matrix.png", normalize=True))
        saved.append(plot_confusion_matrix(confusion, names, out_dir / "confusion_matrix_raw.png", normalize=False))
    return saved


def plot_results_from_tensorboard(tb_dir: Path, save_path: Path) -> Path | None:
    """从 TensorBoard 事件画训练结果总览（类似官方 results.png）。"""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        return None
    tb_dir = Path(tb_dir)
    if not tb_dir.exists():
        return None
    ea = EventAccumulator(str(tb_dir))
    ea.Reload()
    tags = ea.Tags().get("scalars", [])
    if not tags:
        return None

    # 布局对齐官方：loss 一行 + metrics 一行
    panels = [
        ("loss/box", "train/box_loss"),
        ("loss/obj", "train/obj_loss"),
        ("loss/cls", "train/cls_loss"),
        ("loss/val", "val/loss"),
        ("metrics/precision", "metrics/precision"),
        ("metrics/recall", "metrics/recall"),
        ("metrics/mAP50", "metrics/mAP50"),
        ("metrics/mAP50-95", "metrics/mAP50-95"),
        ("lr", "lr"),
    ]
    available = [(tag, title) for tag, title in panels if tag in tags]
    if not available:
        return None

    n = len(available)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3), dpi=150)
    axes = np.atleast_1d(axes).ravel()
    for ax, (tag, title) in zip(axes, available):
        events = ea.Scalars(tag)
        xs = [e.step for e in events]
        ys = [e.value for e in events]
        ax.plot(xs, ys, marker=".", lw=1.5)
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.grid(True, alpha=0.3)
    for ax in axes[len(available):]:
        ax.axis("off")
    fig.suptitle("Training Results", fontsize=14)
    fig.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    return save_path
