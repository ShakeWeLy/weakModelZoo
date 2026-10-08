"""预测结果可视化：Image | GT | Prediction | 逐器官指标面板。"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402

from .config import ROOT  # noqa: E402
from .labels import LabelSpace  # noqa: E402

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.data.synapse.labels import class_color, make_overlay, metric_class_name  # noqa: E402


def _fmt(value: float | None, percent: bool = False) -> str:
    if value is None:
        return "N/A"
    return f"{value * 100:.1f}%" if percent else f"{value:.3f}"


def _metrics_text(organ_metrics: Mapping[int, Mapping[str, Any]] | None, pred: np.ndarray,
                  space: LabelSpace, has_label: bool) -> str:
    organ_map = space.organ_map()
    if not has_label:
        lines = ["Prediction volume (no GT)", "=" * 34]
        for class_id, organ in organ_map.items():
            area = int((pred == class_id).sum())
            if area:
                lines.append(f"[{organ}]  {area}px  ({area / pred.size * 100:.2f}%)")
        return "\n".join(lines if len(lines) > 2 else [*lines, "No evaluated organ predicted."])

    lines = ["Per-organ metrics", "=" * 34]
    if not organ_metrics:
        return "\n".join([*lines, "No organ present in GT or prediction."])
    overall: dict[str, list[float]] = {k: [] for k in ("dice", "iou", "precision", "recall", "fp_ratio", "fn_ratio")}
    for class_id, m in organ_metrics.items():
        lines.extend([
            f"[{organ_map[class_id]}]",
            f"  D {_fmt(m['dice'])}  IoU {_fmt(m['iou'])}  P {_fmt(m['precision'])}  R {_fmt(m['recall'])}",
            f"  GT {m['gt_area']}px ({m['gt_percent']:.2f}%)  Pred {m['pred_area']}px "
            f"({m['pred_percent']:.2f}%)  P/GT {_fmt(m['pred_gt_ratio'])}",
            f"  FP% {_fmt(m['fp_ratio'], True)}  FN% {_fmt(m['fn_ratio'], True)}  HD95 {_fmt(m.get('hd95'))}",
            "",
        ])
        for key, values in overall.items():
            if m.get(key) is not None:
                values.append(m[key])
    mean = {k: (float(np.mean(v)) if v else None) for k, v in overall.items()}
    lines.extend([
        "OVERALL (slice mean)",
        f"  D {_fmt(mean['dice'])}  IoU {_fmt(mean['iou'])}  P {_fmt(mean['precision'])}  R {_fmt(mean['recall'])}",
        f"  FP% {_fmt(mean['fp_ratio'], True)}  FN% {_fmt(mean['fn_ratio'], True)}",
    ])
    return "\n".join(lines)


def save_overlay(image: np.ndarray, label: np.ndarray | None, pred: np.ndarray,
                 organ_metrics: Mapping[int, Mapping[str, Any]] | None, output_path: Path,
                 space: LabelSpace) -> None:
    """着色前将 0..8 转回原始 14 类 ID 以复用 ORGAN_COLORS。"""
    has_label = label is not None
    pred_viz = space.to_original(pred)
    label_viz = space.to_original(label) if has_label else None

    if has_label:
        fig = plt.figure(figsize=(18, 7.2))
        gs = GridSpec(2, 4, height_ratios=[1, 0.12], width_ratios=[1, 1, 1, 1.15], wspace=0.08, hspace=0.12)
        axes = [fig.add_subplot(gs[0, i]) for i in range(4)]
        ax_img, ax_gt, ax_pred, ax_text = axes
    else:
        fig = plt.figure(figsize=(15, 6.2))
        gs = GridSpec(2, 3, height_ratios=[1, 0.12], width_ratios=[1, 1, 1.1], wspace=0.08, hspace=0.12)
        ax_img, ax_pred, ax_text = (fig.add_subplot(gs[0, i]) for i in range(3))
        ax_gt = None
    ax_legend = fig.add_subplot(gs[1, :])

    ax_img.imshow(image, cmap="gray", vmin=0, vmax=1)
    ax_img.set_title("Image")
    if ax_gt is not None:
        ax_gt.imshow(make_overlay(image, label_viz))
        ax_gt.set_title("Ground Truth")
    ax_pred.imshow(make_overlay(image, pred_viz))
    ax_pred.set_title("Prediction")
    for ax in (ax_img, ax_gt, ax_pred, ax_text, ax_legend):
        if ax is not None:
            ax.axis("off")

    present: set[int] = set(np.unique(pred_viz).tolist())
    if label_viz is not None:
        present |= set(np.unique(label_viz).tolist())
    present.discard(0)
    if present:
        handles = [mpatches.Patch(color=class_color(c), label=f"{c}:{metric_class_name(c)}") for c in sorted(present)]
        ax_legend.legend(handles=handles, loc="center", ncol=min(7, len(handles)), fontsize=7.5, frameon=False)

    ax_text.text(
        0.02, 0.98, _metrics_text(organ_metrics, pred, space, has_label),
        transform=ax_text.transAxes, va="top", ha="left", fontsize=8.5, family="monospace",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.92, edgecolor="#CCCCCC"),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
