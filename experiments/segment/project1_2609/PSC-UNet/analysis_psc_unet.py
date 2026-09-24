"""PSC-UNet 推理与评估：Dice / IoU / Precision / Recall / HD95 等指标与可视化。

用法（在项目根目录执行）：
    python experiments/segment/project1_2609/PSC-UNet/analysis_psc_unet.py
    python experiments/segment/project1_2609/PSC-UNet/analysis_psc_unet.py --split test
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import binary_erosion, distance_transform_edt
from torch.utils.data import DataLoader, Dataset

EXP_DIR = Path(__file__).resolve().parent
ROOT = EXP_DIR.parents[3]
sys.path.insert(0, str(ROOT))

from src.utils.data.synapse.labels import ORGAN_COLORS, EVAL_CLASS_IDS
from src.utils.logger import resolve_experiment_run, update_evaluation_summary
from src.utils.segmentation_metrics import aggregate_metric_lists, compute_binary_metrics

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore


ORGAN_METRICS = {cid: name for cid, name in zip(
    EVAL_CLASS_IDS,
    ["spleen", "right_kidney", "left_kidney", "gallbladder", "liver", "stomach", "aorta", "pancreas"],
)}

ORGAN_NAMES_CN = {
    "spleen": "脾脏",
    "right_kidney": "右肾脏",
    "left_kidney": "左肾脏",
    "gallbladder": "胆囊",
    "liver": "肝脏",
    "stomach": "胃",
    "aorta": "主动脉",
    "pancreas": "胰腺",
}

METRIC_KEYS = (
    "dice", "iou", "precision", "recall",
    "fp_ratio", "fn_ratio",
    "gt_area", "pred_area", "pred_gt_ratio", "gt_percent", "pred_percent",
    "hd95",
)


def load_psc_unet_class():
    module_path = ROOT / "src" / "models" / "segment" / "PSC-UNet" / "PSC-UNet.py"
    spec = importlib.util.spec_from_file_location("psc_unet_module", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载 PSC-UNet 模块: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.PSC_UNet


PSC_UNet = load_psc_unet_class()


class InferenceDataset(Dataset):
    def __init__(
        self,
        data_dir: Path,
        split: str,
        image_size: int,
        repeat_gray_to_rgb: bool = False,
    ):
        self.image_dir = data_dir / split / "images"
        self.label_dir = data_dir / split / "labels"
        self.image_size = image_size
        self.repeat_gray_to_rgb = repeat_gray_to_rgb
        self.samples = sorted(self.image_dir.glob("*.npy"))
        if not self.samples:
            raise FileNotFoundError(f"{self.image_dir} 下未找到 .npy 切片")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path = self.samples[index]
        image = np.load(image_path).astype("float32")
        image = torch.from_numpy(image).unsqueeze(0)
        image = F.interpolate(
            image.unsqueeze(0),
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        if self.repeat_gray_to_rgb:
            image = image.repeat(3, 1, 1)

        label_path = self.label_dir / image_path.name
        if label_path.exists():
            label = np.load(label_path).astype("int64")
            label = torch.from_numpy(label).unsqueeze(0).unsqueeze(0).float()
            label = F.interpolate(
                label,
                size=(self.image_size, self.image_size),
                mode="nearest",
            ).squeeze(0).squeeze(0).long()
        else:
            label = torch.full((self.image_size, self.image_size), -1, dtype=torch.long)
        return image, label, image_path.name


def load_config(config_path: Path) -> dict:
    with config_path.open("rb") as f:
        return tomllib.load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("config.toml"),
    )
    parser.add_argument("--split", choices=["train", "val", "test"], default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def resolve_model_cfg(model_cfg: dict, checkpoint: dict) -> dict:
    saved_cfg = checkpoint.get("config", {}).get("model")
    if saved_cfg:
        return saved_cfg
    return model_cfg


def load_model(checkpoint_path: Path, model_cfg: dict, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_cfg = resolve_model_cfg(model_cfg, checkpoint)
    in_channels = int(model_cfg.get("in_channels", 1))
    if bool(model_cfg.get("repeat_gray_to_rgb", False)):
        in_channels = 3
    model = PSC_UNet(
        in_channels=in_channels,
        out_channels=int(model_cfg["class_nums"]),
        base_dim=int(model_cfg["base_dim"]),
        swin_depths=tuple(int(v) for v in model_cfg.get("swin_depths", [1, 1, 1, 1])),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def surface_mask(mask: np.ndarray) -> np.ndarray:
    if mask.sum() == 0:
        return mask.astype(bool)
    eroded = binary_erosion(mask.astype(bool))
    return mask.astype(bool) & ~eroded


def compute_hd95(pred: np.ndarray, target: np.ndarray) -> float:
    pred = pred.astype(bool)
    target = target.astype(bool)
    if pred.sum() == 0 and target.sum() == 0:
        return 0.0
    if pred.sum() == 0 or target.sum() == 0:
        return float("inf")

    pred_surface = surface_mask(pred)
    target_surface = surface_mask(target)
    if pred_surface.sum() == 0 or target_surface.sum() == 0:
        return float("inf")

    distances_pred_to_gt = distance_transform_edt(~target)[pred_surface]
    distances_gt_to_pred = distance_transform_edt(~pred)[target_surface]
    return float(max(np.percentile(distances_pred_to_gt, 95), np.percentile(distances_gt_to_pred, 95)))


def _fmt_metric(value: float | None, percent: bool = False) -> str:
    if value is None:
        return "N/A"
    if percent:
        return f"{value * 100:.1f}%"
    return f"{value:.3f}"


def _organ_slice_metrics(
    pred: np.ndarray,
    label: np.ndarray,
    total_pixels: int,
) -> dict[int, dict[str, Any]]:
    results: dict[int, dict[str, Any]] = {}
    for class_id in ORGAN_METRICS:
        pred_mask = pred == class_id
        target_mask = label == class_id
        if target_mask.sum() == 0 and pred_mask.sum() == 0:
            continue
        metrics = compute_binary_metrics(pred_mask, target_mask, total_pixels)
        hd95 = compute_hd95(pred_mask, target_mask)
        metrics["hd95"] = hd95 if np.isfinite(hd95) else None
        results[class_id] = metrics
    return results


def _metrics_to_row(prefix: str, organ_name: str, metrics: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {"organ": organ_name}
    for key in METRIC_KEYS:
        value = metrics.get(key)
        row[f"{prefix}_{key}"] = value if value is not None else ""
    return row


def _pred_only_stats(pred: np.ndarray, total_pixels: int) -> dict[int, dict[str, Any]]:
    stats: dict[int, dict[str, Any]] = {}
    for class_id in ORGAN_METRICS:
        area = int((pred == class_id).sum())
        if area == 0:
            continue
        stats[class_id] = {
            "pred_area": area,
            "pred_percent": float(area / total_pixels * 100.0),
        }
    return stats


def save_overlay(
    image: np.ndarray,
    label: np.ndarray | None,
    pred: np.ndarray,
    organ_metrics: dict[int, dict[str, Any]] | None,
    output_path: Path,
    has_label: bool = True,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    vis_colors = {cid: ORGAN_COLORS.get(cid, "#95A5A6") for cid in ORGAN_METRICS}

    def overlay(base: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
        rgb = np.stack([base, base, base], axis=-1)
        for class_id, color in vis_colors.items():
            region = mask == class_id
            if not np.any(region):
                continue
            color_rgb = np.array(plt.matplotlib.colors.to_rgb(color))
            rgb[region] = (1 - alpha) * rgb[region] + alpha * color_rgb
        return np.clip(rgb, 0, 1)

    if has_label:
        fig = plt.figure(figsize=(18, 6))
        gs = GridSpec(1, 4, width_ratios=[1, 1, 1, 1.15], wspace=0.08)
        ax_img = fig.add_subplot(gs[0])
        ax_gt = fig.add_subplot(gs[1])
        ax_pred = fig.add_subplot(gs[2])
        ax_metrics = fig.add_subplot(gs[3])
    else:
        fig = plt.figure(figsize=(15, 5))
        gs = GridSpec(1, 3, width_ratios=[1, 1, 1.1], wspace=0.08)
        ax_img = fig.add_subplot(gs[0])
        ax_gt = None
        ax_pred = fig.add_subplot(gs[1])
        ax_metrics = fig.add_subplot(gs[2])

    ax_img.imshow(image, cmap="gray", vmin=0, vmax=1)
    ax_img.set_title("Image")
    ax_img.axis("off")
    if ax_gt is not None and label is not None:
        ax_gt.imshow(overlay(image, label))
        ax_gt.set_title("Ground Truth")
        ax_gt.axis("off")
    ax_pred.imshow(overlay(image, pred))
    ax_pred.set_title("Prediction")
    ax_pred.axis("off")

    lines = ["Per-organ metrics", "=" * 34]
    if not has_label:
        lines = ["Prediction volume (no GT)", "=" * 34]
        pred_stats = _pred_only_stats(pred, pred.size)
        for class_id, stats in pred_stats.items():
            organ_name = ORGAN_METRICS[class_id]
            lines.append(
                f"[{organ_name}]  {stats['pred_area']}px  ({stats['pred_percent']:.2f}%)"
            )
        if not pred_stats:
            lines.append("No evaluated organ predicted.")
    elif organ_metrics:
        overall = {key: [] for key in ("dice", "iou", "precision", "recall", "fp_ratio", "fn_ratio")}
        for class_id, metrics in organ_metrics.items():
            organ_name = ORGAN_METRICS[class_id]
            lines.append(f"[{organ_name}]")
            lines.append(
                f"  D {_fmt_metric(metrics['dice'])}  "
                f"IoU {_fmt_metric(metrics['iou'])}  "
                f"P {_fmt_metric(metrics['precision'])}  "
                f"R {_fmt_metric(metrics['recall'])}"
            )
            lines.append(
                f"  GT {metrics['gt_area']}px ({metrics['gt_percent']:.2f}%)  "
                f"Pred {metrics['pred_area']}px ({metrics['pred_percent']:.2f}%)  "
                f"P/GT {_fmt_metric(metrics['pred_gt_ratio'])}"
            )
            fp_text = _fmt_metric(metrics["fp_ratio"], percent=True)
            fn_text = _fmt_metric(metrics["fn_ratio"], percent=True)
            lines.append(f"  FP% {fp_text}  FN% {fn_text}")
            for key in overall:
                if metrics.get(key) is not None:
                    overall[key].append(metrics[key])
            lines.append("")

        lines.append("OVERALL (slice mean)")
        lines.append(
            "  D {d}  IoU {i}  P {p}  R {r}".format(
                d=_fmt_metric(float(np.mean(overall["dice"])) if overall["dice"] else None),
                i=_fmt_metric(float(np.mean(overall["iou"])) if overall["iou"] else None),
                p=_fmt_metric(float(np.mean(overall["precision"])) if overall["precision"] else None),
                r=_fmt_metric(float(np.mean(overall["recall"])) if overall["recall"] else None),
            )
        )
        lines.append(
            "  FP% {fp}  FN% {fn}".format(
                fp=_fmt_metric(float(np.mean(overall["fp_ratio"])) if overall["fp_ratio"] else None, percent=True),
                fn=_fmt_metric(float(np.mean(overall["fn_ratio"])) if overall["fn_ratio"] else None, percent=True),
            )
        )
    else:
        lines.append("No organ present in GT or prediction.")

    ax_metrics.axis("off")
    ax_metrics.text(
        0.02, 0.98, "\n".join(lines),
        transform=ax_metrics.transAxes,
        va="top", ha="left",
        fontsize=8.5, family="monospace",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.92, edgecolor="#CCCCCC"),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def run_split(
    model,
    loader: DataLoader,
    device: torch.device,
    output_dir: Path,
    split: str,
    num_classes: int,
    save_predictions: bool,
    save_visualizations: bool,
    visualize_num: int,
) -> dict:
    split_dir = output_dir / split
    pred_dir = split_dir / "predictions"
    vis_dir = split_dir / "visualizations"
    slice_rows: list[dict[str, Any]] = []
    organ_rows: list[dict[str, Any]] = []
    class_metrics: dict[int, dict[str, list[float | None]]] = {}
    visualized = 0

    for images, labels, names in loader:
        images = images.to(device)
        logits = model(images)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        images_np = images.cpu().numpy()

        for index, name in enumerate(names):
            pred = preds[index]
            image = images_np[index, 0]
            label = labels[index].numpy()
            has_label = int(label.min()) >= 0
            total_pixels = int(pred.size)

            row: dict[str, Any] = {
                "slice_name": name,
                "split": split,
                "has_label": has_label,
                "total_pixels": total_pixels,
            }

            if save_predictions:
                pred_dir.mkdir(parents=True, exist_ok=True)
                np.save(pred_dir / name, pred.astype(np.int16))

            organ_slice_metrics: dict[int, dict[str, Any]] = {}
            if has_label:
                organ_slice_metrics = _organ_slice_metrics(pred, label, total_pixels)
                slice_overall = {key: [] for key in ("dice", "iou", "precision", "recall", "fp_ratio", "fn_ratio")}
                for class_id, metrics in organ_slice_metrics.items():
                    organ_name = ORGAN_METRICS[class_id]
                    class_metrics.setdefault(class_id, {key: [] for key in METRIC_KEYS})
                    for key in METRIC_KEYS:
                        value = metrics.get(key)
                        class_metrics[class_id][key].append(value)
                        row[f"{key}_{organ_name}"] = value if value is not None else ""
                    for key in slice_overall:
                        if metrics.get(key) is not None:
                            slice_overall[key].append(metrics[key])
                    organ_rows.append({
                        "slice_name": name,
                        "split": split,
                        "organ": organ_name,
                        "organ_cn": ORGAN_NAMES_CN[organ_name],
                        **{key: metrics.get(key, "") for key in METRIC_KEYS},
                    })

                for key, values in slice_overall.items():
                    row[f"mean_{key}"] = float(np.mean(values)) if values else ""

                if save_visualizations and visualized < visualize_num:
                    save_overlay(
                        image, label, pred, organ_slice_metrics,
                        vis_dir / f"{Path(name).stem}.png",
                        has_label=True,
                    )
                    visualized += 1
            elif save_visualizations and visualized < visualize_num:
                save_overlay(
                    image, None, pred, None,
                    vis_dir / f"{Path(name).stem}.png",
                    has_label=False,
                )
                visualized += 1

            slice_rows.append(row)

    summary = summarize_metrics(class_metrics, num_classes)
    summary["split"] = split
    summary["num_slices"] = len(slice_rows)
    summary["num_labeled_slices"] = sum(1 for row in slice_rows if row["has_label"])

    split_dir.mkdir(parents=True, exist_ok=True)

    if slice_rows:
        with (split_dir / "slice_metrics.csv").open("w", newline="", encoding="utf-8") as f:
            fieldnames = sorted({key for row in slice_rows for key in row})
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(slice_rows)

    if organ_rows:
        with (split_dir / "organ_slice_metrics.csv").open("w", newline="", encoding="utf-8") as f:
            fieldnames = sorted({key for row in organ_rows for key in row})
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(organ_rows)

    with (split_dir / "organ_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "organ", "organ_cn", "num_slices",
                "dice_mean", "dice_percent", "iou_mean", "iou_percent",
                "precision_mean", "precision_percent", "recall_mean", "recall_percent",
                "fp_ratio_mean", "fp_ratio_percent", "fn_ratio_mean", "fn_ratio_percent",
                "hd95_mean", "gt_area_mean", "pred_area_mean",
                "pred_gt_ratio_mean", "gt_percent_mean", "pred_percent_mean",
            ],
        )
        writer.writeheader()
        for organ_name, metrics in summary["per_organ"].items():
            writer.writerow({
                "organ": organ_name,
                "organ_cn": metrics["name_cn"],
                "num_slices": metrics["num_slices"],
                "dice_mean": metrics["dice"],
                "dice_percent": metrics["dice_percent"],
                "iou_mean": metrics["iou"],
                "iou_percent": metrics["iou_percent"],
                "precision_mean": metrics["precision"],
                "precision_percent": metrics["precision_percent"],
                "recall_mean": metrics["recall"],
                "recall_percent": metrics["recall_percent"],
                "fp_ratio_mean": metrics["fp_ratio"],
                "fp_ratio_percent": metrics["fp_ratio_percent"],
                "fn_ratio_mean": metrics["fn_ratio"],
                "fn_ratio_percent": metrics["fn_ratio_percent"],
                "hd95_mean": metrics["hd95"],
                "gt_area_mean": metrics["gt_area_mean"],
                "pred_area_mean": metrics["pred_area_mean"],
                "pred_gt_ratio_mean": metrics["pred_gt_ratio_mean"],
                "gt_percent_mean": metrics["gt_percent_mean"],
                "pred_percent_mean": metrics["pred_percent_mean"],
            })

    with (split_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary


def summarize_metrics(class_metrics: dict[int, dict[str, list[float | None]]], num_classes: int) -> dict:
    per_organ: dict[str, dict[str, Any]] = {}
    pooled: dict[str, list[float | None]] = {key: [] for key in METRIC_KEYS}

    for class_id, organ_name in ORGAN_METRICS.items():
        values = class_metrics.get(class_id, {})
        organ_summary: dict[str, Any] = {
            "name_cn": ORGAN_NAMES_CN[organ_name],
            "num_slices": len(values.get("dice", [])),
        }
        for key in METRIC_KEYS:
            agg = aggregate_metric_lists(values.get(key, []))
            organ_summary[key] = agg["mean"]
            if key in {"dice", "iou", "precision", "recall", "fp_ratio", "fn_ratio"}:
                organ_summary[f"{key}_percent"] = agg["mean_percent"]
            if key in {"gt_area", "pred_area", "pred_gt_ratio", "gt_percent", "pred_percent"}:
                organ_summary[f"{key}_mean"] = agg["mean"]
            if agg["mean"] is not None and key in {"dice", "iou", "precision", "recall", "fp_ratio", "fn_ratio", "hd95"}:
                pooled[key].append(agg["mean"])
        per_organ[organ_name] = organ_summary

    overall: dict[str, Any] = {}
    for key in METRIC_KEYS:
        agg = aggregate_metric_lists(pooled.get(key, []))
        overall[f"mean_{key}"] = agg["mean"]
        if key in {"dice", "iou", "precision", "recall", "fp_ratio", "fn_ratio"}:
            overall[f"mean_{key}_percent"] = agg["mean_percent"]

    return {
        **overall,
        "mean_dice": overall.get("mean_dice"),
        "mean_dice_percent": overall.get("mean_dice_percent"),
        "mean_hd95": overall.get("mean_hd95"),
        "per_organ": per_organ,
        "num_classes": num_classes,
    }


def split_enabled(split: str, configured: list[str] | None, default_splits: list[str]) -> bool:
    if configured is None:
        return split in default_splits
    return split in configured


def print_summary(summary: dict) -> None:
    split = summary["split"]
    print(f"\n===== {split.upper()} =====")
    print(f"切片总数: {summary['num_slices']}")
    print(f"带标签切片: {summary['num_labeled_slices']}")
    if summary.get("mean_dice") is not None:
        print(
            f"平均 Dice: {summary['mean_dice_percent']:.2f}%  "
            f"IoU: {summary.get('mean_iou_percent', 0):.2f}%  "
            f"Prec: {summary.get('mean_precision_percent', 0):.2f}%  "
            f"Rec: {summary.get('mean_recall_percent', 0):.2f}%"
        )
        if summary.get("mean_hd95") is not None:
            print(f"平均 HD95: {summary['mean_hd95']:.2f} mm")
        print(
            f"平均 FP%: {summary.get('mean_fp_ratio_percent', 0):.2f}%  "
            f"FN%: {summary.get('mean_fn_ratio_percent', 0):.2f}%"
        )
    print("各器官指标:")
    for organ_name, metrics in summary["per_organ"].items():
        if metrics["num_slices"] == 0:
            continue
        print(
            f"  {metrics['name_cn']:>6} ({organ_name:<12}) "
            f"Dice={metrics.get('dice_percent', 0) or 0:>6.2f}%  "
            f"IoU={metrics.get('iou_percent', 0) or 0:>6.2f}%  "
            f"P={metrics.get('precision_percent', 0) or 0:>6.2f}%  "
            f"R={metrics.get('recall_percent', 0) or 0:>6.2f}%  "
            f"GT%={metrics.get('gt_percent_mean', 0) or 0:>5.2f}  "
            f"Pred%={metrics.get('pred_percent_mean', 0) or 0:>5.2f}  "
            f"P/GT={metrics.get('pred_gt_ratio_mean', 0) or 0:>5.2f}  "
            f"slices={metrics['num_slices']}"
        )


def merge_evaluation_splits(summary_path: Path, new_splits: list[dict]) -> dict:
    summary: dict[str, Any] = {}
    if summary_path.exists():
        with summary_path.open(encoding="utf-8") as f:
            summary = json.load(f)
    existing = {item["split"]: item for item in summary.get("evaluation", {}).get("splits", [])}
    for item in new_splits:
        existing[item["split"]] = item
    evaluation = summary.get("evaluation", {})
    evaluation["splits"] = [existing[key] for key in sorted(existing)]
    return evaluation


def rebuild_paper_metrics(run_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for split in ("train", "val", "test"):
        summary_path = run_dir / "predictions" / split / "summary.json"
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("mean_dice") is None:
            continue
        row: dict[str, str] = {
            "split": split,
            "Dice (%)": f"{summary['mean_dice_percent']:.2f}",
            "IoU (%)": f"{summary.get('mean_iou_percent', 0):.2f}",
            "Precision (%)": f"{summary.get('mean_precision_percent', 0):.2f}",
            "Recall (%)": f"{summary.get('mean_recall_percent', 0):.2f}",
            "HD95": f"{summary.get('mean_hd95', 0):.2f}",
            "FP ratio (%)": f"{summary.get('mean_fp_ratio_percent', 0):.2f}",
            "FN ratio (%)": f"{summary.get('mean_fn_ratio_percent', 0):.2f}",
        }
        for organ_name, metrics in summary.get("per_organ", {}).items():
            cn = ORGAN_NAMES_CN[organ_name]
            row[f"{cn} Dice (%)"] = (
                f"{metrics['dice_percent']:.2f}" if metrics.get("dice_percent") is not None else ""
            )
            row[f"{cn} IoU (%)"] = (
                f"{metrics['iou_percent']:.2f}" if metrics.get("iou_percent") is not None else ""
            )
        rows.append(row)
    return rows


def main():
    args = parse_args()
    cfg = load_config(args.config)
    dataset_cfg = cfg["dataset"]
    model_cfg = cfg["model"]
    train_cfg = cfg.get("train", {})
    test_cfg = cfg.get("test", {})

    run = resolve_experiment_run(EXP_DIR, cfg)
    data_dir = ROOT / test_cfg.get("data_dir", train_cfg.get("data_dir", "data/synapse_processed"))
    output_dir = run.predictions_dir
    checkpoint_path = run.best_checkpoint
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device or test_cfg.get("device", train_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")))
    image_size = int(dataset_cfg["image_size"])
    batch_size = int(dataset_cfg["batch_size"])
    num_workers = int(dataset_cfg["num_workers"])
    repeat_gray_to_rgb = bool(model_cfg.get("repeat_gray_to_rgb", False))
    splits = [args.split] if args.split else test_cfg.get("splits", ["val", "test"])
    default_prediction_splits = ["val", "test"]
    default_visualization_splits = ["val", "test"]
    prediction_splits = test_cfg.get("save_predictions_splits")
    visualization_splits = test_cfg.get("save_visualizations_splits")
    save_predictions_default = bool(test_cfg.get("save_predictions", True))
    save_visualizations_default = bool(test_cfg.get("save_visualizations", True))
    visualize_num = int(test_cfg.get("visualize_num", 5))

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"未找到 checkpoint: {checkpoint_path}，请先运行 train_psc_unet.py")

    model = load_model(checkpoint_path, model_cfg, device)
    all_summaries = []

    print(f"Experiment: {run.name}")
    print(f"Device: {device}")
    print("Model: PSC-UNet")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Predictions dir: {output_dir}")

    for split in splits:
        save_predictions = save_predictions_default and split_enabled(
            split, prediction_splits, default_prediction_splits
        )
        save_visualizations = save_visualizations_default and split_enabled(
            split, visualization_splits, default_visualization_splits
        )
        dataset = InferenceDataset(data_dir, split, image_size, repeat_gray_to_rgb=repeat_gray_to_rgb)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
        )
        summary = run_split(
            model=model,
            loader=loader,
            device=device,
            output_dir=output_dir,
            split=split,
            num_classes=int(model_cfg["class_nums"]),
            save_predictions=save_predictions,
            save_visualizations=save_visualizations,
            visualize_num=visualize_num,
        )
        all_summaries.append(summary)
        print_summary(summary)

    evaluation = merge_evaluation_splits(run.summary_path, all_summaries)
    evaluation["checkpoint"] = str(checkpoint_path)
    update_evaluation_summary(run.summary_path, evaluation)

    readme_rows = rebuild_paper_metrics(run.run_dir)
    if readme_rows:
        paper_metrics_path = run.run_dir / "paper_metrics.csv"
        fieldnames: list[str] = []
        seen_fields: set[str] = set()
        for row in readme_rows:
            for key in row:
                if key not in seen_fields:
                    fieldnames.append(key)
                    seen_fields.add(key)
        with paper_metrics_path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(readme_rows)
        print(f"\n论文指标表已保存: {paper_metrics_path}")
        print(f"实验摘要已更新: {run.summary_path}")


if __name__ == "__main__":
    main()
