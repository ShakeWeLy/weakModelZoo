"""Attention U-Net 推理与评估：计算 Dice、HD95，并保存预测结果。

用法（在项目根目录执行）：
    python experiments/segment/project1_2609/AttnUNet/analysis_attn_unet.py
    python experiments/segment/project1_2609/AttnUNet/analysis_attn_unet.py --split test
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import binary_erosion, distance_transform_edt
from torch.utils.data import DataLoader, Dataset

EXP_DIR = Path(__file__).resolve().parent
ROOT = EXP_DIR.parents[3]
sys.path.insert(0, str(ROOT))

from src.models.segment.AttentionGateUnet.AttentionGateUnet import AttentionGateUnet
from src.utils.logger import resolve_experiment_run, update_evaluation_summary

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore


ORGAN_METRICS = {
    1: "spleen",
    2: "right_kidney",
    3: "left_kidney",
    4: "gallbladder",
    6: "liver",
    7: "stomach",
    8: "aorta",
    11: "pancreas",
}

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


class InferenceDataset(Dataset):
    def __init__(self, data_dir: Path, split: str, image_size: int):
        self.image_dir = data_dir / split / "images"
        self.label_dir = data_dir / split / "labels"
        self.image_size = image_size
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
    parser.add_argument("--split", choices=["val", "test"], default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def load_model(checkpoint_path: Path, model_cfg: dict, device: torch.device) -> AttentionGateUnet:
    model = AttentionGateUnet(
        in_channels=1,
        out_channels=int(model_cfg["class_nums"]),
        hidden_channels=int(model_cfg["hidden_channels"]),
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def surface_mask(mask: np.ndarray) -> np.ndarray:
    if mask.sum() == 0:
        return mask.astype(bool)
    eroded = binary_erosion(mask.astype(bool))
    return mask.astype(bool) & ~eroded


def compute_dice(pred: np.ndarray, target: np.ndarray, eps: float = 1e-6) -> float:
    pred = pred.astype(bool)
    target = target.astype(bool)
    if pred.sum() == 0 and target.sum() == 0:
        return 1.0
    if pred.sum() == 0 or target.sum() == 0:
        return 0.0
    intersection = np.logical_and(pred, target).sum()
    return float((2.0 * intersection + eps) / (pred.sum() + target.sum() + eps))


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


def save_overlay(image: np.ndarray, label: np.ndarray, pred: np.ndarray, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    colors = {
        1: "#E74C3C",
        2: "#2ECC71",
        3: "#3498DB",
        4: "#F1C40F",
        6: "#E67E22",
        7: "#1ABC9C",
        8: "#EC407A",
        11: "#9B59B6",
    }

    def overlay(base: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
        rgb = np.stack([base, base, base], axis=-1)
        for class_id, color in colors.items():
            region = mask == class_id
            if not np.any(region):
                continue
            color_rgb = np.array(plt.matplotlib.colors.to_rgb(color))
            rgb[region] = (1 - alpha) * rgb[region] + alpha * color_rgb
        return np.clip(rgb, 0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Image")
    axes[0].axis("off")
    axes[1].imshow(overlay(image, label))
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")
    axes[2].imshow(overlay(image, pred))
    axes[2].set_title("Prediction")
    axes[2].axis("off")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def run_split(
    model: AttentionGateUnet,
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
    slice_rows = []
    class_metrics: dict[int, dict[str, list[float]]] = {}
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

            row = {
                "slice_name": name,
                "split": split,
                "has_label": has_label,
            }

            if save_predictions:
                pred_dir.mkdir(parents=True, exist_ok=True)
                np.save(pred_dir / name, pred.astype(np.int16))

            if has_label:
                for class_id in ORGAN_METRICS:
                    pred_mask = pred == class_id
                    target_mask = label == class_id
                    if target_mask.sum() == 0:
                        continue
                    class_metrics.setdefault(class_id, {"dice": [], "hd95": []})
                    dice = compute_dice(pred_mask, target_mask)
                    hd95 = compute_hd95(pred_mask, target_mask)
                    class_metrics[class_id]["dice"].append(dice)
                    if np.isfinite(hd95):
                        class_metrics[class_id]["hd95"].append(hd95)
                    row[f"dice_{ORGAN_METRICS[class_id]}"] = dice
                    row[f"hd95_{ORGAN_METRICS[class_id]}"] = hd95 if np.isfinite(hd95) else ""

                if save_visualizations and visualized < visualize_num:
                    save_overlay(image, label, pred, vis_dir / f"{Path(name).stem}.png")
                    visualized += 1

            slice_rows.append(row)

    summary = summarize_metrics(class_metrics, num_classes)
    summary["split"] = split
    summary["num_slices"] = len(slice_rows)
    summary["num_labeled_slices"] = sum(1 for row in slice_rows if row["has_label"])

    split_dir = output_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)

    with (split_dir / "slice_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        if slice_rows:
            fieldnames = sorted({key for row in slice_rows for key in row})
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(slice_rows)

    with (split_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary


def summarize_metrics(class_metrics: dict[int, dict[str, list[float]]], num_classes: int) -> dict:
    per_organ = {}
    dice_values = []
    hd95_values = []

    for class_id, organ_name in ORGAN_METRICS.items():
        values = class_metrics.get(class_id, {"dice": [], "hd95": []})
        organ_dice = values.get("dice", [])
        organ_hd95 = values.get("hd95", [])
        per_organ[organ_name] = {
            "name_cn": ORGAN_NAMES_CN[organ_name],
            "dice": float(np.mean(organ_dice)) if organ_dice else None,
            "dice_percent": float(np.mean(organ_dice) * 100) if organ_dice else None,
            "hd95": float(np.mean(organ_hd95)) if organ_hd95 else None,
            "num_slices": len(organ_dice),
        }
        if organ_dice:
            dice_values.extend(organ_dice)
        if organ_hd95:
            hd95_values.extend(organ_hd95)

    return {
        "mean_dice": float(np.mean(dice_values)) if dice_values else None,
        "mean_dice_percent": float(np.mean(dice_values) * 100) if dice_values else None,
        "mean_hd95": float(np.mean(hd95_values)) if hd95_values else None,
        "per_organ": per_organ,
        "num_classes": num_classes,
    }


def print_summary(summary: dict) -> None:
    split = summary["split"]
    print(f"\n===== {split.upper()} =====")
    print(f"切片总数: {summary['num_slices']}")
    print(f"带标签切片: {summary['num_labeled_slices']}")
    if summary["mean_dice"] is not None:
        print(f"平均 Dice: {summary['mean_dice_percent']:.2f}%")
        print(f"平均 HD95: {summary['mean_hd95']:.2f} mm")
    print("各器官指标:")
    for organ_name, metrics in summary["per_organ"].items():
        dice = metrics["dice_percent"]
        hd95 = metrics["hd95"]
        dice_text = f"{dice:.2f}%" if dice is not None else "N/A"
        hd95_text = f"{hd95:.2f}" if hd95 is not None else "N/A"
        print(f"  {metrics['name_cn']:>6} ({organ_name:<12}) Dice={dice_text:>8}  HD95={hd95_text:>8}  slices={metrics['num_slices']}")


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
    splits = [args.split] if args.split else test_cfg.get("splits", ["val", "test"])
    save_predictions = bool(test_cfg.get("save_predictions", True))
    save_visualizations = bool(test_cfg.get("save_visualizations", True))
    visualize_num = int(test_cfg.get("visualize_num", 5))

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"未找到 checkpoint: {checkpoint_path}，请先运行 train_attn_unet.py")

    model = load_model(checkpoint_path, model_cfg, device)
    all_summaries = []

    print(f"Experiment: {run.name}")
    print(f"Device: {device}")
    print(f"Model: AttentionGateUnet")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Predictions dir: {output_dir}")

    for split in splits:
        dataset = InferenceDataset(data_dir, split, image_size)
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

    evaluation = {
        "checkpoint": str(checkpoint_path),
        "splits": all_summaries,
    }
    update_evaluation_summary(run.summary_path, evaluation)

    readme_rows = []
    for summary in all_summaries:
        if summary["mean_dice"] is None:
            continue
        row = {
            "split": summary["split"],
            "Dice (%)": f"{summary['mean_dice_percent']:.2f}",
            "HD95": f"{summary['mean_hd95']:.2f}",
        }
        for organ_name, metrics in summary["per_organ"].items():
            row[ORGAN_NAMES_CN[organ_name] + " (%)"] = (
                f"{metrics['dice_percent']:.2f}" if metrics["dice_percent"] is not None else ""
            )
        readme_rows.append(row)

    if readme_rows:
        paper_metrics_path = run.run_dir / "paper_metrics.csv"
        with paper_metrics_path.open("w", newline="", encoding="utf-8-sig") as f:
            fieldnames = list(readme_rows[0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(readme_rows)
        print(f"\n论文指标表已保存: {paper_metrics_path}")
        print(f"实验摘要已更新: {run.summary_path}")


if __name__ == "__main__":
    main()
