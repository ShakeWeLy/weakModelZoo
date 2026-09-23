"""分析 Synapse 3D NIfTI 数据，生成 train/val/test 2D 数据集。

用法：
    python src/utils/data/synapse/analyze_synapse.py --save-slices
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom

from labels import LABEL_NAMES

PROJECT_ROOT = Path(__file__).resolve().parents[4]


def parse_args() -> argparse.Namespace:
    root = PROJECT_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-dir", type=Path,
                        default=root / "data" / "synapse_RawData" / "Training")
    parser.add_argument("--testing-dir", type=Path,
                        default=root / "data" / "synapse_RawData" / "Testing")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "data" / "synapse_processed",
    )
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-spacing", type=float, default=1.0)
    parser.add_argument("--save-slices", action="store_true")
    return parser.parse_args()


def percentile_normalize(image: np.ndarray) -> np.ndarray:
    image = np.clip(image, -1250.0, 1500.0)
    low, high = np.percentile(image, [1, 99])
    if high <= low:
        return np.zeros_like(image, dtype=np.float32)
    image = np.clip(image, low, high)
    return ((image - low) / (high - low)).astype(np.float32)


def resample_to_1mm(
    image: np.ndarray, label: np.ndarray, affine: np.ndarray, spacing: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    zoom_factors = np.asarray(spacing) / np.sqrt((affine[:3, :3] ** 2).sum(axis=0))
    if np.allclose(zoom_factors, 1.0, atol=1e-3):
        return image, label, affine
    image = zoom(image, zoom_factors, order=1)
    label = zoom(label, zoom_factors, order=0)
    new_affine = affine.copy()
    directions = affine[:3, :3] / np.linalg.norm(affine[:3, :3], axis=0)
    new_affine[:3, :3] = directions * spacing
    return image, label.astype(np.int16), new_affine


def case_summary(case_id: str, image_path: Path, label_path: Path) -> dict:
    image_nii = nib.load(str(image_path))
    label_nii = nib.load(str(label_path))
    image = image_nii.get_fdata(dtype=np.float32)
    label = label_nii.get_fdata(dtype=np.float32).astype(np.int16)
    if image.shape != label.shape:
        raise ValueError(f"{case_id}: image 与 label shape 不一致")
    spacing = np.sqrt((image_nii.affine[:3, :3] ** 2).sum(axis=0))
    labels, counts = np.unique(label, return_counts=True)
    organ_voxels = {
        LABEL_NAMES.get(int(k), f"class_{int(k)}"): int(v)
        for k, v in zip(labels, counts)
        if int(k) != 0
    }
    return {
        "case_id": case_id,
        "shape": list(image.shape),
        "slices": int(image.shape[2]),
        "spacing_mm": spacing.round(4).tolist(),
        "hu_min": float(image.min()),
        "hu_max": float(image.max()),
        "hu_mean": float(image.mean()),
        "label_values": [int(x) for x in labels],
        "organ_voxels": organ_voxels,
        "image_path": str(image_path),
        "label_path": str(label_path),
    }


def stratified_split(summaries: list[dict], val_ratio: float, seed: int):
    rng = np.random.default_rng(seed)
    order = np.arange(len(summaries))
    rng.shuffle(order)
    ranked = sorted(
        order,
        key=lambda i: sum(summaries[i]["organ_voxels"].values()),
    )
    val_count = max(1, round(len(summaries) * val_ratio))
    val_indices = set(ranked[:: max(1, len(ranked) // val_count)][:val_count])
    if len(val_indices) < val_count:
        val_indices.update(ranked[-(val_count - len(val_indices)) :])
    return (
        [summaries[i] for i in range(len(summaries)) if i not in val_indices],
        [summaries[i] for i in range(len(summaries)) if i in val_indices],
    )


def save_slices(summary: dict, split: str, output_dir: Path, target_spacing: float):
    image_nii = nib.load(summary["image_path"])
    image = image_nii.get_fdata(dtype=np.float32)
    if summary["label_path"] is not None:
        label_nii = nib.load(summary["label_path"])
        label = label_nii.get_fdata(dtype=np.float32).astype(np.int16)
        image, label, _ = resample_to_1mm(image, label, image_nii.affine, target_spacing)
    else:
        zoom_factors = target_spacing / np.sqrt((image_nii.affine[:3, :3] ** 2).sum(axis=0))
        image = zoom(image, zoom_factors, order=1)
        label = None
    image = percentile_normalize(image)
    split_dir = output_dir / split
    image_dir, label_dir = split_dir / "images", split_dir / "labels"
    image_dir.mkdir(parents=True, exist_ok=True)
    if label is not None:
        label_dir.mkdir(parents=True, exist_ok=True)
    kept = 0
    for index in range(image.shape[2]):
        mask = label[:, :, index] if label is not None else None
        if mask is not None and not np.any(mask > 0):
            continue
        np.save(image_dir / f"{summary['case_id']}_slice{index:04d}.npy", image[:, :, index])
        if mask is not None:
            np.save(label_dir / f"{summary['case_id']}_slice{index:04d}.npy", mask)
        kept += 1
    return kept


def collect_cases(data_dir: Path, require_label: bool = True) -> list[dict]:
    image_dir, label_dir = data_dir / "img", data_dir / "label"
    summaries = []
    for image_path in sorted(image_dir.glob("*.nii.gz")):
        case_id = image_path.name.removeprefix("img").removesuffix(".nii.gz")
        label_path = label_dir / f"label{case_id}.nii.gz"
        if label_path.exists():
            summaries.append(case_summary(case_id, image_path, label_path))
        elif not require_label:
            image_nii = nib.load(str(image_path))
            spacing = np.sqrt((image_nii.affine[:3, :3] ** 2).sum(axis=0))
            summaries.append({
                "case_id": case_id,
                "shape": list(image_nii.shape),
                "slices": int(image_nii.shape[2]),
                "spacing_mm": spacing.round(4).tolist(),
                "image_path": str(image_path),
                "label_path": None,
            })
    return summaries


def main():
    args = parse_args()
    training = collect_cases(args.training_dir)
    testing = collect_cases(args.testing_dir, require_label=False)
    if not training:
        raise FileNotFoundError(f"训练目录未找到配对数据: {args.training_dir}")
    if not testing:
        raise FileNotFoundError(f"测试目录未找到配对数据: {args.testing_dir}")
    train, val = stratified_split(training, args.val_ratio, args.seed)
    summaries = []
    for item in train:
        item["split"] = "train"
        summaries.append(item)
    for item in val:
        item["split"] = "val"
        summaries.append(item)
    for item in testing:
        item["split"] = "test"
        summaries.append(item)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_ids = {item["case_id"] for item in train}
    val_ids = {item["case_id"] for item in val}
    for item in summaries:
        if item["case_id"] in train_ids:
            item["split"] = "train"
        elif item["case_id"] in val_ids:
            item["split"] = "val"
        else:
            item["split"] = "test"
        if args.save_slices:
            item["non_empty_slices"] = save_slices(
                item, item["split"], args.output_dir, args.target_spacing
            )
    with (args.output_dir / "case_statistics.json").open("w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)
    with (args.output_dir / "case_split.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["case_id", "split", "slices", "shape", "spacing_mm"])
        writer.writeheader()
        for item in summaries:
            writer.writerow({key: item[key] for key in writer.fieldnames})
    print(f"训练集: {len(train)} 例，验证集: {len(val)} 例，测试集: {len(testing)} 例")
    print(f"结果已保存到: {args.output_dir}")


if __name__ == "__main__":
    main()
