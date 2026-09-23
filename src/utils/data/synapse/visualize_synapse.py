"""可视化 Synapse 2D 切片数据，生成论文用图。

用法：
    python src/utils/data/synapse/visualize_synapse.py --case 0001 --slice 0020
    python src/utils/data/synapse/visualize_synapse.py --paper-figure
    python src/utils/data/synapse/visualize_synapse.py --stats-figure
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.colors import ListedColormap

from labels import FOREGROUND_CLASS_IDS, LABEL_NAMES, ORGAN_COLORS, class_color, make_overlay

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DATA_DIR = PROJECT_ROOT / "data" / "synapse_processed"

ORGAN_NAMES = {class_id: LABEL_NAMES[class_id].replace("_", " ").title() for class_id in FOREGROUND_CLASS_IDS}


def setup_paper_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    })


def load_split_info() -> list[dict]:
    rows = []
    with (DATA_DIR / "case_split.csv").open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def find_slice(split: str, case_id: str, slice_id: str | None = None) -> tuple[Path, Path | None]:
    image_dir = DATA_DIR / split / "images"
    label_dir = DATA_DIR / split / "labels"
    if slice_id:
        name = f"{case_id}_slice{slice_id}.npy"
        image_path = image_dir / name
        if not image_path.exists():
            raise FileNotFoundError(f"未找到切片: {image_path}")
        label_path = label_dir / name if label_dir.exists() else None
        return image_path, label_path if label_path and label_path.exists() else None

    candidates = sorted(image_dir.glob(f"{case_id}_slice*.npy"))
    if not candidates:
        raise FileNotFoundError(f"病例 {case_id} 在 {split} 中无切片")
    image_path = candidates[len(candidates) // 2]
    label_path = label_dir / image_path.name if label_dir.exists() else None
    return image_path, label_path if label_path and label_path.exists() else None


def plot_legend(ax):
    patches = [
        mpatches.Patch(color=color, label=ORGAN_NAMES[class_id])
        for class_id, color in ORGAN_COLORS.items()
    ]
    ax.legend(handles=patches, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)


def plot_single(case_id: str, slice_id: str | None, split: str, output: Path):
    setup_paper_style()
    image_path, label_path = find_slice(split, case_id, slice_id)
    image = np.load(image_path)
    label = np.load(label_path) if label_path else None

    if label is not None:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)
        axes[0].set_title("(a) CT Image")
        axes[0].axis("off")

        present_ids = sorted(int(v) for v in np.unique(label))
        gt_colors = [ORGAN_COLORS.get(class_id, class_color(class_id)) for class_id in present_ids]
        axes[1].imshow(label, cmap=ListedColormap(gt_colors), vmin=min(present_ids), vmax=max(present_ids))
        axes[1].set_title("(b) Ground Truth")
        axes[1].axis("off")

        axes[2].imshow(make_overlay(image, label))
        axes[2].set_title("(c) Overlay")
        axes[2].axis("off")
        plot_legend(axes[2])
    else:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(image, cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"Test Case {case_id}")
        ax.axis("off")

    fig.suptitle(f"{split.upper()} | Case {case_id} | {image_path.stem}", y=1.02)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)
    print(f"已保存: {output}")


def plot_paper_figure(output: Path, samples_per_split: int = 2):
    setup_paper_style()
    split_info = load_split_info()
    splits = ["train", "val", "test"]
    selected = {split: [] for split in splits}
    for row in split_info:
        split = row["split"]
        if len(selected[split]) < samples_per_split:
            selected[split].append(row["case_id"])

    fig, axes = plt.subplots(len(splits), samples_per_split * 2, figsize=(4 * samples_per_split, 9))
    if len(splits) == 1:
        axes = np.array([axes])

    for row_idx, split in enumerate(splits):
        for col_idx, case_id in enumerate(selected[split]):
            image_path, label_path = find_slice(split, case_id)
            image = np.load(image_path)
            label = np.load(label_path) if label_path else None

            ax_img = axes[row_idx, col_idx * 2]
            ax_img.imshow(image, cmap="gray", vmin=0, vmax=1)
            ax_img.set_title(f"{split.capitalize()} | Case {case_id}")
            ax_img.axis("off")

            ax_ov = axes[row_idx, col_idx * 2 + 1]
            if label is not None:
                ax_ov.imshow(make_overlay(image, label))
                ax_ov.set_title("Overlay")
            else:
                ax_ov.imshow(image, cmap="gray", vmin=0, vmax=1)
                ax_ov.set_title("No Label")
            ax_ov.axis("off")

    patches = [
        mpatches.Patch(color=color, label=ORGAN_NAMES[class_id])
        for class_id, color in ORGAN_COLORS.items()
    ]
    fig.legend(handles=patches, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Synapse Dataset Samples", y=1.01, fontsize=13)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)
    print(f"已保存: {output}")


def plot_stats_figure(output: Path):
    setup_paper_style()
    with (DATA_DIR / "case_statistics.json").open(encoding="utf-8") as f:
        stats = json.load(f)

    splits = ["train", "val", "test"]
    split_counts = {split: 0 for split in splits}
    slice_counts = {split: 0 for split in splits}
    organ_totals = {name: 0 for name in ORGAN_NAMES.values()}

    for item in stats:
        split = item["split"]
        split_counts[split] += 1
        slice_counts[split] += item.get("non_empty_slices", item.get("slices", 0))
        for organ, count in item.get("organ_voxels", {}).items():
            organ_totals[organ.replace("_", " ").title()] = organ_totals.get(
                organ.replace("_", " ").title(), 0
            ) + count

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].bar(split_counts.keys(), split_counts.values(), color=["#4C72B0", "#55A868", "#C44E52"])
    axes[0].set_title("(a) Cases per Split")
    axes[0].set_ylabel("Number of Cases")

    axes[1].bar(slice_counts.keys(), slice_counts.values(), color=["#4C72B0", "#55A868", "#C44E52"])
    axes[1].set_title("(b) 2D Slices per Split")
    axes[1].set_ylabel("Number of Slices")

    organ_names = list(ORGAN_NAMES.values())
    organ_values = [organ_totals.get(name, 0) for name in organ_names]
    axes[2].barh(organ_names, organ_values, color=list(ORGAN_COLORS.values()))
    axes[2].set_title("(c) Organ Voxel Distribution (Train+Val)")
    axes[2].set_xlabel("Voxel Count")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)
    print(f"已保存: {output}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--case", default="0001")
    parser.add_argument("--slice", default=None, help="例如 0020")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "figures" / "synapse")
    parser.add_argument("--paper-figure", action="store_true")
    parser.add_argument("--stats-figure", action="store_true")
    return parser.parse_args()


def main():
    global DATA_DIR
    args = parse_args()
    DATA_DIR = args.data_dir

    if args.paper_figure:
        plot_paper_figure(args.output_dir / "synapse_paper_samples.png")
    if args.stats_figure:
        plot_stats_figure(args.output_dir / "synapse_dataset_stats.png")
    if not args.paper_figure and not args.stats_figure:
        suffix = f"_{args.slice}" if args.slice else ""
        plot_single(
            args.case,
            args.slice,
            args.split,
            args.output_dir / f"{args.split}_{args.case}{suffix}.png",
        )


if __name__ == "__main__":
    main()
