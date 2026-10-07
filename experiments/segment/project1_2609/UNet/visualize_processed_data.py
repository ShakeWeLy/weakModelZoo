"""可视化训练管线处理后的数据（resize + label_map），风格与 dataset_report 质量抽检图一致。

用法（项目根目录）：
    python experiments/segment/project1_2609/UNet/visualize_processed_data.py
    python experiments/segment/project1_2609/UNet/visualize_processed_data.py --splits train
    python experiments/segment/project1_2609/UNet/visualize_processed_data.py --max-samples 8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.gridspec import GridSpec

EXP_DIR = Path(__file__).resolve().parent
ROOT = EXP_DIR.parents[3]
sys.path.insert(0, str(ROOT))

from src.utils.data.synapse.labels import (
    class_color,
    make_overlay,
    metric_class_name,
    apply_label_map,
    remap_label_from_eval_model,
)

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore


def setup_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.unicode_minus": False,
        }
    )


def parse_case_id(slice_name: str) -> str:
    return slice_name.split("_slice", 1)[0]


def load_config(config_path: Path) -> dict:
    with config_path.open("rb") as f:
        return tomllib.load(f)


def process_sample(
    image_path: Path,
    label_path: Path,
    image_size: int,
    label_map: str | None,
) -> tuple[np.ndarray, np.ndarray]:
    """与 SynapseSliceDataset.__getitem__ 使用相同的处理流程。"""
    image = np.load(image_path).astype("float32")
    label = np.load(label_path).astype("int64")
    label = apply_label_map(label, label_map)

    image_t = torch.from_numpy(image).unsqueeze(0)
    label_t = torch.from_numpy(label)

    image_t = F.interpolate(
        image_t.unsqueeze(0),
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)
    label_t = (
        F.interpolate(
            label_t.unsqueeze(0).unsqueeze(0).float(),
            size=(image_size, image_size),
            mode="nearest",
        )
        .squeeze(0)
        .squeeze(0)
        .long()
    )

    return image_t.squeeze(0).numpy(), label_t.numpy()


def label_for_visualization(label: np.ndarray, label_map: str | None) -> np.ndarray:
    if label_map == "eval8":
        return remap_label_from_eval_model(label)
    return label


def present_class_ids(label: np.ndarray) -> list[int]:
    return sorted(int(class_id) for class_id in np.unique(label) if int(class_id) != 0)


def save_sample_figure(
    image: np.ndarray,
    label_viz: np.ndarray,
    split: str,
    case_id: str,
    slice_name: str,
    output_path: Path,
    dpi: int = 200,
) -> None:
    class_ids = present_class_ids(label_viz)
    show_legend = bool(class_ids)
    fig_height = 5.4 if show_legend else 4.2
    fig = plt.figure(figsize=(12, fig_height))
    if show_legend:
        grid = GridSpec(2, 3, height_ratios=[1, 0.14], hspace=0.18)
        axes = [fig.add_subplot(grid[0, index]) for index in range(3)]
        ax_legend = fig.add_subplot(grid[1, :])
    else:
        grid = GridSpec(1, 3)
        axes = [fig.add_subplot(grid[0, index]) for index in range(3)]
        ax_legend = None

    axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Image")
    axes[0].axis("off")

    axes[1].imshow(make_overlay(np.zeros_like(image), label_viz))
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    axes[2].imshow(make_overlay(image, label_viz))
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    if ax_legend is not None:
        legend_handles = [
            mpatches.Patch(
                color=class_color(class_id),
                label=f"{class_id}:{metric_class_name(class_id).replace('_', ' ').title()}",
            )
            for class_id in class_ids
        ]
        ax_legend.axis("off")
        ax_legend.legend(
            handles=legend_handles,
            loc="center",
            ncol=min(7, len(legend_handles)),
            fontsize=7.5,
            frameon=False,
        )

    fig.suptitle(f"{split} | {case_id} | {slice_name}", y=1.02)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=EXP_DIR / "config.toml",
        help="训练配置文件路径",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=EXP_DIR / "processed_data_preview",
        help="输出目录",
    )
    parser.add_argument("--splits", nargs="+", default=["train"], help="要可视化的数据划分")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="每个 split 最多生成多少张；默认全部",
    )
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="跳过已存在的输出图（断点续跑）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    dataset_cfg = cfg.get("dataset", {})
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("train", {})

    data_dir = ROOT / train_cfg.get("data_dir", "data/synapse_processed")
    image_size = int(dataset_cfg.get("image_size", 224))
    label_map = model_cfg.get("label_map")

    setup_plot_style()
    total_saved = 0

    for split in args.splits:
        image_dir = data_dir / split / "images"
        label_dir = data_dir / split / "labels"
        if not image_dir.exists():
            print(f"[skip] {split}: 未找到 {image_dir}")
            continue

        samples = sorted(image_dir.glob("*.npy"))
        if args.max_samples is not None:
            samples = samples[: args.max_samples]
        if not samples:
            print(f"[skip] {split}: 无 .npy 切片")
            continue

        split_dir = args.output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        total = len(samples)

        for index, image_path in enumerate(samples, start=1):
            slice_name = image_path.name
            case_id = parse_case_id(slice_name)
            out_path = split_dir / f"{split}_{Path(slice_name).stem}.png"

            if args.skip_existing and out_path.exists() and out_path.stat().st_size > 0:
                total_saved += 1
                continue

            label_path = label_dir / slice_name
            image, label = process_sample(image_path, label_path, image_size, label_map)
            label_viz = label_for_visualization(label, label_map)
            save_sample_figure(
                image=image,
                label_viz=label_viz,
                split=split,
                case_id=case_id,
                slice_name=slice_name,
                output_path=out_path,
                dpi=args.dpi,
            )
            total_saved += 1

            if index == 1 or index % 25 == 0 or index == total:
                print(f"[{split}] {index}/{total} -> {out_path.name}")

    print(f"完成，共生成 {total_saved} 张图，输出目录: {args.output_dir}")


if __name__ == "__main__":
    main()
