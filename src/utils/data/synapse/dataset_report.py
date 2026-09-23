"""Synapse 处理后 2D 数据集的初始情况分析报告。

用法（项目根目录）：
    python src/utils/data/synapse/dataset_report.py
    python src/utils/data/synapse/dataset_report.py --data-dir data/synapse_processed --quality-samples 8
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

from labels import LABEL_NAMES, LABEL_NAMES_CN, ORGAN_COLORS, class_color, make_overlay

PROJECT_ROOT = Path(__file__).resolve().parents[4]


def setup_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.unicode_minus": False,
        }
    )


def display_name_en(class_id: int | None = None, name: str | None = None) -> str:
    if name in {"background", "foreground"}:
        return name.title()
    label = name or LABEL_NAMES.get(int(class_id), f"class_{class_id}")
    return label.replace("_", " ").title()


@dataclass
class SliceRecord:
    split: str
    case_id: str
    slice_name: str
    image_path: Path
    label_path: Path | None
    image_shape: tuple[int, ...]
    has_label: bool


@dataclass
class ClassAccumulator:
    case_ids: set[str] = field(default_factory=set)
    slice_hits: int = 0
    total_pixels: int = 0
    instance_areas: list[int] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data" / "synapse_processed")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--quality-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tiny-area-threshold", type=int, default=16)
    parser.add_argument("--fragmentation-threshold", type=int, default=5)
    return parser.parse_args()


def project_root() -> Path:
    return PROJECT_ROOT


def parse_case_id(slice_name: str) -> str:
    return slice_name.split("_slice", 1)[0]


def load_case_split(data_dir: Path) -> dict[str, dict[str, str]]:
    split_path = data_dir / "case_split.csv"
    if not split_path.exists():
        return {}
    rows: dict[str, dict[str, str]] = {}
    with split_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows[row["case_id"]] = row
    return rows


def load_case_statistics(data_dir: Path) -> list[dict]:
    stats_path = data_dir / "case_statistics.json"
    if not stats_path.exists():
        return []
    with stats_path.open(encoding="utf-8") as f:
        return json.load(f)


def collect_slice_records(data_dir: Path, splits: list[str]) -> list[SliceRecord]:
    records: list[SliceRecord] = []
    for split in splits:
        image_dir = data_dir / split / "images"
        label_dir = data_dir / split / "labels"
        if not image_dir.exists():
            continue
        for image_path in sorted(image_dir.glob("*.npy")):
            label_path = label_dir / image_path.name if label_dir.exists() else None
            if label_path is not None and not label_path.exists():
                label_path = None
            image = np.load(image_path, mmap_mode="r")
            records.append(
                SliceRecord(
                    split=split,
                    case_id=parse_case_id(image_path.name),
                    slice_name=image_path.name,
                    image_path=image_path,
                    label_path=label_path,
                    image_shape=tuple(int(v) for v in image.shape),
                    has_label=label_path is not None,
                )
            )
    return records


def instance_areas(label: np.ndarray, class_id: int) -> list[int]:
    mask = label == class_id
    if not np.any(mask):
        return []
    labeled, count = ndimage.label(mask)
    if count == 0:
        return []
    return [int(np.sum(labeled == index)) for index in range(1, count + 1)]


def analyze_label_slice(
    record: SliceRecord,
    accumulators: dict[int, ClassAccumulator],
    case_presence: dict[str, set[int]],
    quality_issues: list[dict],
    tiny_area_threshold: int,
    fragmentation_threshold: int,
) -> dict[str, int]:
    label = np.load(record.label_path)
    unique_values = [int(v) for v in np.unique(label)]
    pixel_counter = Counter(int(v) for v in label.reshape(-1))
    total_pixels = int(label.size)
    foreground_pixels = int(sum(count for value, count in pixel_counter.items() if value != 0))

    if foreground_pixels == 0:
        quality_issues.append(
            {
                "type": "empty_label",
                "split": record.split,
                "case_id": record.case_id,
                "slice_name": record.slice_name,
                "message": "标签全为背景",
            }
        )

    unknown = [value for value in unique_values if value not in LABEL_NAMES]
    if unknown:
        quality_issues.append(
            {
                "type": "unknown_label",
                "split": record.split,
                "case_id": record.case_id,
                "slice_name": record.slice_name,
                "values": unknown,
            }
        )

    for class_id in unique_values:
        if class_id == 0:
            continue
        areas = instance_areas(label, class_id)
        if not areas:
            continue
        case_presence[record.case_id].add(class_id)
        acc = accumulators[class_id]
        acc.case_ids.add(record.case_id)
        acc.slice_hits += 1
        acc.total_pixels += int(sum(areas))
        acc.instance_areas.extend(areas)

        tiny = [area for area in areas if area < tiny_area_threshold]
        if tiny:
            quality_issues.append(
                {
                    "type": "tiny_instance",
                    "split": record.split,
                    "case_id": record.case_id,
                    "slice_name": record.slice_name,
                    "class_id": class_id,
                    "class_name": LABEL_NAMES.get(class_id, f"class_{class_id}"),
                    "areas": tiny,
                }
            )
        if len(areas) >= fragmentation_threshold:
            quality_issues.append(
                {
                    "type": "fragmentation",
                    "split": record.split,
                    "case_id": record.case_id,
                    "slice_name": record.slice_name,
                    "class_id": class_id,
                    "class_name": LABEL_NAMES.get(class_id, f"class_{class_id}"),
                    "component_count": len(areas),
                }
            )

    return {
        "total_pixels": total_pixels,
        "foreground_pixels": foreground_pixels,
        "background_pixels": total_pixels - foreground_pixels,
        "class_pixels": {int(k): int(v) for k, v in pixel_counter.items()},
    }


def build_overview(
    records: list[SliceRecord],
    case_split: dict[str, dict[str, str]],
    case_statistics: list[dict],
) -> dict:
    split_case_ids: dict[str, set[str]] = defaultdict(set)
    split_slice_counts: Counter[str] = Counter()
    shape_counter: Counter[tuple[int, ...]] = Counter()
    labeled_counter = Counter()

    for record in records:
        split_case_ids[record.split].add(record.case_id)
        split_slice_counts[record.split] += 1
        shape_counter[record.image_shape] += 1
        labeled_counter[record.split] += int(record.has_label)

    spacing_values = [item.get("spacing_mm") for item in case_statistics if item.get("spacing_mm")]
    raw_shapes = [tuple(item.get("shape", [])) for item in case_statistics if item.get("shape")]

    return {
        "total_cases": len({record.case_id for record in records}),
        "total_slices": len(records),
        "cases_per_split": {split: len(case_ids) for split, case_ids in split_case_ids.items()},
        "slices_per_split": dict(split_slice_counts),
        "labeled_slices_per_split": dict(labeled_counter),
        "image_shapes_2d": {str(shape): count for shape, count in shape_counter.items()},
        "modalities": 1,
        "channels": 1,
        "class_count_in_metadata": len(LABEL_NAMES),
        "raw_volume_shapes": {str(shape): raw_shapes.count(shape) for shape in set(raw_shapes)},
        "raw_spacing_mm_samples": spacing_values[:5],
        "target_spacing_mm": "1.0 (resampled in analyze_synapse.py)",
    }


def build_slices_per_case(records: list[SliceRecord], case_split: dict[str, dict[str, str]]) -> list[dict]:
    counter: dict[tuple[str, str], int] = defaultdict(int)
    for record in records:
        counter[(record.split, record.case_id)] += 1
    rows = []
    for (split, case_id), slice_count in sorted(counter.items()):
        meta = case_split.get(case_id, {})
        rows.append(
            {
                "split": split,
                "case_id": case_id,
                "slice_count_2d": slice_count,
                "raw_slices": meta.get("slices", ""),
                "raw_shape": meta.get("shape", ""),
                "spacing_mm": meta.get("spacing_mm", ""),
            }
        )
    return rows


def build_class_distribution(
    accumulators: dict[int, ClassAccumulator],
    total_pixels_all_slices: int,
) -> list[dict]:
    rows = []
    for class_id in sorted(accumulators):
        acc = accumulators[class_id]
        areas = np.asarray(acc.instance_areas, dtype=np.int64)
        rows.append(
            {
                "class_id": class_id,
                "name_en": LABEL_NAMES.get(class_id, f"class_{class_id}"),
                "name_cn": LABEL_NAMES_CN.get(class_id, LABEL_NAMES.get(class_id, f"class_{class_id}")),
                "case_count": len(acc.case_ids),
                "slice_hit_count": acc.slice_hits,
                "total_pixels": acc.total_pixels,
                "mean_instance_area": float(areas.mean()) if len(areas) else 0.0,
                "min_instance_area": int(areas.min()) if len(areas) else 0,
                "max_instance_area": int(areas.max()) if len(areas) else 0,
                "pixel_ratio": acc.total_pixels / total_pixels_all_slices if total_pixels_all_slices else 0.0,
            }
        )
    return rows


def build_ratio_table(
    class_pixel_totals: Counter[int],
    background_pixels: int,
    foreground_pixels: int,
) -> list[dict]:
    total = background_pixels + foreground_pixels
    rows = [
        {
            "name": "background",
            "name_en": "background",
            "name_cn": "背景",
            "pixels": background_pixels,
            "ratio": background_pixels / total if total else 0.0,
        },
        {
            "name": "foreground",
            "name_en": "foreground",
            "name_cn": "前景合计",
            "pixels": foreground_pixels,
            "ratio": foreground_pixels / total if total else 0.0,
        },
    ]
    for class_id, pixels in sorted(class_pixel_totals.items()):
        if class_id == 0:
            continue
        name_en = LABEL_NAMES.get(class_id, f"class_{class_id}")
        rows.append(
            {
                "name": name_en,
                "name_en": name_en,
                "name_cn": LABEL_NAMES_CN.get(class_id, name_en),
                "class_id": class_id,
                "pixels": int(pixels),
                "ratio": int(pixels) / total if total else 0.0,
            }
        )
    return rows


def build_case_presence_matrix(case_presence: dict[str, set[int]], case_split: dict[str, dict[str, str]]) -> tuple[list[str], list[dict]]:
    case_ids = sorted(case_presence.keys())
    class_ids = sorted({class_id for values in case_presence.values() for class_id in values})
    rows = []
    for class_id in class_ids:
        row = {
            "class_id": class_id,
            "name_en": LABEL_NAMES.get(class_id, f"class_{class_id}"),
            "name_cn": LABEL_NAMES_CN.get(class_id, LABEL_NAMES.get(class_id, f"class_{class_id}")),
        }
        for case_id in case_ids:
            row[case_id] = "✓" if class_id in case_presence[case_id] else "×"
        rows.append(row)
    return case_ids, rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_presence_markdown(path: Path, case_ids: list[str], rows: list[dict]) -> None:
    header = "| Organ | " + " | ".join(case_ids) + " |"
    sep = "| --- | " + " | ".join(["---"] * len(case_ids)) + " |"
    lines = [header, sep]
    for row in rows:
        lines.append(
            "| {name_cn} ({name_en}) | ".format(**row)
            + " | ".join(str(row[case_id]) for case_id in case_ids)
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_class_pixel_ratio(rows: list[dict], output_path: Path) -> None:
    setup_plot_style()
    names = [display_name_en(row["class_id"], row["name_en"]) for row in rows]
    ratios = [row["pixel_ratio"] * 100 for row in rows]
    plt.figure(figsize=(12, 5))
    plt.bar(names, ratios, color=[ORGAN_COLORS.get(row["class_id"], "#95A5A6") for row in rows])
    plt.ylabel("Pixel Ratio (%)")
    plt.title("Class Pixel Distribution")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_foreground_background(rows: list[dict], output_path: Path) -> None:
    setup_plot_style()
    labels = [
        display_name_en(row.get("class_id"), row.get("name_en", row["name"]))
        for row in rows
    ]
    ratios = [row["ratio"] * 100 for row in rows]
    colors = ["#BDC3C7", "#34495E"]
    for row in rows[2:]:
        colors.append(ORGAN_COLORS.get(row.get("class_id"), "#E67E22"))
    plt.figure(figsize=(12, 5))
    plt.bar(labels, ratios, color=colors)
    plt.ylabel("Ratio (%)")
    plt.title("Background / Foreground / Per-class Ratio")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_area_histograms(accumulators: dict[int, ClassAccumulator], output_dir: Path) -> None:
    setup_plot_style()
    for class_id, acc in sorted(accumulators.items()):
        if not acc.instance_areas:
            continue
        name = LABEL_NAMES.get(class_id, f"class_{class_id}")
        plt.figure(figsize=(7, 4))
        plt.hist(acc.instance_areas, bins=40, color=ORGAN_COLORS.get(class_id, "#95A5A6"), edgecolor="white")
        plt.xlabel("Instance Area (pixels)")
        plt.ylabel("Count")
        plt.title(f"{name} Instance Area Distribution")
        plt.tight_layout()
        plt.savefig(output_dir / f"area_hist_{class_id:02d}_{name}.png", dpi=200)
        plt.close()


def plot_slices_per_case(rows: list[dict], output_path: Path) -> None:
    setup_plot_style()
    colors = {"train": "#4C72B0", "val": "#55A868", "test": "#C44E52"}
    plt.figure(figsize=(12, 5))
    x_labels = [f"{row['split']}:{row['case_id']}" for row in rows]
    values = [row["slice_count_2d"] for row in rows]
    bar_colors = [colors.get(row["split"], "#7F7F7F") for row in rows]
    plt.bar(range(len(rows)), values, color=bar_colors)
    plt.xticks(range(len(rows)), x_labels, rotation=90, fontsize=7)
    plt.ylabel("2D Slice Count")
    plt.title("Slices per Case")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_quality_samples(
    records: list[SliceRecord],
    output_dir: Path,
    sample_count: int,
    seed: int,
) -> list[dict]:
    labeled = [record for record in records if record.has_label]
    if not labeled:
        return []
    rng = random.Random(seed)
    chosen = rng.sample(labeled, k=min(sample_count, len(labeled)))
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_plot_style()
    saved = []
    for record in chosen:
        image = np.load(record.image_path)
        label = np.load(record.label_path)
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)
        axes[0].set_title("Image")
        axes[0].axis("off")
        max_label = int(label.max()) if label.size else 0
        axes[1].imshow(label, cmap="tab20", vmin=0, vmax=max(max_label, 1))
        axes[1].set_title("Ground Truth")
        axes[1].axis("off")
        axes[2].imshow(make_overlay(image, label))
        axes[2].set_title("Overlay")
        axes[2].axis("off")
        fig.suptitle(f"{record.split} | {record.case_id} | {record.slice_name}", y=1.02)
        out_path = output_dir / f"{record.split}_{record.case_id}_{Path(record.slice_name).stem}.png"
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        saved.append(
            {
                "split": record.split,
                "case_id": record.case_id,
                "slice_name": record.slice_name,
                "figure": str(out_path),
            }
        )
    return saved


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir
    output_dir = args.output_dir or (data_dir / "dataset_report")
    figures_dir = output_dir / "figures"
    quality_dir = figures_dir / "quality_samples"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    records = collect_slice_records(data_dir, args.splits)
    if not records:
        raise FileNotFoundError(f"未在 {data_dir} 找到切片数据")

    case_split = load_case_split(data_dir)
    case_statistics = load_case_statistics(data_dir)
    accumulators: dict[int, ClassAccumulator] = defaultdict(ClassAccumulator)
    case_presence: dict[str, set[int]] = defaultdict(set)
    quality_issues: list[dict] = []
    class_pixel_totals: Counter[int] = Counter()
    background_pixels = 0
    foreground_pixels = 0
    total_pixels_all_slices = 0

    for record in records:
        if not record.has_label:
            continue
        stats = analyze_label_slice(
            record,
            accumulators,
            case_presence,
            quality_issues,
            args.tiny_area_threshold,
            args.fragmentation_threshold,
        )
        total_pixels_all_slices += stats["total_pixels"]
        background_pixels += stats["background_pixels"]
        foreground_pixels += stats["foreground_pixels"]
        class_pixel_totals.update(stats["class_pixels"])

    overview = build_overview(records, case_split, case_statistics)
    slices_per_case = build_slices_per_case(records, case_split)
    class_distribution = build_class_distribution(accumulators, total_pixels_all_slices)
    ratio_rows = build_ratio_table(class_pixel_totals, background_pixels, foreground_pixels)
    case_ids, presence_rows = build_case_presence_matrix(case_presence, case_split)
    quality_samples = save_quality_samples(records, quality_dir, args.quality_samples, args.seed)

    write_csv(output_dir / "overview.csv", [{"metric": k, "value": json.dumps(v, ensure_ascii=False)} for k, v in overview.items()])
    write_csv(output_dir / "slices_per_case.csv", slices_per_case)
    write_csv(output_dir / "class_distribution.csv", class_distribution)
    write_csv(output_dir / "foreground_background_ratio.csv", ratio_rows)
    write_csv(output_dir / "case_organ_presence.csv", presence_rows)
    write_presence_markdown(output_dir / "case_organ_presence.md", case_ids, presence_rows)

    if class_distribution:
        plot_class_pixel_ratio(class_distribution, figures_dir / "class_pixel_ratio.png")
        plot_area_histograms(accumulators, figures_dir)
    if ratio_rows:
        plot_foreground_background(ratio_rows, figures_dir / "foreground_background_ratio.png")
    if slices_per_case:
        plot_slices_per_case(slices_per_case, figures_dir / "slices_per_case.png")

    report = {
        "overview": overview,
        "class_distribution": class_distribution,
        "foreground_background_ratio": ratio_rows,
        "quality_issue_count": len(quality_issues),
        "quality_issues_sample": quality_issues[:50],
        "quality_samples": quality_samples,
    }
    with (output_dir / "dataset_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    with (output_dir / "label_quality_issues.json").open("w", encoding="utf-8") as f:
        json.dump(quality_issues, f, ensure_ascii=False, indent=2)

    print(f"数据目录: {data_dir}")
    print(f"报告目录: {output_dir}")
    print(f"病例数: {overview['total_cases']} | 切片数: {overview['total_slices']}")
    print(f"划分: {overview['cases_per_split']} / 切片: {overview['slices_per_split']}")
    print(f"类别分布表: {output_dir / 'class_distribution.csv'}")
    print(f"病例-器官矩阵: {output_dir / 'case_organ_presence.md'}")
    print(f"质量抽检图: {quality_dir}")
    print(f"质量问题条目: {len(quality_issues)}")


if __name__ == "__main__":
    main()
