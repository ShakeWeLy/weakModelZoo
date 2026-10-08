"""由 run 目录下的 CSV / JSON 生成 summary.md 与 paper_metrics 表。"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

from .recorder import read_csv, read_json

if TYPE_CHECKING:
    from .recorder import RunRecorder

SPLITS = ("train", "val", "test")
SPLIT_CN = {"train": "训练集", "val": "验证集", "test": "测试集"}

ORGAN_METRICS_FIELDS = (
    "organ", "organ_cn", "num_slices",
    "dice_mean", "dice_percent", "iou_mean", "iou_percent",
    "precision_mean", "precision_percent", "recall_mean", "recall_percent",
    "fp_ratio_mean", "fp_ratio_percent", "fn_ratio_mean", "fn_ratio_percent",
    "hd95_mean", "gt_area_mean", "pred_area_mean",
    "pred_gt_ratio_mean", "gt_percent_mean", "pred_percent_mean",
)
SIMPLE_ORGAN_FIELDS = (
    ("organ_cn", "organ_cn"), ("num_slices", "num_slices"),
    ("dice_percent", "Dice (%)"), ("iou_percent", "IoU (%)"), ("hd95_mean", "HD95 (px)"),
)


def fmt_num(value: Any, decimals: int = 4) -> str:
    if value is None or value == "":
        return ""
    try:
        return f"{float(value):.{decimals}f}"
    except (TypeError, ValueError):
        return str(value)


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    if not rows:
        return "_（无数据）_\n"
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *("| " + " | ".join(row) + " |" for row in rows),
    ]
    return "\n".join(lines) + "\n"


def build_paper_metrics_rows(predictions_dir: Path) -> list[dict[str, str]]:
    """每个 split 一行：宏平均指标 + 各器官 Dice/IoU（百分比，两位小数）。"""
    rows: list[dict[str, str]] = []
    for split in SPLITS:
        summary = read_json(Path(predictions_dir) / split / "summary.json")
        if summary.get("mean_dice") is None:
            continue
        row = {
            "split": split,
            "Dice (%)": fmt_num(summary.get("mean_dice_percent"), 2),
            "IoU (%)": fmt_num(summary.get("mean_iou_percent"), 2),
            "Precision (%)": fmt_num(summary.get("mean_precision_percent"), 2),
            "Recall (%)": fmt_num(summary.get("mean_recall_percent"), 2),
            "HD95 (px)": fmt_num(summary.get("mean_hd95"), 2),
            "FP ratio (%)": fmt_num(summary.get("mean_fp_ratio_percent"), 2),
            "FN ratio (%)": fmt_num(summary.get("mean_fn_ratio_percent"), 2),
        }
        for organ, metrics in summary.get("per_organ", {}).items():
            cn = metrics.get("name_cn") or organ
            row[f"{cn} Dice (%)"] = fmt_num(metrics.get("dice_percent"), 2)
            row[f"{cn} IoU (%)"] = fmt_num(metrics.get("iou_percent"), 2)
        rows.append(row)
    return rows


def _date_cn(value: str | None) -> str:
    if not value:
        return ""
    try:
        parsed = datetime.strptime(value, "%Y-%m-%d")
    except ValueError:
        return value
    return f"{parsed.year}年{parsed.month}月{parsed.day}日"


def _last_epoch_rows(summary: dict[str, Any]) -> list[list[str]]:
    """summary.json 中记录的最后一个 epoch 的训练 / 验证指标（一行表格）。"""
    last = summary.get("last_epoch") or {}
    if not last:
        return []
    return [[
        str(last.get("epoch", "")),
        fmt_num(last.get("train_loss")), fmt_num(last.get("val_loss")),
        fmt_num(last.get("train_dice")), fmt_num(last.get("val_dice")),
        fmt_num(last.get("lr"), 6), fmt_num(last.get("seconds"), 1),
    ]]


def build_summary_markdown(recorder: RunRecorder) -> str:
    summary = read_json(recorder.summary_path)
    lines = [f"# {summary.get('name', recorder.name)}", ""]
    meta = [
        ("model", summary.get("model")),
        ("date", _date_cn(summary.get("date"))),
        ("description", summary.get("description")),
        ("status", summary.get("status")),
    ]
    if summary.get("best_val_dice") is not None:
        meta.append((
            "best_val_dice",
            f"{fmt_num(summary['best_val_dice'])} (epoch {summary.get('best_epoch', '?')})",
        ))
    if summary.get("completed_epochs") is not None:
        meta.append((
            "completed_epochs",
            f"{summary['completed_epochs']} / {summary.get('total_epochs') or '?'}",
        ))
    lines.extend(f"> {key}: {value}" for key, value in meta if value)

    section = 1
    lines.extend([
        "", f"### {section} 训练收尾", "",
        "训练曲线（loss / dice / lr / 逐类 Dice）与超参记录在 `tensorboard/`，"
        f"查看命令：`tensorboard --logdir {recorder.tensorboard_dir}`。",
        "",
        "最后一个 epoch 的指标（来源 `summary.json` 的 `last_epoch`）：",
        "",
        markdown_table(["epoch", "train_loss", "val_loss", "train_dice", "val_dice", "lr", "seconds"],
                       _last_epoch_rows(summary)),
    ])

    paper_rows = build_paper_metrics_rows(recorder.predictions_dir)
    if paper_rows:
        section += 1
        fields = [key for key in paper_rows[0] if key != "split"]
        lines.extend([
            f"### {section} 整体指标对比",
            "",
            "基于 analyze 推理结果；宏平均为各目标器官切片级均值的平均（不含背景），百分比列单位为 %。",
            "",
            markdown_table(["split", *fields], [[row["split"], *(row.get(f, "") for f in fields)] for row in paper_rows]),
        ])

    organ_tables: dict[str, list[dict[str, str]]] = {
        split: rows
        for split in SPLITS
        if (rows := read_csv(recorder.predictions_dir / split / "organ_metrics.csv"))
    }
    for split, rows in organ_tables.items():
        section += 1
        lines.extend([
            f"### {section} {SPLIT_CN[split]}各器官详细指标",
            "",
            f"来源 `predictions/{split}/organ_metrics.csv`；按器官聚合的切片级均值，`*_percent` 列单位为 %。",
            "",
            markdown_table(
                ORGAN_METRICS_FIELDS,
                [[row.get("organ", ""), row.get("organ_cn", ""), row.get("num_slices", ""),
                  *(fmt_num(row.get(f)) for f in ORGAN_METRICS_FIELDS[3:])] for row in rows],
            ),
        ])

    if organ_tables:
        section += 1
        lines.extend([f"### {section} 各器官精简指标", "", "仅保留 Dice、IoU、HD95 与参与统计的切片数。", ""])
        for split, rows in organ_tables.items():
            table_rows = [
                [row.get(key, "") if key in {"organ_cn", "num_slices"} else fmt_num(row.get(key))
                 for key, _ in SIMPLE_ORGAN_FIELDS]
                for row in rows
            ]
            lines.extend([
                f"**{SPLIT_CN[split]}精简版**",
                "",
                markdown_table([title for _, title in SIMPLE_ORGAN_FIELDS], table_rows),
            ])

    return "\n".join(lines).rstrip() + "\n"
