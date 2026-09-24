"""训练过程记录工具：实验目录、CSV 指标、checkpoint、文件日志。"""

from __future__ import annotations

import csv
import json
import logging
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

DEFAULT_TRAIN_FIELDS = (
    "epoch",
    "train_loss",
    "train_dice",
    "val_loss",
    "val_dice",
    "seconds",
)


@dataclass(frozen=True)
class ExperimentRun:
    """单次实验运行目录。

    runs_root/name/
    ├── config.yaml
    ├── history.csv
    ├── summary.json
    ├── summary.md
    ├── checkpoints/
    │   ├── best.pth
    │   └── last.pth
    ├── predictions/
    └── train.log
    """

    name: str
    run_dir: Path
    config_path: Path
    history_path: Path
    summary_path: Path
    checkpoints_dir: Path
    predictions_dir: Path

    @property
    def best_checkpoint(self) -> Path:
        return self.checkpoints_dir / "best.pth"

    @property
    def last_checkpoint(self) -> Path:
        return self.checkpoints_dir / "last.pth"


def resolve_runs_root(exp_dir: Path, cfg: Mapping[str, Any]) -> Path:
    exp_cfg = cfg.get("experiments", {})
    runs_dir = exp_cfg.get("runs_dir", "runs")
    runs_path = Path(runs_dir)
    if runs_path.is_absolute():
        return runs_path
    return exp_dir / runs_path


def resolve_experiment_run(exp_dir: Path, cfg: Mapping[str, Any]) -> ExperimentRun:
    exp_cfg = cfg.get("experiments", {})
    name = exp_cfg.get("name")
    if not name:
        raise ValueError("config 缺少 [experiments].name")
    run_dir = resolve_runs_root(exp_dir, cfg) / name
    if not run_dir.exists():
        raise FileNotFoundError(f"未找到实验目录: {run_dir}")
    return ExperimentRun(
        name=name,
        run_dir=run_dir,
        config_path=run_dir / "config.yaml",
        history_path=run_dir / "history.csv",
        summary_path=run_dir / "summary.json",
        checkpoints_dir=run_dir / "checkpoints",
        predictions_dir=run_dir / "predictions",
    )


def create_experiment_run(
    runs_root: Path,
    name: str,
    config: Mapping[str, Any],
    config_source: Path | None = None,
) -> ExperimentRun:
    runs_root = Path(runs_root)
    run_dir = runs_root / name
    if run_dir.exists():
        raise FileExistsError(f"实验名称已存在，请更换 [experiments].name: {run_dir}")

    checkpoints_dir = run_dir / "checkpoints"
    predictions_dir = run_dir / "predictions"
    run_dir.mkdir(parents=True)
    checkpoints_dir.mkdir()
    predictions_dir.mkdir()

    config_path = run_dir / "config.yaml"
    _save_config_yaml(config_path, config, config_source)

    exp_cfg = config.get("experiments", {})
    summary = {
        "name": name,
        "date": exp_cfg.get("date"),
        "description": exp_cfg.get("description"),
        "status": "running",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "best_val_dice": None,
        "best_epoch": None,
        "completed_epochs": 0,
        "total_epochs": config.get("hyperparameters", {}).get("num_epochs"),
    }
    write_summary(run_dir / "summary.json", summary)

    return ExperimentRun(
        name=name,
        run_dir=run_dir,
        config_path=config_path,
        history_path=run_dir / "history.csv",
        summary_path=run_dir / "summary.json",
        checkpoints_dir=checkpoints_dir,
        predictions_dir=predictions_dir,
    )


def write_summary(summary_path: Path, summary: Mapping[str, Any]) -> None:
    summary_path = Path(summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def update_training_summary(
    summary_path: Path,
    *,
    status: str,
    best_val_dice: float,
    best_epoch: int,
    completed_epochs: int,
) -> None:
    summary_path = Path(summary_path)
    summary: dict[str, Any] = {}
    if summary_path.exists():
        with summary_path.open(encoding="utf-8") as f:
            summary = json.load(f)
    summary.update(
        {
            "status": status,
            "best_val_dice": best_val_dice,
            "best_epoch": best_epoch,
            "completed_epochs": completed_epochs,
            "finished_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    write_summary(summary_path, summary)
    write_summary_md(summary_path.parent)


def update_evaluation_summary(summary_path: Path, evaluation: Mapping[str, Any]) -> None:
    summary_path = Path(summary_path)
    summary: dict[str, Any] = {}
    if summary_path.exists():
        with summary_path.open(encoding="utf-8") as f:
            summary = json.load(f)
    summary["evaluation"] = evaluation
    write_summary(summary_path, summary)
    write_summary_md(summary_path.parent)


def _fmt_summary_num(value: Any, decimals: int = 4) -> str:
    if value is None or value == "":
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{number:.{decimals}f}"


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _format_summary_date_cn(date_str: str | None) -> str:
    if not date_str:
        return ""
    try:
        parsed = datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        return date_str
    return f"{parsed.year}年{parsed.month}月{parsed.day}日"


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    if not rows:
        return "_（无数据）_\n"
    header_line = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join("---" for _ in headers) + " |"
    body = "\n".join("| " + " | ".join(row) + " |" for row in rows)
    return f"{header_line}\n{separator}\n{body}\n"


ORGAN_METRICS_HEADERS = (
    "organ",
    "organ_cn",
    "num_slices",
    "dice_mean",
    "dice_percent (%)",
    "iou_mean",
    "iou_percent (%)",
    "precision_mean",
    "precision_percent (%)",
    "recall_mean",
    "recall_percent (%)",
    "fp_ratio_mean",
    "fp_ratio_percent (%)",
    "fn_ratio_mean",
    "fn_ratio_percent (%)",
    "hd95_mean",
    "gt_area_mean",
    "pred_area_mean",
    "pred_gt_ratio_mean",
    "gt_percent_mean",
    "pred_percent_mean",
)

ORGAN_METRICS_SIMPLE_HEADERS = ("organ_cn", "num_slices", "Dice (%)", "IoU (%)", "HD95")


def _organ_metrics_rows(rows: list[dict[str, str]], *, simple: bool = False) -> list[list[str]]:
    table_rows: list[list[str]] = []
    for row in rows:
        if simple:
            table_rows.append([
                row.get("organ_cn", ""),
                row.get("num_slices", ""),
                _fmt_summary_num(row.get("dice_percent")),
                _fmt_summary_num(row.get("iou_percent")),
                _fmt_summary_num(row.get("hd95_mean")),
            ])
            continue
        table_rows.append([
            row.get("organ", ""),
            row.get("organ_cn", ""),
            row.get("num_slices", ""),
            _fmt_summary_num(row.get("dice_mean")),
            _fmt_summary_num(row.get("dice_percent")),
            _fmt_summary_num(row.get("iou_mean")),
            _fmt_summary_num(row.get("iou_percent")),
            _fmt_summary_num(row.get("precision_mean")),
            _fmt_summary_num(row.get("precision_percent")),
            _fmt_summary_num(row.get("recall_mean")),
            _fmt_summary_num(row.get("recall_percent")),
            _fmt_summary_num(row.get("fp_ratio_mean")),
            _fmt_summary_num(row.get("fp_ratio_percent")),
            _fmt_summary_num(row.get("fn_ratio_mean")),
            _fmt_summary_num(row.get("fn_ratio_percent")),
            _fmt_summary_num(row.get("hd95_mean")),
            _fmt_summary_num(row.get("gt_area_mean")),
            _fmt_summary_num(row.get("pred_area_mean")),
            _fmt_summary_num(row.get("pred_gt_ratio_mean")),
            _fmt_summary_num(row.get("gt_percent_mean")),
            _fmt_summary_num(row.get("pred_percent_mean")),
        ])
    return table_rows


def _paper_metrics_from_evaluation(summary: Mapping[str, Any]) -> tuple[list[list[str]], list[str]]:
    splits = summary.get("evaluation", {}).get("splits", [])
    rows_by_split = {
        item["split"]: item for item in splits if item.get("split") in {"train", "val"}
    }
    if not rows_by_split:
        return [], []

    organ_order: list[tuple[str, str]] = []
    for split in ("train", "val"):
        per_organ = rows_by_split.get(split, {}).get("per_organ", {})
        for organ_name, metrics in per_organ.items():
            cn = metrics.get("name_cn", organ_name)
            token = (cn, organ_name)
            if token not in organ_order:
                organ_order.append(token)

    fieldnames = [
        "Dice (%)",
        "IoU (%)",
        "Precision (%)",
        "Recall (%)",
        "HD95",
        "FP ratio (%)",
        "FN ratio (%)",
    ]
    for cn, _ in organ_order:
        fieldnames.extend([f"{cn} Dice (%)", f"{cn} IoU (%)"])

    table_rows: list[list[str]] = []
    for split in ("train", "val"):
        item = rows_by_split.get(split)
        if item is None or item.get("mean_dice_percent") is None:
            continue
        row = [
            split,
            _fmt_summary_num(item.get("mean_dice_percent")),
            _fmt_summary_num(item.get("mean_iou_percent")),
            _fmt_summary_num(item.get("mean_precision_percent")),
            _fmt_summary_num(item.get("mean_recall_percent")),
            _fmt_summary_num(item.get("mean_hd95")),
            _fmt_summary_num(item.get("mean_fp_ratio_percent")),
            _fmt_summary_num(item.get("mean_fn_ratio_percent")),
        ]
        per_organ = item.get("per_organ", {})
        for _, organ_name in organ_order:
            metrics = per_organ.get(organ_name, {})
            row.append(_fmt_summary_num(metrics.get("dice_percent")))
            row.append(_fmt_summary_num(metrics.get("iou_percent")))
        table_rows.append(row)
    return table_rows, fieldnames


def _paper_metrics_rows(paper_metrics_path: Path) -> tuple[list[list[str]], list[str]]:
    rows = _read_csv_rows(paper_metrics_path)
    if not rows:
        return [], []
    fieldnames = [key for key in rows[0] if key != "split"]
    table_rows: list[list[str]] = []
    for row in rows:
        if row.get("split") not in {"train", "val"}:
            continue
        table_rows.append([
            row.get("split", ""),
            *[_fmt_summary_num(row.get(field)) for field in fieldnames],
        ])
    return table_rows, fieldnames


def _build_training_log_lines(
    history_path: Path,
    *,
    total_epochs: int | None,
    tail_epochs: int = 6,
) -> list[str]:
    rows = _read_csv_rows(history_path)
    if not rows:
        return []
    lines: list[str] = []
    for row in rows[-tail_epochs:]:
        epoch = row.get("epoch", "")
        total = total_epochs if total_epochs is not None else "?"
        lines.append(
            f"Epoch [{epoch}/{total}] "
            f"train_loss={_fmt_summary_num(row.get('train_loss'))} "
            f"train_dice={_fmt_summary_num(row.get('train_dice'))} "
            f"val_loss={_fmt_summary_num(row.get('val_loss'))} "
            f"val_dice={_fmt_summary_num(row.get('val_dice'))} "
            f"time={_fmt_summary_num(row.get('seconds'), 1)}s"
        )
    return lines


def _build_class_dice_lines(class_metrics_path: Path, epoch: int) -> list[str]:
    rows = _read_csv_rows(class_metrics_path)
    if not rows:
        return []
    lines: list[str] = []
    epoch_text = str(epoch)
    for split in ("val", "train"):
        parts = [
            f"{row['class_name']}={_fmt_summary_num(row.get('dice'))}"
            for row in rows
            if row.get("epoch") == epoch_text and row.get("split") == split
        ]
        if parts:
            lines.append(f"Epoch [{epoch}] class dice [{split}]: " + ", ".join(parts))
    return lines


def _latest_class_metrics_epoch(class_metrics_path: Path, completed_epochs: int | None) -> int | None:
    rows = _read_csv_rows(class_metrics_path)
    if not rows:
        return None
    epochs = sorted({int(row["epoch"]) for row in rows if row.get("epoch")})
    if completed_epochs is not None:
        eligible = [epoch for epoch in epochs if epoch <= completed_epochs]
        if eligible:
            return eligible[-1]
    return epochs[-1] if epochs else None


def build_summary_markdown(run_dir: Path) -> str:
    run_dir = Path(run_dir)
    summary = {}
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        with summary_path.open(encoding="utf-8") as f:
            summary = json.load(f)

    name = summary.get("name", run_dir.name)
    date_cn = _format_summary_date_cn(summary.get("date"))
    description = summary.get("description", "")
    status = summary.get("status", "")
    best_val_dice = summary.get("best_val_dice")
    best_epoch = summary.get("best_epoch")
    completed_epochs = summary.get("completed_epochs")
    total_epochs = summary.get("total_epochs")

    lines: list[str] = [
        f"# {name}",
        "",
        f"> date: {date_cn}",
    ]
    if description:
        lines.append(f"> description: {description}")
    if status:
        lines.append(f"> status: {status}")
    if best_val_dice is not None:
        best_epoch_text = best_epoch if best_epoch is not None else "?"
        lines.append(
            f"> best_val_dice: {_fmt_summary_num(best_val_dice)} (epoch {best_epoch_text})"
        )
    if completed_epochs is not None:
        total_epochs_text = total_epochs if total_epochs is not None else "?"
        lines.append(f"> completed_epochs: {completed_epochs} / {total_epochs_text}")
    lines.extend(["", "### 1 训练收尾日志", "", "最近若干 epoch 的训练/验证指标，来源 `history.csv`。", ""])

    training_lines = _build_training_log_lines(
        run_dir / "history.csv",
        total_epochs=total_epochs,
        tail_epochs=7,
    )
    class_epoch = _latest_class_metrics_epoch(run_dir / "class_metrics.csv", completed_epochs)
    if class_epoch is not None and training_lines:
        class_lines = _build_class_dice_lines(run_dir / "class_metrics.csv", class_epoch)
        enriched_lines: list[str] = []
        for line in training_lines:
            enriched_lines.append(line)
            epoch_token = line.split("]", 1)[0].removeprefix("Epoch [")
            epoch_value = epoch_token.split("/", 1)[0]
            if epoch_value.isdigit() and int(epoch_value) == class_epoch:
                enriched_lines.extend(class_lines)
        training_lines = enriched_lines

    if training_lines:
        lines.extend(["```bash", *training_lines, "```", ""])
    else:
        lines.append("_（无 history.csv 数据）_\n")

    paper_metrics_path = run_dir / "paper_metrics.csv"
    paper_rows, paper_fields = _paper_metrics_from_evaluation(summary)
    if not paper_rows and paper_metrics_path.exists():
        paper_rows, paper_fields = _paper_metrics_rows(paper_metrics_path)

    section_no = 2
    if paper_rows:
        lines.extend([
            f"### {section_no} 整体指标对比（train / val）",
            "",
            "基于 best checkpoint 推理结果汇总；宏平均 Dice/IoU 等为 8 个目标器官均值（不含背景）。",
            "百分比列单位为 %，数值保留四位小数。",
            "",
            _markdown_table(["split", *paper_fields], paper_rows),
        ])
        section_no += 1

    for split in ("train", "val"):
        organ_path = run_dir / "predictions" / split / "organ_metrics.csv"
        organ_rows = _read_csv_rows(organ_path)
        if not organ_rows:
            continue
        split_cn = "训练集" if split == "train" else "验证集"
        lines.extend([
            f"### {section_no} {split_cn}各器官详细指标",
            "",
            f"来源 `predictions/{split}/organ_metrics.csv`；"
            "按器官聚合的切片级均值，`*_percent` 列单位为 %。",
            "",
            _markdown_table(list(ORGAN_METRICS_HEADERS), _organ_metrics_rows(organ_rows)),
        ])
        section_no += 1

    simple_sections: list[str] = []
    for split in ("train", "val"):
        organ_path = run_dir / "predictions" / split / "organ_metrics.csv"
        organ_rows = _read_csv_rows(organ_path)
        if not organ_rows:
            continue
        split_cn = "训练集" if split == "train" else "验证集"
        simple_sections.append(
            f"**{split_cn}精简版**\n\n"
            + _markdown_table(
                list(ORGAN_METRICS_SIMPLE_HEADERS),
                _organ_metrics_rows(organ_rows, simple=True),
            )
        )

    if simple_sections:
        lines.extend([
            f"### {section_no} 各器官精简指标",
            "",
            "仅保留 Dice、IoU、HD95 与参与统计的切片数，便于快速对比。",
            "",
            *simple_sections,
        ])

    return "\n".join(lines).rstrip() + "\n"


def write_summary_md(run_dir: Path) -> Path:
    run_dir = Path(run_dir)
    summary_md_path = run_dir / "summary.md"
    summary_md_path.write_text(build_summary_markdown(run_dir), encoding="utf-8")
    return summary_md_path


def _save_config_yaml(
    config_path: Path,
    config: Mapping[str, Any],
    config_source: Path | None,
) -> None:
    if config_source is not None and config_source.suffix.lower() in {".toml", ".yaml", ".yml"}:
        if config_source.suffix.lower() == ".toml":
            _write_yaml_dict(config_path, config)
            return
        shutil.copy2(config_source, config_path)
        return
    _write_yaml_dict(config_path, config)


def _write_yaml_dict(path: Path, data: Mapping[str, Any]) -> None:
    try:
        import yaml
    except ImportError:
        with path.open("w", encoding="utf-8") as f:
            f.write(_dump_yaml(data))
        return
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(dict(data), f, allow_unicode=True, sort_keys=False)


def _dump_yaml(data: Any, indent: int = 0) -> str:
    prefix = " " * indent
    if isinstance(data, dict):
        lines: list[str] = []
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                lines.append(f"{prefix}{key}:")
                lines.append(_dump_yaml(value, indent + 2).rstrip())
            else:
                lines.append(f"{prefix}{key}: {_format_yaml_scalar(value)}")
        return "\n".join(lines) + "\n"
    if isinstance(data, list):
        lines = []
        for item in data:
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}-")
                lines.append(_dump_yaml(item, indent + 2).rstrip())
            else:
                lines.append(f"{prefix}- {_format_yaml_scalar(item)}")
        return "\n".join(lines) + "\n"
    return f"{prefix}{_format_yaml_scalar(data)}\n"


def _format_yaml_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        if any(ch in value for ch in ":{}[]&*#?|-<>=!%@`") or value.strip() != value:
            return json.dumps(value, ensure_ascii=False)
        return value
    return json.dumps(value, ensure_ascii=False)


def setup_file_logger(
    output_dir: Path,
    name: str = "train",
    filename: str = "train.log",
) -> logging.Logger:
    """同时输出到控制台与 output_dir/filename，文件以追加模式写入。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    file_handler = logging.FileHandler(output_dir / filename, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(stream_handler)

    return logger


class CsvMetricsLogger:
    """追加写入 CSV 指标，每个 epoch 后立即 flush，防止中断丢数据。"""

    def __init__(self, path: Path, fieldnames: Sequence[str] | None = None) -> None:
        self.path = Path(path)
        self.fieldnames = tuple(fieldnames or DEFAULT_TRAIN_FIELDS)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = None
        self._writer: csv.DictWriter | None = None
        self._open()

    def _open(self) -> None:
        write_header = not self.path.exists() or self.path.stat().st_size == 0
        self._file = self.path.open("a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=self.fieldnames, extrasaction="ignore")
        if write_header:
            self._writer.writeheader()
            self._file.flush()

    def log(self, metrics: Mapping[str, Any]) -> None:
        if self._writer is None or self._file is None:
            raise RuntimeError("CsvMetricsLogger 已关闭")
        row = {field: metrics.get(field) for field in self.fieldnames}
        self._writer.writerow(row)
        self._file.flush()

    def last_epoch(self) -> int | None:
        if not self.path.exists() or self.path.stat().st_size == 0:
            return None
        with self.path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            epochs = [int(row["epoch"]) for row in reader if row.get("epoch")]
        return max(epochs) if epochs else None

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
            self._writer = None

    def __enter__(self) -> CsvMetricsLogger:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


DEFAULT_CLASS_METRIC_FIELDS = (
    "epoch",
    "split",
    "class_id",
    "class_name",
    "dice",
)


class ClassMetricsLogger:
    """追加写入逐类别指标 CSV。"""

    def __init__(self, path: Path, fieldnames: Sequence[str] | None = None) -> None:
        self.path = Path(path)
        self.fieldnames = tuple(fieldnames or DEFAULT_CLASS_METRIC_FIELDS)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = None
        self._writer: csv.DictWriter | None = None
        self._open()

    def _open(self) -> None:
        write_header = not self.path.exists() or self.path.stat().st_size == 0
        self._file = self.path.open("a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=self.fieldnames, extrasaction="ignore")
        if write_header:
            self._writer.writeheader()
            self._file.flush()

    def log_rows(self, rows: Sequence[Mapping[str, Any]]) -> None:
        if self._writer is None or self._file is None:
            raise RuntimeError("ClassMetricsLogger 已关闭")
        for row in rows:
            self._writer.writerow({field: row.get(field) for field in self.fieldnames})
        self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
            self._writer = None

    def __enter__(self) -> ClassMetricsLogger:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


class CheckpointManager:
    """管理 checkpoints/best.pth 与 checkpoints/last.pth。"""

    def __init__(
        self,
        checkpoint_dir: Path,
        metric_key: str = "best_val_dice",
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_path = self.checkpoint_dir / "best.pth"
        self.last_path = self.checkpoint_dir / "last.pth"
        self.metric_key = metric_key
        self.best_metric = -1.0
        self.best_epoch = 0

    def save_last(self, epoch: int, state: dict[str, Any]) -> None:
        torch.save({**state, "epoch": epoch}, self.last_path)

    def maybe_save_best(self, metric: float, epoch: int, state: dict[str, Any]) -> bool:
        if metric <= self.best_metric:
            return False
        self.best_metric = metric
        self.best_epoch = epoch
        payload = {**state, self.metric_key: metric, "epoch": epoch}
        torch.save(payload, self.best_path)
        return True
