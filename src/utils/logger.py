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


def update_evaluation_summary(summary_path: Path, evaluation: Mapping[str, Any]) -> None:
    summary_path = Path(summary_path)
    summary: dict[str, Any] = {}
    if summary_path.exists():
        with summary_path.open(encoding="utf-8") as f:
            summary = json.load(f)
    summary["evaluation"] = evaluation
    write_summary(summary_path, summary)


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
