"""单次实验的结果记录：目录布局、文件日志、TensorBoard、CSV（预测结果）、checkpoint、summary.json。

目录布局::

    <project>/<Model>/runs/<name>/
    ├── config.yaml          合并后的完整配置
    ├── train.log            训练过程的文本日志（每 epoch 一行摘要）
    ├── analyze.log          analyze 日志
    ├── tensorboard/         训练指标（loss / dice / lr / 逐类 Dice / 超参），tensorboard --logdir 查看
    ├── summary.json / summary.md
    ├── paper_metrics.csv    analyze 后生成
    ├── checkpoints/{best,last}.pth
    └── predictions/<split>/...   analyze 的预测掩码、可视化与逐切片 / 逐器官 CSV
"""

from __future__ import annotations

import csv
import json
import logging
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import yaml
from torch.utils.tensorboard import SummaryWriter


def unique_run_name(runs_root: Path, base_name: str, *, max_tries: int = 999) -> str:
    """``base_name`` 已存在时：以 ``_V<n>`` 结尾则递增版本号，否则追加 ``_2``、``_3``…"""
    if not (runs_root / base_name).exists():
        return base_name
    match = re.match(r"^(.*_V)(\d+)$", base_name)
    if match:
        prefix, start = match.group(1), int(match.group(2))
        candidates = (f"{prefix}{v}" for v in range(start + 1, start + max_tries))
    else:
        candidates = (f"{base_name}_{v}" for v in range(2, max_tries + 1))
    for candidate in candidates:
        if not (runs_root / candidate).exists():
            return candidate
    raise RuntimeError(f"无法在 {runs_root} 下为 {base_name!r} 分配 run 名称")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str] | None = None,
              *, encoding: str = "utf-8") -> None:
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            fieldnames.extend(key for key in row if key not in fieldnames)
    with Path(path).open("w", newline="", encoding=encoding) as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float):
        return round(value, 6)
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Mapping[str, Any]) -> None:
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class RunRecorder:
    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        self.name = self.run_dir.name
        self.checkpoints_dir = self.run_dir / "checkpoints"
        self.predictions_dir = self.run_dir / "predictions"
        self.tensorboard_dir = self.run_dir / "tensorboard"
        self.config_path = self.run_dir / "config.yaml"
        self.summary_path = self.run_dir / "summary.json"
        self.paper_metrics_path = self.run_dir / "paper_metrics.csv"
        self._logger: logging.Logger | None = None

    @classmethod
    def create(
        cls,
        runs_root: Path,
        name: str,
        cfg: Mapping[str, Any],
        *,
        overwrite: bool = False,
    ) -> RunRecorder:
        run_dir = Path(runs_root) / name
        if run_dir.exists():
            if not overwrite:
                raise FileExistsError(f"run 已存在: {run_dir}")
            shutil.rmtree(run_dir)
        recorder = cls(run_dir)
        recorder.checkpoints_dir.mkdir(parents=True)
        recorder.predictions_dir.mkdir()
        with recorder.config_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(dict(cfg), f, allow_unicode=True, sort_keys=False)
        exp = cfg.get("experiments", {})
        write_json(recorder.summary_path, {
            "name": name,
            "model": cfg.get("model", {}).get("display_name"),
            "date": exp.get("date"),
            "description": exp.get("description"),
            "status": "running",
            "created_at": _utc_now(),
            "best_val_dice": None,
            "best_epoch": None,
            "completed_epochs": 0,
            "total_epochs": cfg.get("hyperparameters", {}).get("num_epochs"),
        })
        return recorder

    @classmethod
    def open(cls, run_dir: Path) -> RunRecorder:
        run_dir = Path(run_dir)
        if not run_dir.is_dir():
            raise FileNotFoundError(f"未找到实验目录: {run_dir}")
        return cls(run_dir)

    def load_config(self) -> dict[str, Any]:
        with self.config_path.open(encoding="utf-8") as f:
            return yaml.safe_load(f)

    def logger(self, name: str = "project1_2610", filename: str = "train.log") -> logging.Logger:
        """控制台 + run_dir/filename（追加）双输出。"""
        if self._logger is not None:
            return self._logger
        logger = logging.getLogger(f"{name}.{self.name}.{filename}")
        logger.setLevel(logging.INFO)
        logger.handlers.clear()
        logger.propagate = False
        file_handler = logging.FileHandler(self.run_dir / filename, mode="a", encoding="utf-8")
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        )
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
        self._logger = logger
        return logger

    def close_logger(self) -> None:
        if self._logger is None:
            return
        for handler in list(self._logger.handlers):
            handler.close()
            self._logger.removeHandler(handler)
        self._logger = None

    def tensorboard_writer(self) -> SummaryWriter:
        """训练指标写入 ``runs/<name>/tensorboard/``；使用后需调用 ``close()``。"""
        return SummaryWriter(log_dir=str(self.tensorboard_dir))

    def checkpoint_path(self, kind: str) -> Path:
        return self.checkpoints_dir / f"{kind}.pth"

    def save_checkpoint(self, kind: str, state: Mapping[str, Any]) -> Path:
        path = self.checkpoint_path(kind)
        torch.save(dict(state), path)
        return path

    def update_summary(self, **fields: Any) -> dict[str, Any]:
        summary = read_json(self.summary_path)
        summary.update(fields)
        write_json(self.summary_path, summary)
        return summary

    def finish_training(self, *, status: str, best_val_dice: float | None, best_epoch: int | None,
                        completed_epochs: int, last_metrics: Mapping[str, Any] | None = None) -> None:
        self.update_summary(
            status=status,
            best_val_dice=best_val_dice,
            best_epoch=best_epoch,
            completed_epochs=completed_epochs,
            last_epoch=dict(last_metrics) if last_metrics else None,
            finished_at=_utc_now(),
        )
        self.write_summary_md()

    def write_summary_md(self) -> Path:
        from .report import build_summary_markdown

        path = self.run_dir / "summary.md"
        path.write_text(build_summary_markdown(self), encoding="utf-8")
        return path
