"""训练过程记录工具：CSV 指标追加写入、checkpoint 管理、文件日志。"""

from __future__ import annotations

import csv
import logging
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
        """读取已记录的最大 epoch，用于断点续训提示。"""
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
    """管理 best 与周期性 checkpoint，启动时从已有 best 权重恢复最优指标。"""

    def __init__(
        self,
        output_dir: Path,
        best_name: str,
        periodic_prefix: str,
        save_every: int = 50,
        metric_key: str = "best_val_dice",
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.best_path = self.output_dir / best_name
        self.periodic_prefix = periodic_prefix
        self.save_every = save_every
        self.metric_key = metric_key
        self.best_metric = self._load_best_metric()

    def _load_best_metric(self) -> float:
        if not self.best_path.exists():
            return -1.0
        try:
            checkpoint = torch.load(self.best_path, map_location="cpu", weights_only=False)
            return float(checkpoint.get(self.metric_key, checkpoint.get("val_dice", -1.0)))
        except Exception:
            return -1.0

    def maybe_save_best(self, metric: float, epoch: int, state: dict[str, Any]) -> bool:
        if metric <= self.best_metric:
            return False
        self.best_metric = metric
        payload = {**state, self.metric_key: metric, "epoch": epoch}
        torch.save(payload, self.best_path)
        return True

    def maybe_save_periodic(self, epoch: int, state: dict[str, Any]) -> None:
        if self.save_every <= 0 or epoch % self.save_every != 0:
            return
        path = self.output_dir / f"{self.periodic_prefix}_epoch_{epoch:03d}.pth"
        torch.save({**state, "epoch": epoch}, path)
