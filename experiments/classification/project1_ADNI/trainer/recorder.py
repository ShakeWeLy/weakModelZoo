"""单次实验的结果记录：目录布局、文件日志、TensorBoard、逐折 CSV、checkpoint、summary。

目录布局::

    <project>/<Model>/runs/<name>/
    ├── config.yaml            合并后的完整配置
    ├── train.log              训练文本日志（每 epoch 一行摘要）
    ├── summary.json / summary.md
    ├── cv_metrics.csv         各折最终指标 + mean / std
    ├── folds/fold<k>_history.csv   每折每 epoch 的 loss 与验证指标
    ├── checkpoints/fold<k>_last.pth
    ├── tensorboard/fold<k>/   训练曲线，tensorboard --logdir 查看
    └── data/mock_pearson.npz  使用 mock 数据时保存的数据
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

from .metrics import METRIC_KEYS


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


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (round(v, 6) if isinstance(v, float) else v) for k, v in row.items()})


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class RunRecorder:
    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        self.name = self.run_dir.name
        self.folds_dir = self.run_dir / "folds"
        self.checkpoints_dir = self.run_dir / "checkpoints"
        self.tensorboard_dir = self.run_dir / "tensorboard"
        self.config_path = self.run_dir / "config.yaml"
        self.summary_path = self.run_dir / "summary.json"
        self._logger: logging.Logger | None = None

    @classmethod
    def create(cls, runs_root: Path, name: str, cfg: Mapping[str, Any], *, overwrite: bool = False) -> RunRecorder:
        run_dir = Path(runs_root) / name
        if run_dir.exists():
            if not overwrite:
                raise FileExistsError(f"run 已存在: {run_dir}")
            shutil.rmtree(run_dir)
        recorder = cls(run_dir)
        for d in (recorder.folds_dir, recorder.checkpoints_dir, recorder.tensorboard_dir):
            d.mkdir(parents=True)
        with recorder.config_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(dict(cfg), f, allow_unicode=True, sort_keys=False)
        exp = cfg.get("experiments", {})
        recorder._write_summary({
            "name": name,
            "model": cfg.get("model", {}).get("display_name"),
            "date": exp.get("date"),
            "description": exp.get("description"),
            "status": "running",
            "created_at": _utc_now(),
            "num_folds": cfg.get("dataset", {}).get("num_folds"),
            "total_epochs": cfg.get("hyperparameters", {}).get("num_epochs"),
        })
        return recorder

    def load_config(self) -> dict[str, Any]:
        with self.config_path.open(encoding="utf-8") as f:
            return yaml.safe_load(f)

    def logger(self, filename: str = "train.log") -> logging.Logger:
        """控制台 + run_dir/filename（追加）双输出。"""
        if self._logger is not None:
            return self._logger
        logger = logging.getLogger(f"project1_ADNI.{self.name}")
        logger.setLevel(logging.INFO)
        logger.handlers.clear()
        logger.propagate = False
        file_handler = logging.FileHandler(self.run_dir / filename, mode="a", encoding="utf-8")
        file_handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
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

    def tensorboard_writer(self, fold: int) -> SummaryWriter:
        """每折一个 writer；使用后需调用 ``close()``。"""
        return SummaryWriter(log_dir=str(self.tensorboard_dir / f"fold{fold}"))

    def save_fold_history(self, fold: int, rows: Sequence[Mapping[str, Any]]) -> Path:
        path = self.folds_dir / f"fold{fold}_history.csv"
        if rows:
            write_csv(path, rows, list(rows[0].keys()))
        return path

    def save_checkpoint(self, fold: int, state: Mapping[str, Any]) -> Path:
        path = self.checkpoints_dir / f"fold{fold}_last.pth"
        torch.save(dict(state), path)
        return path

    def finish(self, *, status: str, fold_results: Sequence[Mapping[str, float]],
               summary: Mapping[str, Mapping[str, float]]) -> None:
        """写入 cv_metrics.csv、summary.json 与 summary.md。"""
        rows = [{"fold": i + 1, **{k: m[k] for k in METRIC_KEYS}} for i, m in enumerate(fold_results)]
        rows.append({"fold": "mean", **{k: summary[k]["mean"] for k in METRIC_KEYS}} if summary else {"fold": "mean"})
        rows.append({"fold": "std", **{k: summary[k]["std"] for k in METRIC_KEYS}} if summary else {"fold": "std"})
        write_csv(self.run_dir / "cv_metrics.csv", rows, ["fold", *METRIC_KEYS])

        self._write_summary({
            "status": status,
            "finished_at": _utc_now(),
            "folds": list(fold_results),
            "summary": dict(summary),
        })
        self._write_summary_md(status, fold_results, summary)

    def _write_summary(self, fields: Mapping[str, Any]) -> None:
        data = {}
        if self.summary_path.exists():
            data = json.loads(self.summary_path.read_text(encoding="utf-8"))
        data.update(fields)
        self.summary_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def _write_summary_md(self, status: str, fold_results: Sequence[Mapping[str, float]],
                          summary: Mapping[str, Mapping[str, float]]) -> None:
        cfg = self.load_config()
        lines = [
            f"# {self.name}",
            "",
            f"- 模型: {cfg['model']['display_name']}",
            f"- 状态: {status}",
            f"- 描述: {cfg['experiments'].get('description', '')}",
            "",
            "## 五折交叉验证（mean ± std）",
            "",
            "| 指标 | " + " | ".join(k.upper() for k in METRIC_KEYS) + " |",
            "|---|" + "---|" * len(METRIC_KEYS),
        ]
        if summary:
            lines.append("| 结果 | " + " | ".join(
                f"{summary[k]['mean']:.4f} ± {summary[k]['std']:.4f}" for k in METRIC_KEYS) + " |")
        lines += ["", "## 各折结果", "", "| fold | " + " | ".join(k.upper() for k in METRIC_KEYS) + " |",
                  "|---|" + "---|" * len(METRIC_KEYS)]
        for i, m in enumerate(fold_results):
            lines.append(f"| {i + 1} | " + " | ".join(f"{m[k]:.4f}" for k in METRIC_KEYS) + " |")
        (self.run_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
