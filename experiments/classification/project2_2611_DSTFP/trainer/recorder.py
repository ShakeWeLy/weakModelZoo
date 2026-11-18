"""实验目录与命名工具（占位）。

布局::

    <project>/runs/<display_name>/<seq>_<tag>_V<n>/
    ├── config.yaml
    ├── train.log
    ├── summary.json / summary.md
    ├── checkpoints/
    └── tensorboard/
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


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


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class RunRecorder:
    """占位 recorder：只创建目录骨架与 summary.json，训练逻辑接通后再扩展。"""

    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        self.name = self.run_dir.name
        self.checkpoints_dir = self.run_dir / "checkpoints"
        self.tensorboard_dir = self.run_dir / "tensorboard"
        self.summary_path = self.run_dir / "summary.json"

    @classmethod
    def create(cls, runs_root: Path, name: str, cfg: Mapping[str, Any], *, overwrite: bool = False) -> RunRecorder:
        import shutil

        run_dir = Path(runs_root) / name
        if run_dir.exists():
            if not overwrite:
                raise FileExistsError(f"run 已存在: {run_dir}")
            shutil.rmtree(run_dir)
        recorder = cls(run_dir)
        for d in (recorder.checkpoints_dir, recorder.tensorboard_dir):
            d.mkdir(parents=True, exist_ok=True)
        summary = {
            "name": name,
            "model": cfg.get("model", {}).get("display_name"),
            "date": cfg.get("experiments", {}).get("date"),
            "description": cfg.get("experiments", {}).get("description"),
            "status": "initialized",
            "created_at": _utc_now(),
            "note": "placeholder run; training pipeline not wired yet",
        }
        recorder.summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return recorder
