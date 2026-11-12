"""训练器占位。接通后可参考 project1_ADNI 的 CVTrainer，或直接包装 DSTFP LightningModule。"""

from __future__ import annotations

from typing import Any, Mapping

import torch


def resolve_device(requested: str | None = None) -> torch.device:
    req = (requested or "cuda").lower()
    if req.startswith("cuda") and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class PlaceholderTrainer:
    def __init__(self, cfg: Mapping[str, Any], recorder: Any, device: torch.device):
        self.cfg = cfg
        self.recorder = recorder
        self.device = device

    def fit(self) -> None:
        raise NotImplementedError(
            "训练流程尚未实现。计划："
            "1) build_datamodule；2) build_model / 或包装 LitMultimodalClassifier；"
            "3) 写入 checkpoints / tensorboard / summary.json。"
        )
