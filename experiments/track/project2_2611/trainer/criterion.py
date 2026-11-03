"""按 arch 构建跟踪损失。"""

from __future__ import annotations

from typing import Any, Mapping

from torch import nn

from src.models.track.SiamSPN import SiamSPNLoss


def build_criterion(cfg: Mapping[str, Any]) -> nn.Module:
    arch = str(cfg["model"]["arch"]).lower()
    hyp = cfg.get("loss", {})
    if arch == "siamspn_pp":
        return SiamSPNLoss(hyp)
    raise ValueError(f"未知 arch 的损失: {arch}")
