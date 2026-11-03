"""按 arch 构建跟踪损失。"""

from __future__ import annotations

from typing import Any, Mapping

from torch import nn

from src.models.track.SiamRPN import SiamRPNLoss


def build_criterion(cfg: Mapping[str, Any]) -> nn.Module:
    arch = str(cfg["model"]["arch"]).lower()
    hyp = cfg.get("loss", {})
    if arch == "siamrpn_pp":
        return SiamRPNLoss(hyp)
    raise ValueError(f"未知 arch 的损失: {arch}")
