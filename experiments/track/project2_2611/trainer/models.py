"""跟踪模型注册与构建。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping

import torch.nn as nn

from src.models.track.SiamSPN import SiamSPN_pp
from src.models.track.SiamSPN.load_mmtrack import DEFAULT_MMTRACK_WEIGHTS

from .config import ROOT

MODEL_ARCHS: dict[str, Callable[..., nn.Module]] = {
    "siamspn_pp": SiamSPN_pp,
}


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def build_model(cfg: Mapping[str, Any]) -> nn.Module:
    model_cfg = cfg["model"]
    arch = str(model_cfg["arch"]).lower()
    if arch not in MODEL_ARCHS:
        raise ValueError(f"未知 arch: {arch}（可选 {', '.join(MODEL_ARCHS)}）")
    params = dict(model_cfg.get("params", {}))

    # 相对路径权重解析到仓库根
    weights_path = params.get("weights_path")
    if weights_path in (None, "", "default"):
        if params.get("pretrained", True):
            params["weights_path"] = str(DEFAULT_MMTRACK_WEIGHTS)
    else:
        wp = Path(str(weights_path))
        if not wp.is_absolute():
            params["weights_path"] = str((ROOT / wp).resolve())

    return MODEL_ARCHS[arch](**params)
