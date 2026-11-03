"""加载 MMTracking Siamese RPN (ResNet-50) 权重到 SiamRPN_pp。

适用于如 ``siamese_rpn_r50_1x_lasot_*.pth``（meta.mmtrack_version 存在）。
键名映射：
- ``backbone.*`` → 同名
- ``neck.convs.{i}.conv/bn`` → ``neck.layers.{i}.downsample.0/1``
- ``head.cls_heads/reg_heads`` → ``rpn_head.rpns.{i}.cls/loc``
- ``head.cls_weight/reg_weight`` → ``rpn_head.cls_weight/loc_weight``
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

# 仓库根 / pretrained_weights/siamrpnpp/
_REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_MMTRACK_WEIGHTS = (
    _REPO_ROOT
    / "pretrained_weights"
    / "siamrpnpp"
    / "siamese_rpn_r50_1x_lasot_20201218_051019-3c522eff.pth"
)


def remap_mmtrack_key(key: str) -> str | None:
    """把 MMTracking 参数名映射到本仓库命名；无法识别时返回 ``None``。"""
    if key.startswith("module."):
        key = key[len("module.") :]

    if key.startswith("backbone."):
        return key

    if key.startswith("neck.convs."):
        rest = key[len("neck.convs.") :]
        i, sub = rest.split(".", 1)
        if sub.startswith("conv."):
            return f"neck.layers.{i}.downsample.0.{sub[len('conv.'):]}"
        if sub.startswith("bn."):
            return f"neck.layers.{i}.downsample.1.{sub[len('bn.'):]}"
        return None

    if key == "head.cls_weight":
        return "rpn_head.cls_weight"
    if key == "head.reg_weight":
        return "rpn_head.loc_weight"

    for kind, ours_kind in (("cls_heads", "cls"), ("reg_heads", "loc")):
        prefix = f"head.{kind}."
        if not key.startswith(prefix):
            continue
        rest = key[len(prefix) :]
        i, sub = rest.split(".", 1)
        base = f"rpn_head.rpns.{i}.{ours_kind}"
        mapping = (
            ("kernel_convs.conv.", f"{base}.conv_kernel.0."),
            ("kernel_convs.bn.", f"{base}.conv_kernel.1."),
            ("search_convs.conv.", f"{base}.conv_search.0."),
            ("search_convs.bn.", f"{base}.conv_search.1."),
            ("head_convs.0.conv.", f"{base}.head.0."),
            ("head_convs.0.bn.", f"{base}.head.1."),
            ("head_convs.1.conv.", f"{base}.head.3."),
        )
        for src, dst in mapping:
            if sub.startswith(src):
                return dst + sub[len(src) :]
        return None
    return None


def convert_mmtrack_state_dict(state: dict[str, Any]) -> OrderedDict[str, torch.Tensor]:
    converted: OrderedDict[str, torch.Tensor] = OrderedDict()
    skipped: list[str] = []
    for k, v in state.items():
        nk = remap_mmtrack_key(k)
        if nk is None:
            skipped.append(k)
            continue
        converted[nk] = v
    if skipped:
        print(f"[mmtrack] skip unmapped keys: {len(skipped)} e.g. {skipped[:5]}")
    return converted


def load_mmtrack_checkpoint(
    model: nn.Module,
    weights_path: str | Path | None = None,
    *,
    strict: bool = True,
) -> tuple[list[str], list[str]]:
    """加载 MMTracking R50 SiamRPN 权重。

    Returns:
        ``(missing_keys, unexpected_keys)``
    """
    path = Path(weights_path) if weights_path is not None else DEFAULT_MMTRACK_WEIGHTS
    if not path.is_file():
        raise FileNotFoundError(
            f"MMTracking 权重不存在: {path}\n"
            f"请放到 pretrained_weights/siamrpnpp/siamese_rpn_r50_1x_lasot_*.pth"
        )
    raw = torch.load(path, map_location="cpu", weights_only=False)
    state = raw["state_dict"] if isinstance(raw, dict) and "state_dict" in raw else raw
    converted = convert_mmtrack_state_dict(state)
    missing, unexpected = model.load_state_dict(converted, strict=strict)
    print(
        f"[mmtrack] loaded {path.name}: tensors={len(converted)} "
        f"missing={len(missing)} unexpected={len(unexpected)}"
    )
    return list(missing), list(unexpected)
