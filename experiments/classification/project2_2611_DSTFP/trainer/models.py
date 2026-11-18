"""模型注册表（占位）。

``configs/models/<x>.toml`` 中 ``[model].arch`` 选择这里的条目。
DSTFP 上游网络实现位于 ``DSTFP/module/models/``，当前仓库缺失，故仅登记路径与类名占位。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .config import ROOT


@dataclass(frozen=True)
class ModelSpec:
    """占位：记录预期源码位置，真正 ``build_model`` 接通后再加载。"""

    module_path: str
    class_name: str
    note: str = ""


MODEL_REGISTRY: dict[str, ModelSpec] = {
    # 双模态 DSTFP（Lightning 侧：LitMultimodalClassifier）
    "dstfp": ModelSpec(
        module_path="DSTFP/module/models/load_model.py",
        class_name="load_model",
        note="期望: swin4d_ver7 + swin3d_transformer + fusion_H_W；models/ 目录当前缺失",
    ),
    # 单模态 fMRI
    "swin4d_fmri": ModelSpec(
        module_path="DSTFP/module/models/swin4d_transformer_ver7.py",
        class_name="SwinTransformer4D",
        note="SwinTransformer4D 实现文件当前缺失",
    ),
    # 单模态 FA
    "swin3d_fa": ModelSpec(
        module_path="DSTFP/module/models/swin3d_transformer.py",
        class_name="SwinTransformer3D",
        note="SwinTransformer3D 实现文件当前缺失",
    ),
}


def build_model(cfg: Mapping[str, Any], **kwargs: Any):
    arch = str(cfg["model"]["arch"])
    if arch not in MODEL_REGISTRY:
        raise KeyError(f"未知 arch={arch!r}，已注册: {sorted(MODEL_REGISTRY)}")
    spec = MODEL_REGISTRY[arch]
    path = ROOT / spec.module_path
    raise NotImplementedError(
        f"DSTFP 模型尚未接通。"
        f" arch={arch}, expect={spec.class_name} @ {path}"
        f" ({spec.note})"
    )
