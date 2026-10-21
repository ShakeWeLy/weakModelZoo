"""官方 TransUNet 封装：R50+ViT-B/16 骨干，默认加载 ImageNet-21k ``R50+ViT-B_16.npz``。

接口对齐 project1_2610：``in_channels`` / ``out_channels`` / ``image_size`` 由框架注入。
输入若为单通道，在 ``VisionTransformer.forward`` 内复制为 3 通道（与上游一致）。
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch.nn as nn

from src.models.segment.TransUNet_official.vit_seg_modeling import CONFIGS, VisionTransformer

_REPO_ROOT = Path(__file__).resolve().parents[4]
_DEFAULT_PRETRAIN = "pretrained_weights/transunet/R50+ViT-B_16.npz"


def _resolve_weight_path(path: str | None) -> Path | None:
    if path is None:
        return None
    text = str(path).strip()
    if not text:
        return None
    p = Path(text)
    if not p.is_absolute():
        p = _REPO_ROOT / p
    return p.resolve()


class TransUNetOfficial(nn.Module):
    """与官方 ``networks.vit_seg_modeling.VisionTransformer`` 等价的可配置封装。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        image_size: int = 224,
        *,
        vit_name: str = "R50-ViT-B_16",
        vit_patches_size: int = 16,
        n_skip: int = 3,
        load_pretrained: bool = True,
        pretrained_path: str | None = _DEFAULT_PRETRAIN,
    ):
        super().__init__()
        # 预训练 R50/ViT 固定 3 通道；单通道在 VisionTransformer.forward 内 repeat
        del in_channels
        if vit_name not in CONFIGS:
            raise KeyError(f"未知 vit_name={vit_name!r}，可选: {', '.join(CONFIGS)}")

        config = copy.deepcopy(CONFIGS[vit_name])
        config.n_classes = int(out_channels)
        config.n_skip = int(n_skip)
        if vit_name.find("R50") != -1:
            config.patches.grid = (
                int(image_size) // int(vit_patches_size),
                int(image_size) // int(vit_patches_size),
            )

        self.vit = VisionTransformer(
            config,
            img_size=int(image_size),
            num_classes=int(out_channels),
        )

        if load_pretrained:
            path = _resolve_weight_path(pretrained_path)
            if path is None or not path.is_file():
                raise FileNotFoundError(
                    f"TransUNet 官方预训练权重不存在: {pretrained_path}；"
                    f"请放到 {_DEFAULT_PRETRAIN}，或设 load_pretrained=false"
                )
            print(f"pretrained_path:{path}")
            self.vit.load_from(weights=np.load(str(path)))

    def forward(self, x):
        return self.vit(x)
