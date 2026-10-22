"""官方 Swin-UNet 封装：骨干来自上游 ``SwinTransformerSys``，默认加载 ImageNet Swin-Tiny 预训练。

接口对齐 project1_2610：``in_channels`` / ``out_channels`` / ``image_size`` 由框架注入。
骨干始终按 3 通道构建（与 ImageNet 预训练一致）；若输入为单通道，在 ``forward`` 中复制为 3 通道。
"""

from __future__ import annotations

import copy
from pathlib import Path

import torch
import torch.nn as nn

from src.models.segment.Swin_Unet_official.swin_transformer_unet_skip_expand_decoder_sys import (  # noqa: E402
    SwinTransformerSys,
)

_REPO_ROOT = Path(__file__).resolve().parents[4]
_DEFAULT_PRETRAIN = "pretrained_weights/swin/swin_tiny_patch4_window7_224.pth"


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


class SwinUNetOfficial(nn.Module):
    """与官方 ``networks/vision_transformer.SwinUnet`` 等价的可配置封装。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        image_size: int = 224,
        *,
        patch_size: int = 4,
        embed_dim: int = 96,
        depths: list[int] | tuple[int, ...] = (2, 2, 2, 2),
        depths_decoder: list[int] | tuple[int, ...] = (2, 2, 2, 1),
        num_heads: list[int] | tuple[int, ...] = (3, 6, 12, 24),
        window_size: int = 7,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.2,
        ape: bool = False,
        patch_norm: bool = True,
        use_checkpoint: bool = False,
        final_upsample: str = "expand_first",
        load_pretrained: bool = True,
        pretrained_path: str | None = _DEFAULT_PRETRAIN,
    ):
        super().__init__()
        # in_channels 由框架注入；预训练 Swin 第一层固定为 3 通道，与官方一致
        del in_channels
        self.num_classes = int(out_channels)
        self.image_size = int(image_size)
        self.swin_unet = SwinTransformerSys(
            img_size=self.image_size,
            patch_size=int(patch_size),
            in_chans=3,
            num_classes=self.num_classes,
            embed_dim=int(embed_dim),
            depths=list(depths),
            depths_decoder=list(depths_decoder),
            num_heads=list(num_heads),
            window_size=int(window_size),
            mlp_ratio=float(mlp_ratio),
            qkv_bias=bool(qkv_bias),
            qk_scale=qk_scale,
            drop_rate=float(drop_rate),
            attn_drop_rate=float(attn_drop_rate),
            drop_path_rate=float(drop_path_rate),
            ape=bool(ape),
            patch_norm=bool(patch_norm),
            use_checkpoint=bool(use_checkpoint),
            final_upsample=str(final_upsample),
        )
        if load_pretrained:
            path = _resolve_weight_path(pretrained_path)
            if path is None or not path.is_file():
                raise FileNotFoundError(
                    f"Swin-UNet 官方预训练权重不存在: {pretrained_path}；"
                    f"请放到 {_DEFAULT_PRETRAIN}，或设 load_pretrained=false"
                )
            self.load_from(path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        return self.swin_unet(x)

    def load_from(self, pretrained_path: str | Path) -> None:
        """加载官方 ImageNet Swin-Tiny 权重（与上游 ``SwinUnet.load_from`` 相同逻辑）。"""
        path = Path(pretrained_path)
        print(f"pretrained_path:{path}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pretrained_dict = torch.load(path, map_location=device, weights_only=False)
        if "model" not in pretrained_dict:
            print("---start load pretrained modle by splitting---")
            pretrained_dict = {k[17:]: v for k, v in pretrained_dict.items()}
            for k in list(pretrained_dict.keys()):
                if "output" in k:
                    print(f"delete key:{k}")
                    del pretrained_dict[k]
            self.swin_unet.load_state_dict(pretrained_dict, strict=False)
            return

        print("---start load pretrained modle of swin encoder---")
        pretrained_dict = pretrained_dict["model"]
        model_dict = self.swin_unet.state_dict()
        full_dict = copy.deepcopy(pretrained_dict)
        for k, v in pretrained_dict.items():
            if "layers." in k:
                current_layer_num = 3 - int(k[7:8])
                current_k = "layers_up." + str(current_layer_num) + k[8:]
                full_dict.update({current_k: v})
        for k in list(full_dict.keys()):
            if k in model_dict and full_dict[k].shape != model_dict[k].shape:
                print(
                    f"delete:{k};shape pretrain:{full_dict[k].shape};shape model:{model_dict[k].shape}"
                )
                del full_dict[k]
        self.swin_unet.load_state_dict(full_dict, strict=False)
