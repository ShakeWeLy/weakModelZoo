"""SiamSPN++：基于深层骨干 + 多层 depthwise 互相关 RPN 的孪生跟踪器。

结构对齐 SiamRPN++（CVPR 2019 / pysot）::

    Backbone (ResNet-50 atrous) → AdjustAllLayer → MultiRPN (weighted)

输出::

    model.train() / forward(template, search)
        -> dict(cls, loc)  原始 logits，供 SiamSPNLoss 使用
    model.template(z) + model.track(x)
        -> dict(cls, loc)  推理时缓存模板特征

用法::

    from src.models.track.SiamSPN import SiamSPN_pp, SiamSPNLoss
    model = SiamSPN_pp()
    criterion = SiamSPNLoss()
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from src.models.track.SiamSPN.backbone import ResNetAtrous, resnet50
from src.models.track.SiamSPN.common import AdjustAllLayer, MultiRPN

# 默认：5 个 ratios × 1 个 scale
DEFAULT_ANCHOR_NUM = 5
DEFAULT_USED_LAYERS: tuple[int, ...] = (2, 3, 4)
DEFAULT_ADJUST_CHANNELS: tuple[int, ...] = (256, 256, 256)


class SiamSPN_pp(nn.Module):
    """SiamSPN++ 主网络。

    Args:
        anchor_num: 每个位置的锚框数（默认 5）。
        used_layers: 骨干输出层索引，默认 layer2/3/4。
        adjust_channels: Adjust 后各层通道，默认全 256。
        weighted: MultiRPN 是否使用可学习层权重融合。
        backbone: 可选外部骨干；默认构建 ResNet-50 atrous。
    """

    def __init__(
        self,
        anchor_num: int = DEFAULT_ANCHOR_NUM,
        used_layers: Sequence[int] = DEFAULT_USED_LAYERS,
        adjust_channels: Sequence[int] = DEFAULT_ADJUST_CHANNELS,
        weighted: bool = True,
        backbone: ResNetAtrous | None = None,
    ) -> None:
        super().__init__()
        self.anchor_num = anchor_num
        self.used_layers = tuple(used_layers)

        self.backbone = backbone if backbone is not None else resnet50(used_layers=self.used_layers)
        in_channels = self.backbone.out_channels
        if len(adjust_channels) != len(in_channels):
            raise ValueError(
                f"adjust_channels 长度 ({len(adjust_channels)}) 需与骨干输出层数 ({len(in_channels)}) 一致"
            )

        self.neck = AdjustAllLayer(in_channels, adjust_channels)
        self.rpn_head = MultiRPN(anchor_num, adjust_channels, weighted=weighted)

        # 推理时缓存的模板特征
        self.zf: list[torch.Tensor] | None = None

    def extract(self, x: torch.Tensor) -> list[torch.Tensor]:
        """骨干 + Adjust，返回多层特征。"""
        return self.neck(self.backbone(x))

    def template(self, z: torch.Tensor) -> None:
        """缓存模板特征（跟踪初始化）。"""
        self.zf = self.extract(z)

    def track(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """用缓存模板对搜索图做一次跟踪前向。"""
        if self.zf is None:
            raise RuntimeError("请先调用 template(z) 缓存模板特征")
        xf = self.extract(x)
        cls, loc = self.rpn_head(self.zf, xf)
        return {"cls": cls, "loc": loc}

    def forward(
        self,
        template: torch.Tensor,
        search: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """训练前向：模板与搜索成对输入。

        Args:
            template: ``(B, 3, Z, Z)``，常见 Z=127
            search: ``(B, 3, X, X)``，常见 X=255

        Returns:
            ``cls``: ``(B, 2*anchor_num, H, W)``
            ``loc``: ``(B, 4*anchor_num, H, W)``
        """
        zf = self.extract(template)
        xf = self.extract(search)
        cls, loc = self.rpn_head(zf, xf)
        return {"cls": cls, "loc": loc}
