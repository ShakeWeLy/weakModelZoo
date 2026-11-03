"""SiamRPN++：基于深层骨干 + 多层 depthwise 互相关 RPN 的孪生跟踪器。

结构对齐 SiamRPN++（CVPR 2019 / pysot）::

    Backbone (ResNet-50 atrous) → AdjustAllLayer → MultiRPN (weighted)

默认加载 MMTracking 整网预训练
（``pretrained_weights/siamrpnpp/siamese_rpn_r50_1x_lasot_*.pth``），
不再使用原生 ImageNet ResNet 部分加载。

用法::

    from src.models.track.SiamRPN import SiamRPN_pp, SiamRPNLoss
    model = SiamRPN_pp()  # 默认加载 MMTracking 权重
    criterion = SiamRPNLoss()
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn

from src.models.track.SiamRPN.backbone import ResNetAtrous, resnet50
from src.models.track.SiamRPN.common import AdjustAllLayer, MultiRPN
from src.models.track.SiamRPN.load_mmtrack import DEFAULT_MMTRACK_WEIGHTS, load_mmtrack_checkpoint

# 默认：5 个 ratios × 1 个 scale
DEFAULT_ANCHOR_NUM = 5
DEFAULT_USED_LAYERS: tuple[int, ...] = (2, 3, 4)
DEFAULT_ADJUST_CHANNELS: tuple[int, ...] = (256, 256, 256)


class SiamRPN_pp(nn.Module):
    """SiamRPN++ 主网络。

    Args:
        anchor_num: 每个位置的锚框数（默认 5）。
        used_layers: 骨干输出层索引，默认 layer2/3/4。
        adjust_channels: Adjust 后各层通道，默认全 256。
        weighted: MultiRPN 是否使用可学习层权重融合。
        backbone: 可选外部骨干；默认构建 ResNet-50 atrous。
        pretrained: 是否加载 MMTracking 整网权重（默认 True）。
        weights_path: MMTracking ``.pth``；``None`` 用默认路径。
    """

    def __init__(
        self,
        anchor_num: int = DEFAULT_ANCHOR_NUM,
        used_layers: Sequence[int] = DEFAULT_USED_LAYERS,
        adjust_channels: Sequence[int] = DEFAULT_ADJUST_CHANNELS,
        weighted: bool = True,
        backbone: ResNetAtrous | None = None,
        pretrained: bool = True,
        weights_path: str | Path | None = None,
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

        if pretrained:
            self.load_mmtrack(weights_path)

    def freeze_backbone(self, freeze: bool = True) -> None:
        """冻结 / 解冻主干参数（前若干 epoch 常用）。"""
        for p in self.backbone.parameters():
            p.requires_grad = not freeze
        if freeze:
            self.backbone.eval()

    def load_mmtrack(self, weights_path: str | Path | None = None, *, strict: bool = True) -> tuple[list[str], list[str]]:
        """加载 MMTracking ResNet-50 SiamRPN 整网权重（自动键名映射）。"""
        path = weights_path if weights_path is not None else DEFAULT_MMTRACK_WEIGHTS
        return load_mmtrack_checkpoint(self, path, strict=strict)

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
