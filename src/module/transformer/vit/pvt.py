from collections.abc import Sequence

import torch
import torch.nn as nn


class PVT(nn.Module):
    """
    PVT (Pyramid Vision Transformer, ICCV 2021) 占位实现。

    思路: 引入金字塔结构，逐阶段下采样（分辨率逐级减半），使 ViT 能输出多尺度特征；
    并用空间缩减注意力 (Spatial Reduction Attention) 降低 K/V 的计算量。

    Args:
        in_channels: 输入图像通道数
        num_classes: 分类类别数
        embed_dims: 各阶段的嵌入维度
        depths: 各阶段 Transformer 块数量
        num_heads: 各阶段注意力头数
        sr_ratios: 各阶段空间缩减比例（用于 SRA）
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1000,
        embed_dims: Sequence[int] = (64, 128, 320, 512),
        depths: Sequence[int] = (2, 2, 2, 2),
        num_heads: Sequence[int] = (1, 2, 5, 8),
        sr_ratios: Sequence[int] = (8, 4, 2, 1),
    ) -> None:
        super().__init__()
        # TODO: 在此实现逐阶段 Patch Embedding、SRA 注意力块与分类头
        raise NotImplementedError("PVT 尚未实现")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: 在此实现前向传播
        raise NotImplementedError("PVT.forward 尚未实现")
