from collections.abc import Sequence

import torch
import torch.nn as nn


class MaxViT(nn.Module):
    """
    MaxViT (Multi-Axis Vision Transformer, ECCV 2022) 占位实现。

    思路: 每个块依次执行三种注意力：MBConv（局部卷积）、Block Attention（局部窗口注意力）
    与 Grid Attention（稀疏全局注意力，在网格维度上采样），以线性复杂度获得全局感受野。

    Args:
        in_channels: 输入图像通道数
        num_classes: 分类类别数
        embed_dims: 各阶段的通道数
        depths: 各阶段块数量
        head_dim: 每个注意力头的维度
        window_size: Block / Grid 注意力的窗口（网格）大小
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1000,
        embed_dims: Sequence[int] = (64, 128, 256, 512),
        depths: Sequence[int] = (2, 2, 5, 2),
        head_dim: int = 32,
        window_size: int = 7,
    ) -> None:
        super().__init__()
        # TODO: 在此实现 MBConv、Block/Grid 注意力与分类头
        raise NotImplementedError("MaxViT 尚未实现")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: 在此实现前向传播
        raise NotImplementedError("MaxViT.forward 尚未实现")
