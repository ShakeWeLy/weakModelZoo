from collections.abc import Sequence

import torch
import torch.nn as nn


class CrossViT(nn.Module):
    """
    CrossViT (ICCV 2021) 占位实现。

    思路: 构建两条使用不同 patch 尺寸的分支（小 patch 分支与大 patch 分支），
    每个分支各自做自注意力，并通过 Cross-Attention 让两个分支的 CLS token 互相交换信息，
    实现多尺度特征融合。

    Args:
        in_channels: 输入图像通道数
        num_classes: 分类类别数
        patch_sizes: 两条分支的 patch 尺寸，例如 (12, 16)
        embed_dims: 两条分支的嵌入维度
        depths: 两条分支的 Transformer 块数量
        num_heads: 两条分支的注意力头数
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1000,
        patch_sizes: Sequence[int] = (12, 16),
        embed_dims: Sequence[int] = (192, 384),
        depths: Sequence[int] = (1, 4),
        num_heads: Sequence[int] = (3, 6),
    ) -> None:
        super().__init__()
        # TODO: 在此实现双分支 Patch Embedding、Cross-Attention 融合与分类头
        raise NotImplementedError("CrossViT 尚未实现")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: 在此实现前向传播
        raise NotImplementedError("CrossViT.forward 尚未实现")
