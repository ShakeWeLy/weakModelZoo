from collections.abc import Sequence

import torch
import torch.nn as nn


class T2TViT(nn.Module):
    """
    T2T-ViT (Tokens-to-Token ViT, ICCV 2021) 占位实现。

    思路: 用 Tokens-to-Token (T2T) 模块代替普通的 Patch Embedding，
    通过多次 "展开 (unfold) + 重组" 迭代聚合相邻 token，让局部结构被更好地建模，
    再送入标准 Transformer 编码器。

    Args:
        in_channels: 输入图像通道数
        image_size: 输入图像边长
        embed_dim: 最终嵌入维度
        depth: Transformer 块数量
        num_heads: 注意力头数
        num_classes: 分类类别数
        t2t_kernel_sizes: 各次 T2T 迭代的 unfold 核大小
        t2t_strides: 各次 T2T 迭代的 unfold 步幅
    """

    def __init__(
        self,
        in_channels: int = 3,
        image_size: int = 224,
        embed_dim: int = 384,
        depth: int = 14,
        num_heads: int = 6,
        num_classes: int = 1000,
        t2t_kernel_sizes: Sequence[int] = (7, 3, 3),
        t2t_strides: Sequence[int] = (4, 2, 2),
    ) -> None:
        super().__init__()
        # TODO: 在此实现 Tokens-to-Token 模块、Transformer 编码器与分类头
        raise NotImplementedError("T2TViT 尚未实现")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: 在此实现前向传播
        raise NotImplementedError("T2TViT.forward 尚未实现")
