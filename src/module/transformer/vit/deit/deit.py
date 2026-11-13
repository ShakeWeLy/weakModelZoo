import torch
import torch.nn as nn


class DeiT(nn.Module):
    """
    DeiT (Data-efficient Image Transformers, ICML 2021) 占位实现。

    思路: 结构与 ViT 基本一致，额外引入一个蒸馏 token (distillation token)，
    通过教师模型（通常为 CNN）的软/硬标签进行知识蒸馏，从而在较少数据上训练 ViT。

    Args:
        in_channels: 输入图像通道数
        image_size: 输入图像边长
        patch_size: patch 边长
        embed_dim: 嵌入维度
        depth: Transformer 块数量
        num_heads: 注意力头数
        mlp_ratio: FFN 隐藏层相对 embed_dim 的倍数
        num_classes: 分类类别数
        use_distillation_token: 是否启用蒸馏 token
    """

    def __init__(
        self,
        in_channels: int = 3,
        image_size: int = 224,
        patch_size: int = 16,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        num_classes: int = 1000,
        use_distillation_token: bool = True,
    ) -> None:
        super().__init__()
        # TODO: 在此实现 Patch Embedding、cls/distillation token、Transformer 块与分类头
        raise NotImplementedError("DeiT 尚未实现")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: 在此实现前向传播
        raise NotImplementedError("DeiT.forward 尚未实现")
