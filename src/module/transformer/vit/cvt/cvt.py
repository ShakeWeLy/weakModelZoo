from calendar import c
from collections.abc import Sequence
from turtle import forward

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.module.conv import DSConv


class CvtEmbedding(nn.Module):
    def __init__(self, in_channels: int, kernel_size: int, stride: int, padding: int,  embed_dims: int = 1):
        super().__init__()
        self.embed_dims = embed_dims
        self.dw_conv = nn.Conv2d(in_channels, embed_dims, kernel_size=kernel_size, stride=stride, padding=padding)
        self.norm = nn.LayerNorm(embed_dims)
        self.act = nn.GELU()
        self.proj = nn.Linear(embed_dims, embed_dims)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, _, _ = x.shape
        x = self.dw_conv(x)
        _, C, W, H = x.shape
        x = x.reshape(B, self.embed_dims, -1).transpose(1, 2)  # [B, L, C]
        x = self.norm(x)
        x = self.act(x)
        x = self.proj(x)
        x = x.transpose(1, 2).reshape(B, self.embed_dims, W, H)
        return x

class CvtAttention(nn.Module):
    def __init__(self, in_channels: int,
    embed_dims: int,
    num_heads: int,
    embed_kernel_size: int,
    embed_stride: int,
    embed_padding: int,
    proj_kernel_size: int,
    proj_stride: int,
    proj_padding: int):
        super().__init__()
        self.embed_dims = embed_dims
        self.dw_conv_q = DSConv(in_channels, embed_dims, kernel_size=proj_kernel_size, stride=proj_stride, padding=proj_padding)
        self.dw_conv_k = DSConv(in_channels, embed_dims, kernel_size=proj_kernel_size, stride=proj_stride, padding=proj_padding)
        self.dw_conv_v = DSConv(in_channels, embed_dims, kernel_size=proj_kernel_size, stride=proj_stride, padding=proj_padding)
        self.norm = nn.LayerNorm(embed_dims)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        x_q = self.dw_conv_q(x).reshape(B, self.embed_dims, -1).transpose(1, 2)  # [B, L, embed_dims]
        x_k = self.dw_conv_k(x).reshape(B, self.embed_dims, -1).transpose(1, 2)
        x_v = self.dw_conv_v(x).reshape(B, self.embed_dims, -1).transpose(1, 2)
        x_q = self.act(self.norm(x_q))
        x_k = self.act(self.norm(x_k))
        x_v = self.act(self.norm(x_v))
        attention = torch.matmul(x_q, x_k.transpose(1, 2)) / (self.embed_dims ** 0.5)
        attention = F.softmax(attention, dim=-1)
        x = torch.matmul(attention, x_v)
        x = x.transpose(1, 2).reshape(B, self.embed_dims, H, W)
        return x # [B, embed_dims, H, W]

class CvtBlock(nn.Module):
    def __init__(self,
                in_channels: int,
                embed_dims: int,
                num_heads: int,
                embed_kernel_size: int,
                embed_stride: int,
                embed_padding: int,
                proj_kernel_size: int,
                proj_stride: int,
                proj_padding: int,
                ffn_hidden_channels: int,
                ):
        super().__init__()
        self.attention = CvtAttention(in_channels, embed_dims, num_heads, embed_kernel_size, embed_stride, embed_padding, proj_kernel_size, proj_stride, proj_padding)
        self.feedForward = nn.Sequential(
            nn.Linear(embed_dims, ffn_hidden_channels),
            nn.GELU(),
            nn.Linear(ffn_hidden_channels, embed_dims),
        )

        self.norm1 = nn.LayerNorm(embed_dims)
        self.norm2 = nn.LayerNorm(embed_dims)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.norm1(x + self.attention(x))
        x = self.norm2(x + self.feedForward(x.reshape(B, C, -1).transpose(1,2)).transpose(1,2).reshape(B, C, H, W))
        return x # [B, C, H, W]
 

class CvT(nn.Module):
    """
    CvT (Convolutional vision Transformer, ICCV 2021) 占位实现。

    思路: 在 ViT 中引入卷积：用带步幅的卷积替代线性 Patch Embedding（卷积 token 嵌入），
    并用深度可分离卷积替代 Q/K/V 的线性投影（卷积投影），同样采用多阶段结构。

    Args:
        in_channels: 输入图像通道数
        num_classes: 分类类别数
        embed_dims: 各阶段的嵌入维度
        depths: 各阶段 Transformer 块数量
        num_heads: 各阶段注意力头数
        patch_sizes: 各阶段卷积 token 嵌入的卷积核大小
        patch_strides: 各阶段卷积 token 嵌入的步幅
    """

    def __init__(
        self,
        in_channels: int,
        embed_kernel_size: int,
        embed_stride: int,
        embed_padding: int,
        depth: int,
        proj_kernel_size: int,
        proj_stride: int,
        proj_padding: int,
        embed_dims: int,
        num_heads: int,
        ffn_hidden_channels: int,
        num_classes: int = 1000,
    ) -> None:
        super().__init__()
        self.embed = CvtEmbedding(in_channels, embed_kernel_size, embed_stride, embed_padding, embed_dims=1)
        self.blocks = nn.ModuleList(
            [
                CvtBlock(in_channels, embed_dims, num_heads, embed_kernel_size, embed_stride, embed_padding, proj_kernel_size, proj_stride, proj_padding, ffn_hidden_channels)
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dims)
        self.head = nn.Linear(embed_dims, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed(x)
        B, C, H, W = x.shape
        for block in self.blocks:
            x = block(x)
        x = x.reshape(B, C, -1).transpose(1,2)
        x = self.norm(x)
        x = x.mean(dim=1)
        return self.head(x)
