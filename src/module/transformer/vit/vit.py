import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.module.conv import BaseConv
from src.module.transformer.positionEmbedding.positionEmbedding import LearnablePositionalEmbedding
from src.module.transformer.transformerBlock import TransformerDecoderBlock


class PatchEmbedding(nn.Module):
    """将图像切分为 patch 并映射到 embed_dim。"""

    def __init__(
        self,
        in_channels: int,
        image_size: int,
        patch_size: int,
        embed_dim: int,
        use_cls_token: bool = True,
    ):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size 必须能被 patch_size 整除")

        self.patch_size = patch_size
        self.use_cls_token = use_cls_token
        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size ** 2

        self.patch_embed = BaseConv(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
            act=False,
            bn=False,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if use_cls_token else None
        num_tokens = self.num_patches + (1 if use_cls_token else 0)
        self.position_embedding = LearnablePositionalEmbedding(num_tokens, embed_dim)
        if use_cls_token:
            nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        batch_size, _, height, width = x.shape
        if height % self.patch_size != 0 or width % self.patch_size != 0:
            raise ValueError(
                f"输入尺寸 ({height}, {width}) 必须能被 patch_size={self.patch_size} 整除"
            )

        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)  # [B, N, embed_dim]

        if self.use_cls_token:
            cls_token = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_token, x], dim=1)

        return self.position_embedding(x)


class ViT(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        image_size: int = 224,
        patch_size: int = 16,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        ffn_hidden_channels: int = 3072,
        num_classes: int = 1000,
        use_cls_token: bool = True,
    ):
        super().__init__()
        self.use_cls_token = use_cls_token
        self.patch_embed = PatchEmbedding(
            in_channels=in_channels,
            image_size=image_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            use_cls_token=use_cls_token,
        )
        self.blocks = nn.ModuleList(
            [
                TransformerDecoderBlock(embed_dim, num_heads, ffn_hidden_channels)
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        if self.use_cls_token:
            x = x[:, 0]
        else:
            x = x.mean(dim=1)
        return self.head(x)


if __name__ == "__main__":
    x = torch.randn(2, 3, 224, 224)
    model = ViT(
        in_channels=3,
        image_size=224,
        patch_size=16,
        embed_dim=768,
        depth=2,
        num_heads=8,
        ffn_hidden_channels=1024,
        num_classes=14,
    )
    y = model(x)
    print(y.shape)
