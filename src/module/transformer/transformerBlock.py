'''
TransformerBlock is a block that consists of a multi-head attention mechanism and a feed-forward network.
'''

import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.module.attention.SelfAttention.multiHeadAttention import MultiHeadAttention


class TransformerDecoderBlock(nn.Module):
    def __init__(self, embed_dim: int = 32, num_heads: int = 8, ffn_hidden_channels: int = 1024):
        super().__init__()
        self.multiHeadAttention = MultiHeadAttention(embed_dim, num_heads)

        self.feedForward = nn.Sequential(
            nn.Linear(embed_dim, ffn_hidden_channels),
            nn.GELU(),
            nn.Linear(ffn_hidden_channels, embed_dim),
        )

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x + self.multiHeadAttention(x))
        x = self.norm2(x + self.feedForward(x))
        return x


if __name__ == "__main__":
    x = torch.randn(2, 196, 768)
    model = TransformerDecoderBlock(
        embed_dim=768,
        num_heads=8,
        ffn_hidden_channels=1024,
    )
    y = model(x)
    print(y.shape)
