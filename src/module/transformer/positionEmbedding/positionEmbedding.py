import torch
import torch.nn as nn


class LearnablePositionalEmbedding(nn.Module):
    def __init__(self, num_tokens: int, embed_dim: int):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != self.pos_embed.shape[1]:
            raise ValueError(
                f"序列长度 {x.shape[1]} 与位置编码长度 {self.pos_embed.shape[1]} 不一致"
            )
        return x + self.pos_embed
