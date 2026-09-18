'''
MultiHeadAttention is a self-attention mechanism that uses multiple heads to process the input.
Because the self-attention mechanism is a bottleneck, using multiple heads can help the model to process the input more efficiently.
'''

import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int = 32, num_heads: int = 8):
        super().__init__()
        self.embed_dim = embed_dim
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim must be divisible by num_heads, but got {embed_dim} and {num_heads}"
            )
        self.head_embed_dim = embed_dim // num_heads
        self.num_heads = num_heads
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.out = nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Input must be a 3D tensor, but got {x.ndim}D tensor")
        batch_size, seq_len, channels = x.shape
        if channels != self.embed_dim:
            raise ValueError(f"Input channels must be {self.embed_dim}, but got {channels}")

        qkv = self.qkv(x).reshape(
            batch_size, seq_len, 3, self.num_heads, self.head_embed_dim
        )
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attention = torch.matmul(q, k.transpose(-2, -1)) / (self.head_embed_dim ** 0.5)
        if attn_mask is not None:
            attention = attention + attn_mask.unsqueeze(1)
        attention = torch.softmax(attention, dim=-1)
        out = torch.matmul(attention, v)
        out = out.transpose(1, 2).reshape(batch_size, seq_len, self.embed_dim)
        return self.out(out)


if __name__ == "__main__":
    x = torch.randn(2, 196, 768)
    model = MultiHeadAttention(embed_dim=768, num_heads=8)
    y = model(x)
    print(y.shape)




'''
                    X
                    │
                Linear
                    │
                  Q K V
                    │
          ┌─────────┼─────────┐
          ↓         ↓         ↓
       Head 1     Head 2     ... Head 8
          │         │             │
       Q₁K₁ᵀ     Q₂K₂ᵀ          Q₈K₈ᵀ
          │         │             │
       Softmax    Softmax       Softmax
          │         │             │
       × V₁       × V₂          × V₈
          │         │             │
          └─────────┼─────────────┘
                    ↓
                  Concat
                    ↓
                Linear
                    ↓
                   Out
'''

