import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, hidden_channels: int = 32):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.query = nn.Linear(in_channels, hidden_channels)
        self.key = nn.Linear(in_channels, hidden_channels)
        self.value = nn.Linear(in_channels, hidden_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) != 3: 
            raise ValueError(f"Input must be a 3D tensor, but got {len(x.shape)}D tensor")
        b, n, c = x.shape  # batch_size, sequence_length, hidden_channels
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        attention = torch.matmul(query, key.transpose(-2, -1)) / (self.hidden_channels ** 0.5)
        attention = torch.softmax(attention, dim=-1)
        return torch.matmul(attention, value)