import torch
import torch.nn as nn

from src.module.conv import BaseConv


class AttentionGate(nn.Module):
    """Attention Gate: 用 gating 特征 g 为 skip 特征 x 生成空间注意力。

    参考 Attention U-Net，输出为 sigmoid(W(g) + W(x)) * x。
    """

    def __init__(self, g_in_channels: int, x_in_channels: int, hidden_channels: int = 32) -> None:
        super().__init__()
        self.w_g = BaseConv(
            g_in_channels, hidden_channels, kernel_size=1, stride=1, padding=0, act=False
        )
        self.w_x = BaseConv(
            x_in_channels, hidden_channels, kernel_size=1, stride=1, padding=0, act=False
        )
        self.relu = nn.ReLU(inplace=True)
        self.psi = nn.Sequential(
            BaseConv(hidden_channels, 1, kernel_size=1, stride=1, padding=0, act=False),
            nn.Sigmoid(),
        )

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if g.shape[2:] != x.shape[2:]:
            raise ValueError(
                f"g 与 x 的空间尺寸必须一致，当前 g={g.shape}, x={x.shape}"
            )
        attention = self.psi(self.relu(self.w_g(g) + self.w_x(x)))
        return attention * x