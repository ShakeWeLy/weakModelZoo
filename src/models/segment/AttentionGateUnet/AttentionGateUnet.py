from typing import override

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.segment.unet.unet import DoubleConv, UNet
from src.module.attention.AttentionGate.attentionGate import AttentionGate


@override
class AttentionGateUp(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        gate_hidden_channels: int | None = None,
    ):
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels,
            in_channels // 2,
            kernel_size=2,
            stride=2,
        )
        self.conv = DoubleConv(in_channels, out_channels)
        gate_hidden = gate_hidden_channels or max(out_channels // 2, 1)
        self.attention_gate = AttentionGate(
            g_in_channels=out_channels,
            x_in_channels=out_channels,
            hidden_channels=gate_hidden,
        )

    def forward(self, x_input: torch.Tensor, x_skip: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x_input)
        diff_y = x_skip.size(2) - x1.size(2)
        diff_x = x_skip.size(3) - x1.size(3)
        x1 = F.pad(
            x1,
            [
                diff_x // 2,
                diff_x - diff_x // 2,
                diff_y // 2,
                diff_y - diff_y // 2,
            ],
        )
        x_skip = self.attention_gate(x1, x_skip)
        x = torch.cat([x_skip, x1], dim=1)
        return self.conv(x)


class AttentionGateUnet(UNet):
    def __init__(self, in_channels: int, out_channels: int, hidden_channels: int) -> None:
        super().__init__(in_channels, out_channels, hidden_channels)
        hidden = hidden_channels
        self.up1 = AttentionGateUp(hidden * 16, hidden * 8)
        self.up2 = AttentionGateUp(hidden * 8, hidden * 4)
        self.up3 = AttentionGateUp(hidden * 4, hidden * 2)
        self.up4 = AttentionGateUp(hidden * 2, hidden)
