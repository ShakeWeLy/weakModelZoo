from typing import TypeAlias
import torch
import torch.nn as nn

NormArg: TypeAlias = bool | nn.Module | None
ActArg: TypeAlias = bool | nn.Module | None


class BaseConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int | None = None,
        groups: int = 1,
        bias: bool | None = None,
        act: ActArg = True,
        bn: NormArg = True,
    ) -> None:
        super().__init__()

        if padding is None:
            padding = kernel_size // 2

        if not 1 <= groups <= in_channels:
            raise ValueError("groups 必须位于 1 和 in_channels 之间")
        if in_channels % groups != 0 or out_channels % groups != 0:
            raise ValueError("in_channels 和 out_channels 必须能被 groups 整除")

        if bias is None:
            bias = not bool(bn)

        layers: list[nn.Module] = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                groups=groups,
                bias=bias,
            )
        ]

        if isinstance(bn, nn.Module):
            layers.append(bn)
        elif bn:
            layers.append(nn.BatchNorm2d(out_channels))

        if isinstance(act, nn.Module):
            layers.append(act)
        elif act:
            layers.append(nn.SiLU())

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)