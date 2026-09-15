import torch
import torch.nn as nn

from ..base import ActArg, NormArg
from .DepthwiseConv import DepthwiseConv
from .PointwiseConv import PointwiseConv

class DWConv(nn.Module):
    """Depthwise convolution followed by pointwise convolution."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int | None = None,
        bias: bool | None = None,
        act: ActArg = True,
        bn: NormArg = True,
    ) -> None:
        super().__init__()
        self.depthwise_conv = DepthwiseConv(in_channels, kernel_size, stride, padding, bias, act, bn)
        self.pointwise_conv = PointwiseConv(in_channels, out_channels, kernel_size=1, stride=1, bias=bias, act=act, bn=bn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise_conv(self.depthwise_conv(x))