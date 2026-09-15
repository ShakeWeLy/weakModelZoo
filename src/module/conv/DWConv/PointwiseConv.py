from ..base import ActArg, BaseConv, NormArg

class PointwiseConv(BaseConv):
    """Pointwise convolution followed by optional BN and activation."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,  
        kernel_size: int = 1,  # use kernel size 1 to implement pointwise convolution
        stride: int = 1,
        padding: int = 0,
        groups: int = 1,
        bias: bool | None = None,
        act: ActArg = True,
        bn: NormArg = True,
    ) -> None:
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, groups, bias, act, bn)