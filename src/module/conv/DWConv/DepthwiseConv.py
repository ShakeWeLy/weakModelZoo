from ..base import ActArg, BaseConv, NormArg


class DepthwiseConv(BaseConv):
    """Depthwise convolution followed by optional BN and activation."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int | None = None,
        bias: bool | None = None,
        act: ActArg = True,
        bn: NormArg = True,
    ) -> None:
        super().__init__(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=channels,  # use groups equal to channels to implement depthwise convolution
            bias=bias,
            act=act,
            bn=bn,
        )