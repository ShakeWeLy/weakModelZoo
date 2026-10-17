import torch
import torch.nn as nn

from ..base import ActArg, BaseConv, NormArg


class PConv(nn.Module):
    """
    Partial Convolution (FasterNet, CVPR 2023)。

    思路: 只对输入通道的一部分 (channels // n_div) 做空间卷积，其余通道直接保持不变，
    从而大幅降低 FLOPs 与访存开销。

    Args:
        channels: 输入/输出通道数（PConv 不改变通道数）
        kernel_size: 卷积核大小，默认 3
        n_div: 参与卷积的通道比例分母，卷积通道 = channels // n_div，默认 4
        stride: 步幅，只能为 1（见 __init__ 中的说明），默认 1
        padding: 填充，默认 None 时取 kernel_size // 2
        act: 激活函数配置（作用于卷积分支）
        bn: 归一化配置（作用于卷积分支）
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        n_div: int = 4,
        stride: int = 1,
        padding: int | None = None,
        act: ActArg = True,
        bn: NormArg = True,
    ) -> None:
        super().__init__()

        # [修复] 之前的写法是 `channels // n_div != 0`, 对所有合法输入都会误报错,
        # 例如 channels=16, n_div=4 时 16 // 4 = 4 != 0 直接抛出异常。
        # 正确的判断应使用取余 `%`。
        if channels % n_div != 0:
            raise ValueError("channels 必须能被 n_div 整除")

        # [修复] stride 必须为 1。
        # 卷积分支 (前 channels // n_div 个通道) 会被下采样, 而直接保留的其余通道不会,
        # forward 中 torch.cat 时空间尺寸不一致, 报错:
        # "Sizes of tensors must match except in dimension 1"。
        # FasterNet 原文的 PConv 也只使用 stride=1, 因此这里直接禁止其他取值。
        if stride != 1:
            raise ValueError("PConv 仅支持 stride=1, 因为保留的通道无法与下采样后的卷积分支拼接")

        self.channels_conv = channels // n_div
        self.rest_conv = channels - self.channels_conv
        self.conv = BaseConv(
            in_channels=self.channels_conv,
            out_channels=self.channels_conv,
            stride=stride,
            padding=kernel_size // 2 if padding is None else padding,
            kernel_size=kernel_size,
            act=act,
            bn=bn,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [修复] 之前的写法是 `cat([x, conv(x1)])`: 完整的输入 x 被整体拼接进来,
        # 输出通道变成 channels + channels_conv, 与 channels 不一致。
        # 正确做法: 只对前一部分 x1 做卷积, 后一部分 x2 原样保留, 再在通道维拼接。
        x1, x2 = torch.split(x, [self.channels_conv, self.rest_conv], dim=1)
        return torch.cat([self.conv(x1), x2], dim=1)
