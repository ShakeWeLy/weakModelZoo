import torch
import torch.nn as nn

from ..base import ActArg, BaseConv, NormArg


class GSConv(nn.Module):
    """
    GSConv (Slim-neck, 2022)。

    思路: 把输出通道平均分成两半，一半用标准卷积提取特征，另一半用深度卷积 (DepthwiseConv)
    廉价提取，拼接后通过 channel shuffle 混合两部分信息，兼顾精度与计算量。

    Args:
        in_channels: 输入通道数
        out_channels: 输出通道数，必须为偶数，且能被 groups 整除
        kernel_size: 卷积核大小，默认 3
        stride: 步幅，默认 1（只作用于标准卷积分支，深度卷积分支固定为 1）
        groups: channel shuffle 的分组数，默认 2
        act: 激活函数配置
        bn: 归一化配置
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 2,
        act: ActArg = True,
        bn: NormArg = True,
    ) -> None:
        super().__init__()

        # [修复] 之前 out_channels 为奇数 (如 33) 时, 两路拼接后的总通道数与 out_channels 不一致;
        # 而 nn.ChannelShuffle 要求通道数能被 groups 整除, 这个错误只会在 forward 时才报出
        # ("Number of channels must be divisible by groups")。这里提前检查, 给出清晰提示。
        if out_channels % 2 != 0:
            raise ValueError("out_channels 必须为偶数, 因为两路各占一半通道")
        if out_channels % groups != 0:
            raise ValueError("out_channels 必须能被 groups 整除 (channel shuffle 的要求)")

        # 两路各占一半通道。
        # [修复] 之前使用 ratio 控制比例, 但深度卷积要求输入输出通道相同, 所以两路只能各占一半,
        # ratio 参数已移除。
        self.half_channels = out_channels // 2

        # 标准卷积分支: 负责下采样 (stride 在这里生效)
        self.sc_conv = BaseConv(
            in_channels=in_channels,
            out_channels=self.half_channels,
            kernel_size=kernel_size,
            stride=stride,
            act=act,
            bn=bn,
        )

        # 深度卷积分支: 输入输出通道相同 (都是 half_channels), groups = half_channels, 即每个通道各自卷积。
        # [修复] stride 必须固定为 1。之前这里传的是外部的 stride, 但 sc_conv 已经下采样过一次,
        # 深度卷积分支再下采样会导致两路空间尺寸不一致, torch.cat 报错:
        # "Sizes of tensors must match except in dimension 1"。
        # [修复] 之前使用的 DSConv (深度 + 逐点) 已替换为纯深度卷积, 与 Slim-neck 原文一致。
        self.dw_conv = BaseConv(
            in_channels=self.half_channels,
            out_channels=self.half_channels,
            kernel_size=kernel_size,
            stride=1,
            groups=self.half_channels,
            act=act,
            bn=bn,
        )

        # [修复] 之前属性名拼写为 chanel_shuffle, 已更正为 channel_shuffle。
        self.channel_shuffle = nn.ChannelShuffle(groups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_sc = self.sc_conv(x)
        x_dw = self.dw_conv(x_sc)
        x = torch.cat([x_sc, x_dw], dim=1)
        return self.channel_shuffle(x)
