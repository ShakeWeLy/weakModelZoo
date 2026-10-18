import torch
import torch.nn as nn

from ...conv.base import BaseConv


class Inception(nn.Module):
    """
    Inception 模块 (GoogLeNet, CVPR 2015, 即 Inception v1) 。

    思路: 并行使用 4 个分支提取多尺度特征, 最后在通道维度拼接,
    让网络自己选择合适的卷积核大小。

    四个分支:
        branch1: 1x1 卷积
        branch2: 1x1 降维 -> 3x3 卷积
        branch3: 1x1 降维 -> 5x5 卷积
        branch4: 3x3 最大池化 (步幅 1) -> 1x1 卷积

    其中 1x1 降维层用于减少 3x3 / 5x5 卷积的输入通道数, 降低计算量 (这是 Inception 的关键设计)。

    通道划分: out_channels 平均分给 4 个分支, 余数归最后一个分支 (branch4)。
    降维通道数为 max(in_channels // 4, 1)。

    Args:
        in_channels: 输入通道数
        out_channels: 输出通道数 (4 个分支输出之和), 至少为 4
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()

        if out_channels < 4:
            raise ValueError("out_channels 至少为 4, 才能分给 4 个分支")

        # 每个分支输出的通道数
        ch1 = out_channels // 4
        ch2 = out_channels // 4
        ch3 = out_channels // 4
        ch4 = out_channels - ch1 - ch2 - ch3  # 余数归最后一个分支

        # 1x1 降维通道数
        reduce_channels = max(in_channels // 4, 1)

        # branch1: 1x1 卷积
        self.branch1 = BaseConv(in_channels, ch1, kernel_size=1)

        # branch2: 1x1 降维 -> 3x3 卷积
        self.branch2_reduce = BaseConv(in_channels, reduce_channels, kernel_size=1)
        self.branch2_conv = BaseConv(reduce_channels, ch2, kernel_size=3)

        # branch3: 1x1 降维 -> 5x5 卷积
        self.branch3_reduce = BaseConv(in_channels, reduce_channels, kernel_size=1)
        self.branch3_conv = BaseConv(reduce_channels, ch3, kernel_size=5)

        # branch4: 3x3 最大池化 (步幅 1, padding 1, 尺寸不变) -> 1x1 卷积
        self.branch4_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.branch4_conv = BaseConv(in_channels, ch4, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 四个分支, 输出的空间尺寸都与输入相同
        out1 = self.branch1(x)

        reduced2 = self.branch2_reduce(x)
        out2 = self.branch2_conv(reduced2)

        reduced3 = self.branch3_reduce(x)
        out3 = self.branch3_conv(reduced3)

        pooled4 = self.branch4_pool(x)
        out4 = self.branch4_conv(pooled4)

        # 沿通道维度拼接
        return torch.cat([out1, out2, out3, out4], dim=1)
