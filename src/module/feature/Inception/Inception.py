import torch
import torch.nn as nn

from src.module.conv import BaseConv


class InceptionV1(nn.Module):
    '''
    Inception 模块 (GoogLeNet, CVPR 2015, 即 Inception v1) 。

    思路: 并行使用 4 个分支提取多尺度特征, 最后在通道维度拼接,
    大卷积核使用小的channel以降低计算量, 
    让网络自己选择合适的卷积核大小。

    四个分支:
        branch1: 1x1 卷积
        branch2: 1x1 降维 -> 3x3 卷积
        branch3: 1x1 降维 -> 5x5 卷积
        branch4: 3x3 最大池化 (步幅 1) -> 1x1 卷积

    其中 1x1 降维层用于减少 3x3 / 5x5 卷积的输入通道数, 降低计算量 (这是 Inception 的关键设计)。
    '''
    def __init__(self, in_channels: int, out_channels_list: list[int] | None = None) -> None:
        super().__init__()
        if out_channels_list is None:
            out_channels_list = [64, 128, 32, 32]
        reduce_channel = max(in_channels // 4, 1)

        if len(out_channels_list) != 4:
            raise ValueError("out_channels_list 长度必须为 4")

        self.branch1 = BaseConv(in_channels, out_channels_list[0], kernel_size=1)
        self.branch2 = nn.Sequential(
            BaseConv(in_channels, reduce_channel, kernel_size=1),
            BaseConv(reduce_channel, out_channels_list[1], kernel_size=3),
        )
        self.branch3 = nn.Sequential(
            BaseConv(in_channels, reduce_channel, kernel_size=1),
            BaseConv(reduce_channel, out_channels_list[2], kernel_size=5),
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            BaseConv(in_channels, out_channels_list[3], kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], dim=1)


class InceptionV2(nn.Module):
    '''
    Inception 模块 (GoogLeNet, CVPR 2015, 即 Inception v2) 。

    思路: 用两个3*3的卷积核代替5*5的大卷积核, 减少参数数量, 提高计算效率。
    '''
    def __init__(self, in_channels: int, out_channels_list: list[int] | None = None) -> None:
        super().__init__()
        if out_channels_list is None:
            out_channels_list = [64, 128, 32, 32]
        reduce_channel = max(in_channels // 4, 1)

        if len(out_channels_list) != 4:
            raise ValueError("out_channels_list 长度必须为 4")

        self.branch1 = BaseConv(in_channels, out_channels_list[0], kernel_size=1)
        self.branch2 = nn.Sequential(
            BaseConv(in_channels, reduce_channel, kernel_size=1),
            BaseConv(reduce_channel, out_channels_list[1], kernel_size=3),
        )
        self.branch3 = nn.Sequential(
            BaseConv(in_channels, reduce_channel, kernel_size=1),
            BaseConv(reduce_channel, out_channels_list[2], kernel_size=3),
            BaseConv(out_channels_list[2], out_channels_list[2], kernel_size=3),
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            BaseConv(in_channels, out_channels_list[3], kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], dim=1)


class InceptionV2_B(nn.Module):
    '''
    Inception 模块 (GoogLeNet, CVPR 2015, 即 Inception v2) 。

    思路: 一个 3×3 的卷积等价于首先执行一个 1×3 的卷积再执行一个 3×1 的卷积。
    这样同样可以只使用约（1x3 + 3x1) / (3x3) = 67%的计算开销
    '''
    def __init__(self, in_channels: int, out_channels_list: list[int] | None = None) -> None:
        super().__init__()
        if out_channels_list is None:
            out_channels_list = [64, 128, 32, 32]
        reduce_channel = max(in_channels // 4, 1)

        if len(out_channels_list) != 4:
            raise ValueError("out_channels_list 长度必须为 4")

        self.branch1 = BaseConv(in_channels, out_channels_list[0], kernel_size=1)
        # padding 交给 BaseConv：对 (3,1)/(1,3) 自动得到 (1,0)/(0,1)，保持空间尺寸
        self.branch2 = nn.Sequential(
            BaseConv(in_channels, reduce_channel, kernel_size=1),
            BaseConv(reduce_channel, out_channels_list[1], kernel_size=(3, 1)),
            BaseConv(out_channels_list[1], out_channels_list[1], kernel_size=(1, 3)),
        )
        self.branch3 = nn.Sequential(
            BaseConv(in_channels, reduce_channel, kernel_size=1),
            BaseConv(reduce_channel, reduce_channel, kernel_size=(3, 1)),
            BaseConv(reduce_channel, out_channels_list[2], kernel_size=(1, 3)),
            BaseConv(out_channels_list[2], out_channels_list[2], kernel_size=(3, 1)),
            BaseConv(out_channels_list[2], out_channels_list[2], kernel_size=(1, 3)),
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            BaseConv(in_channels, out_channels_list[3], kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], dim=1)


class InceptionV3(nn.Module):
    """
    Inception-v3：针对 Inception-v2 的升级，增加了以下内容：
    (1). RMSProp 优化器。
    (2). 分解为 7x7 卷积。
    (3). 辅助分类 BatchNorm。
    (4). 标签平滑 (Label Smoothing)。
    """
    pass


class InceptionV4(nn.Module):
    """Inception-v4 占位，待实现。"""
    pass


# 兼容旧名：原先单类 Inception 对应 v1
Inception = InceptionV1
