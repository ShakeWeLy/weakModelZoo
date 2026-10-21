from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class FPN(nn.Module):
    """
    FPN 特征金字塔网络 (CVPR 2017)。

    固定处理 3 个尺度的特征, features[0] 最浅 (分辨率最高), features[2] 最深 (分辨率最低)。

    步骤:
        1. 横向连接: 每一层用 1x1 卷积把通道数统一为 out_channels
        2. 自顶向下: 深层特征上采样到浅层尺寸, 与浅层的横向特征逐元素相加
        3. 平滑: 每一层再用 3x3 卷积, 减轻上采样和相加带来的混叠效应

    Args:
        in_channels: 三个尺度的输入通道数, 由浅到深, 例如 [256, 512, 1024]
        out_channels: 金字塔每一层的统一输出通道数
    """

    def __init__(self, in_channels: Sequence[int], out_channels: int) -> None:
        super().__init__()

        if len(in_channels) != 3:
            raise ValueError("FPN 需要恰好 3 个输入尺度")

        # 横向连接: 1x1 卷积, 只改变通道数, 不改变尺寸
        self.lateral0 = nn.Conv2d(in_channels[0], out_channels, kernel_size=1)
        self.lateral1 = nn.Conv2d(in_channels[1], out_channels, kernel_size=1)
        self.lateral2 = nn.Conv2d(in_channels[2], out_channels, kernel_size=1)

        # 平滑: 3x3 卷积, 作用在自顶向下路径的每一层上, 尺寸不变
        self.smooth0 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.smooth1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.smooth2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        feat0 = features[0]
        feat1 = features[1]
        feat2 = features[2]

        # 横向连接
        lateral0 = self.lateral0(feat0)
        lateral1 = self.lateral1(feat1)
        lateral2 = self.lateral2(feat2)

        # 自顶向下: 最深层作为起点, 逐级上采样并与浅层相加
        top2 = lateral2
        top1 = lateral1 + F.interpolate(top2, size=lateral1.shape[-2:], mode="nearest")
        top0 = lateral0 + F.interpolate(top1, size=lateral0.shape[-2:], mode="nearest")

        # 平滑后输出, 顺序与输入一致 (由浅到深)
        out0 = self.smooth0(top0)
        out1 = self.smooth1(top1)
        out2 = self.smooth2(top2)
        return [out0, out1, out2]
