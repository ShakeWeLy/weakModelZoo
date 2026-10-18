from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...conv.base import BaseConv


class PANet(nn.Module):
    """
    PANet 路径聚合网络 (CVPR 2018)。

    固定处理 3 个尺度的特征, features[0] 最浅 (分辨率最高), features[2] 最深 (分辨率最低)。
    相邻两层的空间尺寸应为 2 倍关系 (例如 H/8, H/16, H/32)。

    在 FPN 的基础上增加了一条自底向上的路径, 把浅层的定位信息传递给深层:

        1. 自顶向下 (与 FPN 相同): 横向 1x1 卷积 + 上采样相加 + 3x3 平滑, 得到 P0, P1, P2
        2. 自底向上:
           - N0 = P0 (最浅层直接使用)
           - N1 = Fuse(下采样(N0) + P1)  下采样用 3x3 步幅 2 卷积
           - N2 = Fuse(下采样(N1) + P2)
           其中 Fuse 为 3x3 卷积。

    Args:
        in_channels: 三个尺度输入的通道数, 由浅到深, 例如 [256, 512, 1024]
        out_channels: 四条路径输出的统一通道数
    """

    def __init__(self, in_channels: Sequence[int], out_channels: int) -> None:
        super().__init__()

        if len(in_channels) != 3:
            raise ValueError("PANet 需要恰好 3 个输入尺度")

        # ---------- 自顶向下路径 (与 FPN 相同) ----------
        # 横向连接: 1x1 卷积, 统一通道数
        self.lateral0 = BaseConv(in_channels[0], out_channels, kernel_size=1)
        self.lateral1 = BaseConv(in_channels[1], out_channels, kernel_size=1)
        self.lateral2 = BaseConv(in_channels[2], out_channels, kernel_size=1)

        # 自顶向下平滑: 3x3 卷积, 尺寸不变
        self.td_smooth0 = BaseConv(out_channels, out_channels, kernel_size=3)
        self.td_smooth1 = BaseConv(out_channels, out_channels, kernel_size=3)
        self.td_smooth2 = BaseConv(out_channels, out_channels, kernel_size=3)

        # ---------- 自底向上路径 ----------
        # 下采样: 3x3 卷积, 步幅 2, 分辨率减半
        self.bu_down0 = BaseConv(out_channels, out_channels, kernel_size=3, stride=2)
        self.bu_down1 = BaseConv(out_channels, out_channels, kernel_size=3, stride=2)

        # 融合后的平滑: 3x3 卷积, 尺寸不变
        self.bu_fuse1 = BaseConv(out_channels, out_channels, kernel_size=3)
        self.bu_fuse2 = BaseConv(out_channels, out_channels, kernel_size=3)

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        feat0 = features[0]
        feat1 = features[1]
        feat2 = features[2]

        # ---------- 自顶向下 ----------
        lateral0 = self.lateral0(feat0)
        lateral1 = self.lateral1(feat1)
        lateral2 = self.lateral2(feat2)

        top2 = lateral2
        top1 = lateral1 + F.interpolate(top2, size=lateral1.shape[-2:], mode="nearest")
        top0 = lateral0 + F.interpolate(top1, size=lateral0.shape[-2:], mode="nearest")

        p0 = self.td_smooth0(top0)
        p1 = self.td_smooth1(top1)
        p2 = self.td_smooth2(top2)

        # ---------- 自底向上 ----------
        # 最浅层直接作为起点
        n0 = p0

        # 第 1 层: 把浅层 N0 下采样后, 与 P1 相加, 再融合
        down0 = self.bu_down0(n0)
        n1 = self.bu_fuse1(down0 + p1)

        # 第 2 层: 把 N1 下采样后, 与 P2 相加, 再融合
        down1 = self.bu_down1(n1)
        n2 = self.bu_fuse2(down1 + p2)

        return [n0, n1, n2]
