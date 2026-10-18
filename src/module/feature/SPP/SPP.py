import torch
import torch.nn as nn

from ...conv.base import BaseConv


class SPP(nn.Module):
    """
    SPP 空间金字塔池化 (SPPNet, ECCV 2014 / TPAMI 2015)。

    本实现采用 YOLOv3-SPP 的写法: 池化步幅为 1、padding = k // 2, 因此空间尺寸保持不变,
    可以直接放在检测网络的特征图之后。

    步骤:
        1. cv1: 1x1 卷积, 通道数降为 in_channels // 2, 减少后续池化的计算量
        2. 对 cv1 的输出分别做 k x k 最大池化 (k 取自 pool_sizes), 感受野逐个变大
        3. 把原始特征与各池化结果在通道维度拼接, 再用 cv2 (1x1 卷积) 投影到 out_channels

    注意: 原版 SPPNet 使用自适应池化把特征图划分成 1x1、2x2、4x4 等固定网格, 输出固定长度向量,
    主要用于分类。本实现属于 YOLO 系列的检测用变体, 输出仍是特征图。

    Args:
        in_channels: 输入通道数, 至少为 2
        out_channels: 输出通道数
        pool_sizes: 各最大池化核大小, 必须都是正奇数, 默认 (5, 9, 13)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        pool_sizes: tuple[int, ...] = (5, 9, 13),
    ) -> None:
        super().__init__()

        if in_channels < 2:
            raise ValueError("in_channels 至少为 2")
        if any(k < 1 or k % 2 == 0 for k in pool_sizes):
            raise ValueError("pool_sizes 中的每个值都必须是正奇数, 才能保持空间尺寸不变")

        hidden_channels = in_channels // 2

        # cv1: 1x1 卷积降维
        self.cv1 = BaseConv(in_channels, hidden_channels, kernel_size=1)

        # 多个最大池化, 步幅为 1 且 padding = k // 2, 输出尺寸与输入相同
        # 这里使用 ModuleList, 因为 pool_sizes 的长度由参数决定
        self.pools = nn.ModuleList(
            [nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2) for k in pool_sizes]
        )

        # cv2: 拼接后的通道数 = 原始特征 (1 份) + 各池化结果 (len(pool_sizes) 份)
        concat_channels = hidden_channels * (1 + len(pool_sizes))
        self.cv2 = BaseConv(concat_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 降维
        x = self.cv1(x)

        # 各尺度池化结果, 顺序与 pool_sizes 一致
        pooled_outputs = [pool(x) for pool in self.pools]

        # 原始特征 + 所有池化结果 沿通道拼接, 然后投影
        out = torch.cat([x] + pooled_outputs, dim=1)
        return self.cv2(out)
