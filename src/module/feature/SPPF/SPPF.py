import torch
import torch.nn as nn

from ...conv.base import BaseConv


class SPPF(nn.Module):
    """
    SPPF (Spatial Pyramid Pooling - Fast, YOLOv5, 2020) 快速空间金字塔池化。

    与 SPP 的区别: SPP 并行使用 5、9、13 三种大核池化, SPPF 只使用一个 5x5 池化,
    并把它串联三次, 得到的结果与 SPP(pool_sizes=(5, 9, 13)) 完全相同, 计算量更低。

    原理: 对同一个最大池化连续做两次 5x5 (步幅 1) 等价于一次 9x9, 三次等价于一次 13x13。

    步骤:
        1. cv1: 1x1 卷积, 通道数降为 in_channels // 2
        2. y1 = pool(x), y2 = pool(y1), y3 = pool(y2)
        3. 拼接 [x, y1, y2, y3], 再用 cv2 (1x1 卷积) 投影到 out_channels

    Args:
        in_channels: 输入通道数, 至少为 2
        out_channels: 输出通道数
        kernel_size: 池化核大小, 必须是正奇数, 默认 5
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 5) -> None:
        super().__init__()

        if in_channels < 2:
            raise ValueError("in_channels 至少为 2")
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("kernel_size 必须是正奇数, 才能保持空间尺寸不变")

        hidden_channels = in_channels // 2

        # cv1: 1x1 卷积降维
        self.cv1 = BaseConv(in_channels, hidden_channels, kernel_size=1)

        # 同一个池化层会被串联调用三次, 池化没有可学习参数, 因此可以安全复用
        self.pool = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

        # cv2: 拼接 4 份特征 (x, y1, y2, y3), 通道数为 4 * hidden_channels
        self.cv2 = BaseConv(hidden_channels * 4, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 降维
        x = self.cv1(x)

        # 串联池化: 每次都在上一次的结果上继续池化, 感受野依次为 k, 2k-1, 3k-2
        y1 = self.pool(x)
        y2 = self.pool(y1)
        y3 = self.pool(y2)

        # 拼接并投影
        out = torch.cat([x, y1, y2, y3], dim=1)
        return self.cv2(out)
