from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class ASFF(nn.Module):
    """
    ASFF 自适应空间特征融合 (Learning Spatial Fusion for Single-Shot Object Detection, arXiv 2019, 1911.09516)。

    固定处理 3 个尺度的特征 (与原论文一致), features[0] 分辨率最高, features[2] 分辨率最低。

    步骤:
        1. 把三个尺度都统一到目标层 level 的分辨率和 out_channels 通道:
           - 分辨率比目标高的 (i < level): 用 kernel = stride = 2**(level - i) 的卷积下采样
           - 分辨率相同的 (i == level): 1x1 卷积
           - 分辨率比目标低的 (i > level): 1x1 卷积, forward 中再上采样
        2. 三个尺度拼接后, 用 1x1 卷积预测 3 张空间权重图, 经 softmax 归一化
           (每个像素上三个权重之和为 1)
        3. 三个尺度按权重逐像素加权求和, 得到 level 层的输出。

    Args:
        level: 目标输出层级, 只能是 0、1、2 (0 分辨率最高, 2 分辨率最低)
        in_channels: 三个尺度输入的通道数, 例如 [256, 512, 1024]
        out_channels: 融合后的输出通道数
    """

    def __init__(self, level: int, in_channels: Sequence[int], out_channels: int) -> None:
        super().__init__()

        if level not in (0, 1, 2):
            raise ValueError("level 只能是 0、1 或 2")
        if len(in_channels) != 3:
            raise ValueError("ASFF 需要恰好 3 个输入尺度")

        self.level = level

        # 每个输入需要的下采样倍数: 比目标层分辨率高多少倍, 就用多大的 kernel 和 stride
        # 分辨率不高于目标层的输入, 倍数为 1 (即 1x1 卷积, 不改变尺寸)
        scale0 = 2 ** max(level - 0, 0)
        scale1 = 2 ** max(level - 1, 0)
        scale2 = 2 ** max(level - 2, 0)

        self.resize0 = nn.Sequential(
            nn.Conv2d(in_channels[0], out_channels, kernel_size=scale0, stride=scale0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.resize1 = nn.Sequential(
            nn.Conv2d(in_channels[1], out_channels, kernel_size=scale1, stride=scale1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.resize2 = nn.Sequential(
            nn.Conv2d(in_channels[2], out_channels, kernel_size=scale2, stride=scale2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        # 权重预测: 输入是三个尺度拼接后的 3 * out_channels 通道, 输出 3 张权重图 (每个尺度一张)
        self.weight_conv = nn.Conv2d(out_channels * 3, 3, kernel_size=1)

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        feat0 = features[0]
        feat1 = features[1]
        feat2 = features[2]

        # 目标层的空间尺寸, 三个尺度都要对齐到这个尺寸
        target_size = features[self.level].shape[-2:]

        x0 = self.resize0(feat0)
        x1 = self.resize1(feat1)
        x2 = self.resize2(feat2)

        # 上采样到目标尺寸 (如果已经是目标尺寸, 这一步不改变结果)
        x0 = F.interpolate(x0, size=target_size, mode="bilinear", align_corners=False)
        x1 = F.interpolate(x1, size=target_size, mode="bilinear", align_corners=False)
        x2 = F.interpolate(x2, size=target_size, mode="bilinear", align_corners=False)

        # 预测空间权重, 在三个尺度之间做 softmax, 每个像素的三个权重之和为 1
        concat = torch.cat([x0, x1, x2], dim=1)
        weights = torch.softmax(self.weight_conv(concat), dim=1)
        w0 = weights[:, 0:1]
        w1 = weights[:, 1:2]
        w2 = weights[:, 2:3]

        # 逐像素加权求和
        return x0 * w0 + x1 * w1 + x2 * w2
