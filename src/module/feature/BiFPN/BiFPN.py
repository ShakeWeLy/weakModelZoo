from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...conv import DSConv
from ...conv.base import BaseConv


class WeightedFusion(nn.Module):
    """
    快速归一化融合 (Fast Normalized Fusion, EfficientDet 提出)。

    公式: O = sum_i( w_i * I_i ) / ( sum_j w_j + eps ),  其中 w_i = ReLU(w_i')

    与 softmax 加权相比, 只需要一次除法, 速度更快, 且 w_i 始终非负。

    Args:
        num_inputs: 参与融合的输入个数
        eps: 防止除零的小常数
    """

    def __init__(self, num_inputs: int, eps: float = 1e-4) -> None:
        super().__init__()
        # 每个输入一个可学习的标量权重, 初始值为 1 (等权重)
        self.weight = nn.Parameter(torch.ones(num_inputs))
        self.eps = eps

    def forward(self, inputs: list[torch.Tensor]) -> torch.Tensor:
        # 保证权重非负
        w = F.relu(self.weight)
        # 归一化: 所有权重之和为 1 (加 eps 防止除零)
        w = w / (w.sum() + self.eps)

        fused = inputs[0] * w[0]
        for i in range(1, len(inputs)):
            fused = fused + inputs[i] * w[i]
        return fused


class BiFPNLayer(nn.Module):
    """
    一层 BiFPN, 处理 3 个尺度 (features[0] 最浅, features[2] 最深), 输入输出通道数相同。

    自顶向下:
        td1  = Fuse(x1, 上采样(x2))
        out0 = Fuse(x0, 上采样(td1))
    自底向上:
        out1 = Fuse(x1, td1, 下采样(out0))
        out2 = Fuse(x2, 下采样(out1))

    每个 Fuse 都是 "加权融合 + 深度可分离卷积"。
    """

    def __init__(self, channels: int, eps: float) -> None:
        super().__init__()

        # 下采样: 3x3 最大池化, 步幅 2, 分辨率减半
        self.down = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 自顶向下节点
        self.fuse_td1 = WeightedFusion(num_inputs=2, eps=eps)
        self.conv_td1 = DSConv(channels, channels, kernel_size=3)

        self.fuse_out0 = WeightedFusion(num_inputs=2, eps=eps)
        self.conv_out0 = DSConv(channels, channels, kernel_size=3)

        # 自底向上节点 (中间层有 3 个输入: 原始输入 + 自顶向下结果 + 下采样结果)
        self.fuse_out1 = WeightedFusion(num_inputs=3, eps=eps)
        self.conv_out1 = DSConv(channels, channels, kernel_size=3)

        self.fuse_out2 = WeightedFusion(num_inputs=2, eps=eps)
        self.conv_out2 = DSConv(channels, channels, kernel_size=3)

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        x0 = features[0]
        x1 = features[1]
        x2 = features[2]

        # ---------- 自顶向下 ----------
        up2 = F.interpolate(x2, size=x1.shape[-2:], mode="nearest")
        td1 = self.conv_td1(self.fuse_td1([x1, up2]))

        up1 = F.interpolate(td1, size=x0.shape[-2:], mode="nearest")
        out0 = self.conv_out0(self.fuse_out0([x0, up1]))

        # ---------- 自底向上 ----------
        down0 = self.down(out0)
        out1 = self.conv_out1(self.fuse_out1([x1, td1, down0]))

        down1 = self.down(out1)
        out2 = self.conv_out2(self.fuse_out2([x2, down1]))

        return [out0, out1, out2]


class BiFPN(nn.Module):
    """
    BiFPN 双向特征金字塔网络 (EfficientDet, CVPR 2020)。

    固定处理 3 个尺度的特征, features[0] 最浅 (分辨率最高), features[2] 最深 (分辨率最低)。
    相邻两层的空间尺寸应为 2 倍关系 (例如 H/8, H/16, H/32)。

    相对于 PANet 的改进:
        1. 双向路径 (自顶向下 + 自底向上) 的每个节点都加入了可学习的加权融合, 让网络自己决定各输入的重要性;
        2. 每次融合之后的卷积都使用深度可分离卷积 (DSConv), 降低参数量与计算量;
        3. 同一层可以堆叠多次 (num_layers), 每层拥有独立的参数, 逐步加深融合。

    结构:
        1. 输入先经过 1x1 卷积统一通道数到 out_channels;
        2. 堆叠 num_layers 个 BiFPNLayer。

    Args:
        in_channels: 三个尺度输入的通道数, 由浅到深, 例如 [256, 512, 1024]
        out_channels: 输出的统一通道数
        num_layers: BiFPN 层的堆叠次数, 至少为 1, 默认 1
        eps: 快速归一化融合中的数值稳定项, 默认 1e-4
    """

    def __init__(
        self,
        in_channels: Sequence[int],
        out_channels: int,
        num_layers: int = 1,
        eps: float = 1e-4,
    ) -> None:
        super().__init__()

        if len(in_channels) != 3:
            raise ValueError("BiFPN 需要恰好 3 个输入尺度")
        if num_layers < 1:
            raise ValueError("num_layers 至少为 1")

        # 输入投影: 1x1 卷积统一通道数, 不使用激活函数
        self.proj0 = BaseConv(in_channels[0], out_channels, kernel_size=1, act=False)
        self.proj1 = BaseConv(in_channels[1], out_channels, kernel_size=1, act=False)
        self.proj2 = BaseConv(in_channels[2], out_channels, kernel_size=1, act=False)

        # 堆叠 BiFPN 层, 每层参数独立
        self.layers = nn.ModuleList(
            [BiFPNLayer(out_channels, eps=eps) for _ in range(num_layers)]
        )

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        # 统一通道数
        x0 = self.proj0(features[0])
        x1 = self.proj1(features[1])
        x2 = self.proj2(features[2])

        outputs = [x0, x1, x2]
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs
