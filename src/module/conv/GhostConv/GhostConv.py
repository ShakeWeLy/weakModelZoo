import math

import torch
import torch.nn as nn

from ..base import ActArg, BaseConv, NormArg


class GhostConv(nn.Module):
    """
    GhostConv (GhostNet, CVPR 2020)。

    思路: 先用少量普通卷积生成本征特征图，再用廉价的深度卷积生成 "幽灵" 特征图，
    最后与本征特征拼接得到 out_channels 个输出通道。

    Args:
        in_channels: 输入通道数
        out_channels: 输出通道数
        kernel_size: 主卷积核大小，默认 1
        stride: 步幅，默认 1（只作用于主卷积，廉价分支固定为 1）
        ratio: 压缩比，本征特征通道 = ceil(out_channels / ratio)，必须 >= 2，默认 2
        dw_kernel_size: 廉价深度卷积的核大小，默认 3
        act: 激活函数配置
        bn: 归一化配置
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        ratio: int = 2,
        dw_kernel_size: int = 3,
        act: ActArg = True,
        bn: NormArg = True,
    ) -> None:
        super().__init__()

        # [修复] ratio=1 时 cheap_num = 0, 卷积权重为空, PyTorch 只会报
        # "expected weight to be at least 1 at dimension 0" 这类难以理解的错误。
        # 这里提前校验, 并给出明确提示。
        if ratio < 2:
            raise ValueError("ratio 必须 >= 2")

        self.out_channels = out_channels
        # 本征通道数向上取整, 保证 identity_num * ratio >= out_channels。
        # 当 out_channels 不能被 ratio 整除时, 多出来的通道会在 forward 里被切掉。
        self.identity_num = math.ceil(out_channels / ratio)
        # 廉价分支输出通道 = 本征通道 * (ratio - 1), 与本征通道拼接后总数为 identity_num * ratio
        self.cheap_num = self.identity_num * (ratio - 1)

        # 主卷积: 负责下采样 (stride 在这里生效)
        self.conv = BaseConv(
            in_channels=in_channels,
            out_channels=self.identity_num,
            kernel_size=kernel_size,
            stride=stride,
            act=act,
            bn=bn,
        )

        # 廉价分支: 深度卷积 (groups = identity_num, 每个本征特征图各自生成 ratio-1 个幽灵特征图)。
        # [修复] stride 必须固定为 1。
        # 之前这里传的是外部的 stride, 但 self.conv 已经下采样过一次, 廉价分支再下采样
        # 会导致 stride=2 时 identity 与 cheap 的空间尺寸不一致, torch.cat 报错:
        # "Sizes of tensors must match except in dimension 1"。
        # 之前使用的 DSConv (深度 + 逐点) 也已替换为纯深度卷积, 与 GhostNet 原文一致。
        self.cheap_conv = BaseConv(
            in_channels=self.identity_num,
            out_channels=self.cheap_num,
            kernel_size=dw_kernel_size,
            stride=1,
            groups=self.identity_num,
            act=act,
            bn=bn,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_identity = self.conv(x)
        x_cheap = self.cheap_conv(x_identity)
        # [修复] out_channels 不能被 ratio 整除时, identity_num * ratio 会大于 out_channels。
        # 之前直接 cat 返回, 例如 out_channels=33, ratio=2 会得到 34 个通道, 与 Args 不符。
        # 这里切片截断到 out_channels。
        return torch.cat([x_identity, x_cheap], dim=1)[:, :self.out_channels]
