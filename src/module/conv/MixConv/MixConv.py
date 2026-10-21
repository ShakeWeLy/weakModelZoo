import torch
import torch.nn as nn

from ..base import ActArg, NormArg, BaseConv


class MixConv(nn.Module):
    """
    MixConv (MixNet, BMVC 2019) 占位实现。

    思路: 将输入通道分成若干组，每组使用不同大小的深度卷积核（如 3x3、5x5、7x7），
    再拼接输出，以较低参数量获得多尺度感受野。

    输出通道数与输入相同 (in_channels)。如需改变通道数, 请在本模块外部接 1x1 卷积 (与原论文一致)。

    Args:
        in_channels: 输入通道数，必须能被 kernel_sizes 的长度整除
        kernel_sizes: 各分组使用的卷积核大小，默认 (3, 5, 7)
        stride: 步幅，默认 1
        act: 激活函数配置
        bn: 归一化配置
    """

    def __init__(
        self,
        in_channels: int,
        kernel_sizes: tuple[int, ...] = (3, 5, 7),
        stride: int = 1,
        act: ActArg = True,
        bn: NormArg = True,
    ) -> None:
        super().__init__()

        # [修改] 去掉了 out_channels 参数。之前该参数从未生效 (输出通道数始终等于 in_channels),
        # 容易让使用者误以为可以改变通道数。现在与原论文 MixConv 保持一致: 输出通道 = 输入通道,
        # 通道数的变换交给外部的 1x1 卷积。
        #
        # [修复] 判断条件写反。之前是 `in_channels // len(kernel_sizes) != 0`,
        # 对所有 in_channels >= 组数的合法输入都会误报错 (例如 in=18, 组数=3 时 18 // 3 = 6 != 0)。
        # 正确的判断应使用取余 `%`: 只有不能整除时才报错。
        num_groups = len(kernel_sizes)
        if in_channels % num_groups != 0:
            raise ValueError("in_channels 必须能被 kernel_sizes 的长度整除")

        self.in_channels_groups = in_channels // num_groups
        self.kernel_sizes = kernel_sizes
        self.stride = stride
        self.act = act
        self.bn = bn

        # [修复] 之前使用普通 Python list 保存各组卷积层, nn.Module 不会注册 list 中的子模块,
        # 导致这些卷积的参数不参与训练、不会随 .to(device) 移动、也不会出现在 state_dict 中。
        # 改为 nn.ModuleList 才能正确注册。
        #
        # [修复] 之前没有传 groups, BaseConv 默认 groups=1, 实际是标准卷积而不是文档中说的深度卷积。
        # 现在 groups = in_channels_groups, 每组各自做深度卷积 (in == out == 组内通道数)。
        self.conv_groups = nn.ModuleList(
            [
                BaseConv(
                    in_channels=self.in_channels_groups,
                    out_channels=self.in_channels_groups,
                    kernel_size=kernel_size,
                    stride=self.stride,
                    groups=self.in_channels_groups,
                    act=self.act,
                    bn=self.bn,
                )
                for kernel_size in self.kernel_sizes
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 沿通道维切分为若干组, 每组送入对应的深度卷积, 最后在通道维拼接。
        x_groups = torch.split(x, self.in_channels_groups, dim=1)
        x_groups = [conv(x_group) for conv, x_group in zip(self.conv_groups, x_groups)]
        return torch.cat(x_groups, dim=1)
