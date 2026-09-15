import torch
import torch.nn as nn

from ...conv.base import ActArg


class CAM(nn.Module):
    """Channel Attention Module"""
    def __init__(self, in_channels: int, reduction_ratio: int = 16, act: ActArg = True,) -> None:
        super().__init__()

        if reduction_ratio <= 0:
            raise ValueError("reduction_ratio must be greater than 0")

        hidden_channels = max(in_channels // reduction_ratio, 1)

        if act is True:
            act = nn.ReLU()
        elif act is False:
            act = nn.Identity()
        elif not isinstance(act, nn.Module):
            raise TypeError("act must be a bool or nn.Module")

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # self.fc1 = nn.Linear(in_channels, in_channels // reduction_ratio)
        # self.fc2 = nn.Linear(in_channels // reduction_ratio, in_channels)
        # self.act1 = act
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            act,
            nn.Linear(hidden_channels, in_channels)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()

        x_avg = self.avg_pool(x).view(b, c)
        x_max = self.max_pool(x).view(b, c)

        # x_avg = self.fc2(self.act1(self.fc1(x_avg)))
        # x_max = self.fc2(self.act1(self.fc1(x_max)))

        attention = self.mlp(x_avg) + self.mlp(x_max)
        attention = self.sigmoid(attention).view(b, c, 1, 1)
        return attention * x


'''
之前有几个明显问题：

out_channels 没有使用
构造函数接收了 out_channels，但内部始终生成 in_channels 个通道权重。建议删除该参数，或真正用于输出通道变换。

最大池化分支维度错误

x_avg = self.fc1(...)
x_max = self.fc2(...)
fc2 要求输入维度是 in_channels // reduction_ratio，但 x_max 的维度仍是 in_channels。当 reduction_ratio != 1 时会报矩阵乘法维度错误。

正确方式应是两个分支都经过共享的 fc1 和 fc2：

x_avg = self.fc2(self.fc1(x_avg))
x_max = self.fc2(self.fc1(x_max))
缺少非线性激活
fc1 和 fc2 之间通常使用 ReLU、SiLU 等激活函数，否则两层线性层可以合并为一个线性变换。

缺少 sigmoid
通道注意力通常需要：

attention = torch.sigmoid(...)
将权重限制到 [0, 1]。

forward() 返回的不是加权特征
当前返回的是形状为 (B, C, 1, 1) 的注意力权重，而不是经过注意力调制后的特征。通常应返回：
return x_input * attention
隐藏通道可能为 0
in_channels // reduction_ratio
当 in_channels < reduction_ratio 时结果为 0，会导致无效的 Linear 层。应使用：

hidden_channels = max(1, in_channels // reduction_ratio)
'''