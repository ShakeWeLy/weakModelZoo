import torch
import torch.nn as nn
from torch.nn import functional as F

from ...conv.base import BaseConv

class LEM(nn.Module):
    def __init__(self, in_channels,feature_channels, reduction_ratio: int = 16):
        super(LEM, self).__init__()
        self.in_channels = in_channels
        self.feature_channels = feature_channels
        self.out_channels = in_channels // reduction_ratio
        self.conv = BaseConv(in_channels, self.out_channels, kernel_size=1)
        self.conv_x = BaseConv(self.out_channels, self.in_channels, kernel_size=1)
        self.conv_y = BaseConv(self.out_channels, self.in_channels, kernel_size=1)

        self.conv_x_1 = BaseConv(self.in_channels, self.in_channels, kernel_size=1)
        self.conv_y_1 = BaseConv(self.in_channels, self.in_channels, kernel_size=1)
        self.conv_x_3 = BaseConv(self.in_channels, self.in_channels, kernel_size=3, stride=1, padding=1)
        self.conv_y_3 = BaseConv(self.in_channels, self.in_channels, kernel_size=3, stride=1, padding=1)
        self.conv_x_5 = BaseConv(self.in_channels, self.in_channels, kernel_size=5, stride=1, padding=2)
        self.conv_y_5 = BaseConv(self.in_channels, self.in_channels, kernel_size=5, stride=1, padding=2)
        self.conv_x_7 = BaseConv(self.in_channels, self.in_channels, kernel_size=7, stride=1, padding=3)
        self.conv_y_7 = BaseConv(self.in_channels, self.in_channels, kernel_size=7, stride=1, padding=3)

        # [修复] 之前写的是 self.maxpool = AvgPool2d (名字与实际不符, 且 forward 中从未使用),
        # forward 中需要的全局平均池化是 self.avgpool, 之前未定义导致报错。
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        # [修复] 之前输入维度用的是 feature_channels, 但 forward 中送入的是 [B, C] 的 alpha (C = in_channels),
        # 维度对不上, 所以这里改为 in_channels。feature_channels 参数目前未被使用, 仅为保持接口不变。
        self.linear_1 = nn.Linear(self.in_channels, self.in_channels)
        self.linear_3 = nn.Linear(self.in_channels, self.in_channels)
        self.linear_5 = nn.Linear(self.in_channels, self.in_channels)
        self.linear_7 = nn.Linear(self.in_channels, self.in_channels)
        # [修复] 之前使用 BatchNorm2d, 但输入是 [B, 4C] 的二维张量, 需要 BatchNorm1d (通道数为 4C)。
        self.norm_1 = nn.BatchNorm1d(4 * self.in_channels)



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        # [B, C, H, W] -> [B, C, H, 1]
        f_x = x.mean(dim=3, keepdim=True)
        # [B, C, H, W] -> [B, C, 1, W]
        f_y = x.mean(dim=2, keepdim=True)
        f_y = f_y.permute(0, 1 ,3, 2) # [B, C, 1, W] -> [B, C, W, 1]
        # [B, C//r, (H+W), 1]
        f_xy = torch.cat([f_x, f_y], dim=2)
        # [修复] 之前写成 f_xy.split(self.conv(f_xy), [H, W], dim=2), 把卷积结果当成 split 的第一个参数, 报错。
        # 正确做法: 先卷积得到 [B, C//r, H+W, 1], 再沿 dim=2 按 [H, W] 切分。
        f_xy = self.conv(f_xy)
        f_x, f_y = f_xy.split([H, W], dim=2)
        f_y = f_y.permute(0, 1 ,3, 2) # [B, C, W, 1] -> [B, C, 1, W]
        x_x = x * self.conv_x(f_x)  # [B, C, H, W] * [B, C//r, H, 1] -> [B, C, H, W]
        x_y = x * self.conv_y(f_y)  # [B, C, H, W] * [B, C//r, 1, W] -> [B, C, H, W]

        f_1 = torch.add(self.conv_x_1(x_x), self.conv_y_1(x_y))
        f_3 = torch.add(self.conv_x_3(x_x), self.conv_y_3(x_y))
        f_5 = torch.add(self.conv_x_5(x_x), self.conv_y_5(x_y))
        f_7 = torch.add(self.conv_x_7(x_x), self.conv_y_7(x_y))
        # x_all = f_1 + f_3 + f_5 + f_7
        x_all = torch.stack([f_1, f_3, f_5, f_7], dim=0).sum(dim=0)

        alpha = self.avgpool(x_all)  # [B, C, H, W] -> [B, C, 1, 1]
        alpha = alpha.view(B, -1) # [B, C]
        alpha_1 = F.relu(self.linear_1(alpha))
        alpha_3 = F.relu(self.linear_3(alpha))
        alpha_5 = F.relu(self.linear_5(alpha))
        alpha_7 = F.relu(self.linear_7(alpha))
        alpha = torch.cat([alpha_1, alpha_3, alpha_5, alpha_7], dim=1)  #[B, 4C]
        alpha = self.norm_1(alpha)
        alpha_1, alpha_3, alpha_5, alpha_7 = alpha.chunk(4, dim=1)  #[B, 4C] -> [B, C]
        alpha_1 = alpha_1.view(B, C, 1, 1)
        alpha_3 = alpha_3.view(B, C, 1, 1)
        alpha_5 = alpha_5.view(B, C, 1, 1)
        alpha_7 = alpha_7.view(B, C, 1, 1)

        x = torch.stack([f_1*F.sigmoid(alpha_1), f_3*F.sigmoid(alpha_3), f_5*F.sigmoid(alpha_5), f_7*F.sigmoid(alpha_7)], dim=0).sum(dim=0)
        return x
