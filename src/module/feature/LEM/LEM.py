import torch
import torch.nn as nn
from torch.nn import functional as F

from ...conv.base import BaseConv

class LEM(nn.Module):
    def __init__(self, in_channels,feature_channels, reduction_ratio: int = 16):
        super(LEM, self).__init__()
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

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.linear_1 = nn.Linear(self.in_channels, self.feature_channels)
        self.linear_3 = nn.Linear(self.feature_channels, self.in_channels)
        self.linear_5 = nn.Linear(self.feature_channels, self.in_channels)
        self.linear_7 = nn.Linear(self.feature_channels, self.in_channels)
        self.norm_1 = nn.BatchNorm2d(self.in_channels)



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, C, H, W] -> [B, C, W]
        f_x = x.mean(dim=3, keepdim=False)
        # [B, C, H, W] -> [B, C, H]
        f_y = x.mean(dim=2, keepdim=False)
        f_xy = torch.cat([f_x, f_y], dim=-1) #  # dim不是1, 因为我不是C->2C ,而是H,W->(H+W)
        # [B, C//r, (H+W)] -> 
        f_x, f_y = f_xy.split(self.conv(f_xy), dim=1) 
        x_x = x * f_x
        x_y = x * f_y

        f_1 = torch.add(self.conv_x_1(x_x), self.conv_y_1(x_y))
        f_3 = torch.add(self.conv_x_3(x_x), self.conv_y_3(x_y))
        f_5 = torch.add(self.conv_x_5(x_x), self.conv_y_5(x_y))
        f_7 = torch.add(self.conv_x_7(x_x), self.conv_y_7(x_y))
        x_all = torch.add(f_1, f_3, f_5, f_7)

        a = self.maxpool(x_all)
        a_1 = F.relu(self.linear_1(a))
        a_3 = F.relu(self.linear_3(a))
        a_5 = F.relu(self.linear_5(a))
        a_7 = F.relu(self.linear_7(a))
        a = torch.cat([a_1, a_3, a_5, a_7], dim=1)
        a = self.norm_1(a)
        a_1, a_3, a_5, a_7 = a.split(a, dim=1)

        x = torch.add(f_1*a_1, f_3*a_3, f_5*a_5, f_7*a_7)
        return x

