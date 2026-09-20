"""
Dynamic Snake Convolution (DSConv), ICCV 2023 DSCNet.

沿 x 轴蛇形卷积 (morph=0): 卷积核在水平方向展开，y 方向偏移逐点累加。
沿 y 轴蛇形卷积 (morph=1): 卷积核在垂直方向展开，x 方向偏移逐点累加。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _cumulative_offset(offset: torch.Tensor, center: int) -> torch.Tensor:
    """中心点不动，两侧偏移从前一点累加，保证蛇形连续。"""
    offset = offset.permute(1, 0, 2, 3).contiguous()
    cumulative = offset.detach().clone()
    cumulative[center] = 0.0
    for index in range(1, center + 1):
        cumulative[center + index] = cumulative[center + index - 1] + offset[center + index]
        cumulative[center - index] = cumulative[center - index + 1] + offset[center - index]
    return cumulative.permute(1, 0, 2, 3)


def _normalize_grid(coords_x: torch.Tensor, coords_y: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """将像素坐标转为 grid_sample 所需的 [-1, 1] 坐标。"""
    grid_x = 2.0 * coords_x / max(width - 1, 1) - 1.0
    grid_y = 2.0 * coords_y / max(height - 1, 1) - 1.0
    return torch.stack([grid_x, grid_y], dim=-1)


class _DynamicSnakeSampler(nn.Module):
    def __init__(self, kernel_size: int, morph: int, extend_scope: float = 1.0) -> None:
        super().__init__()
        if morph not in (0, 1):
            raise ValueError("morph 必须为 0（沿 x 轴）或 1（沿 y 轴）")
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size 建议为奇数，例如 3、5、7")
        self.kernel_size = kernel_size
        self.morph = morph
        self.extend_scope = extend_scope

    def forward(self, x: torch.Tensor, offset: torch.Tensor, if_offset: bool = True) -> torch.Tensor:
        batch, _, height, width = x.shape
        device = x.device
        num_points = self.kernel_size
        center = num_points // 2

        y_offset, x_offset = torch.split(offset, num_points, dim=1)
        if self.morph == 0:
            if if_offset:
                y_offset = _cumulative_offset(y_offset, center)
            base_y = torch.arange(height, device=device, dtype=x.dtype).view(1, 1, height, 1)
            base_x = torch.arange(width, device=device, dtype=x.dtype).view(1, 1, 1, width)
            delta_x = torch.arange(-center, center + 1, device=device, dtype=x.dtype).view(1, num_points, 1, 1)

            coords_y = base_y.expand(batch, num_points, height, width)
            coords_x = base_x.expand(batch, num_points, height, width) + delta_x
            if if_offset:
                coords_y = coords_y + y_offset * self.extend_scope

            coords_y = coords_y.reshape(batch, num_points * height, width)
            coords_x = coords_x.reshape(batch, num_points * height, width)
            grid = _normalize_grid(coords_x, coords_y, height, width)
            return F.grid_sample(
                x,
                grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            )

        if if_offset:
            x_offset = _cumulative_offset(x_offset, center)
        base_y = torch.arange(height, device=device, dtype=x.dtype).view(1, 1, height, 1)
        base_x = torch.arange(width, device=device, dtype=x.dtype).view(1, 1, 1, width)
        delta_y = torch.arange(-center, center + 1, device=device, dtype=x.dtype).view(1, num_points, 1, 1)

        coords_y = base_y.expand(batch, num_points, height, width) + delta_y
        coords_x = base_x.expand(batch, num_points, height, width)
        if if_offset:
            coords_x = coords_x + x_offset * self.extend_scope

        coords_y = coords_y.reshape(batch, height, num_points * width)
        coords_x = coords_x.reshape(batch, height, num_points * width)
        grid = _normalize_grid(coords_x, coords_y, height, width)
        return F.grid_sample(
            x,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )


class SnakeConv(nn.Module):
    """
    动态蛇形卷积。

    Args:
        in_channels: 输入通道数
        out_channels: 输出通道数
        kernel_size: 蛇形卷积核长度，默认 3
        morph: 0 表示沿 x 轴，1 表示沿 y 轴
        extend_scope: 偏移幅度缩放
        if_offset: 是否启用可学习蛇形偏移
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        morph: int = 0,
        extend_scope: float = 1.0,
        if_offset: bool = True,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.morph = morph
        self.if_offset = if_offset

        self.offset_conv = nn.Conv2d(in_channels, 2 * kernel_size, kernel_size=3, padding=1)
        self.offset_bn = nn.BatchNorm2d(2 * kernel_size)
        self.sampler = _DynamicSnakeSampler(kernel_size=kernel_size, morph=morph, extend_scope=extend_scope)

        if morph == 0:
            self.snake_conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=(kernel_size, 1),
                stride=(kernel_size, 1),
                padding=0,
            )
        else:
            self.snake_conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=(1, kernel_size),
                stride=(1, kernel_size),
                padding=0,
            )

        groups = max(1, out_channels // 4)
        while out_channels % groups != 0:
            groups -= 1
        self.norm = nn.GroupNorm(groups, out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.if_offset:
            offset = torch.tanh(self.offset_bn(self.offset_conv(x)))
        else:
            offset = torch.zeros(
                x.size(0),
                2 * self.kernel_size,
                x.size(2),
                x.size(3),
                device=x.device,
                dtype=x.dtype,
            )
        x = self.sampler(x, offset, if_offset=self.if_offset)
        x = self.snake_conv(x)
        return self.act(self.norm(x))
