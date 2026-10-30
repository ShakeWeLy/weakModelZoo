"""YOLOv5 基础模块：Conv / Bottleneck / C3 / SPPF。

Conv 复用 ``src.module.conv.BaseConv``（Conv2d 无偏置 + BN + SiLU），
BN 参数与官方 YOLOv5 保持一致（eps=1e-3, momentum=0.03）。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.module.conv import BaseConv


def Conv(c1: int, c2: int, k: int = 1, s: int = 1, p: int | None = None, g: int = 1, act: bool = True) -> nn.Module:
    """YOLOv5 标准卷积块：Conv2d(bias=False) + BatchNorm + SiLU。"""
    return BaseConv(
        c1, c2, k, s, p, groups=g, act=act,
        bn=nn.BatchNorm2d(c2, eps=1e-3, momentum=0.03),
    )


class Bottleneck(nn.Module):
    """1x1 降维 -> 3x3 卷积，可选残差连接。"""

    def __init__(self, c1: int, c2: int, shortcut: bool = True, e: float = 1.0) -> None:
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions。"""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, e: float = 0.5) -> None:
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, e=1.0) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast（串行 MaxPool 等价于 SPP）。"""

    def __init__(self, c1: int, c2: int, k: int = 5) -> None:
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), dim=1))
