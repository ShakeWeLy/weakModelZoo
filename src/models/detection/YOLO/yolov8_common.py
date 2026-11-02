"""YOLOv8 基础模块：在 YOLOv5 Conv/SPPF 之上增加 C2f。"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.detection.YOLO.common import Conv


class Bottleneck(nn.Module):
    """YOLOv8 Bottleneck：默认两层 3x3（与 ultralytics 一致）。"""

    def __init__(self, c1: int, c2: int, shortcut: bool = True, e: float = 0.5) -> None:
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 3, 1)
        self.cv2 = Conv(c_, c2, 3, 1)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class C2f(nn.Module):
    """CSP Bottleneck with 2 convolutions（YOLOv8）：分支更多、梯度更丰富。"""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, e: float = 0.5) -> None:
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, e=1.0) for _ in range(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
