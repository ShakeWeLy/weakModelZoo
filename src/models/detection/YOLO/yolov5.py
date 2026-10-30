"""YOLOv5 标准结构（Backbone - PANet Neck - Detect Head），与官方 v7.0 yaml 对齐。

结构概览（以 P3/P4/P5 为三个检测尺度，stride = 8 / 16 / 32）::

    Backbone: stem(6x6) -> P2 -> P3 -> P4 -> P5(SPPF)
    Neck    : FPN（自顶向下）+ PAN（自底向上）
    Detect  : 每个尺度 1x1 conv 输出 na * (5 + nc) 个通道

输出（训练 / 推理）::

    model.train()  -> list[Tensor]，每个元素 (bs, na, ny, nx, 5 + nc)，原始 logits，供 YOLOv5Loss 使用
    model.eval()   -> (pred, raw)
                      pred: (bs, N, 5 + nc)，已解码为 (cx, cy, w, h) 像素坐标 + sigmoid 后的 obj / cls，需再接 NMS
                      raw : 与训练时相同的 list[Tensor]

用法::

    from src.models.detection.YOLO.yolov5 import YOLOv5
    model = YOLOv5(nc=1, variant="s")
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn

from src.models.detection.YOLO.common import C3, SPPF, Conv

# 官方 COCO 预设锚框（输入 640 下的像素尺寸），P3 / P4 / P5 各 3 个
DEFAULT_ANCHORS: tuple[tuple[float, ...], ...] = (
    (10, 13, 16, 30, 33, 23),
    (30, 61, 62, 45, 59, 119),
    (116, 90, 156, 198, 373, 326),
)
DEFAULT_STRIDES: tuple[int, int, int] = (8, 16, 32)

# variant -> (depth_multiple, width_multiple)
VARIANTS: dict[str, tuple[float, float]] = {
    "n": (0.33, 0.25),
    "s": (0.33, 0.50),
    "m": (0.67, 0.75),
    "l": (1.00, 1.00),
    "x": (1.33, 1.25),
}


def make_divisible(x: float, divisor: int = 8) -> int:
    return math.ceil(x / divisor) * divisor


class _Scaler:
    """按 width / depth multiple 缩放通道数与重复次数。"""

    def __init__(self, gw: float, gd: float) -> None:
        self.gw, self.gd = gw, gd

    def c(self, channels: int) -> int:
        return make_divisible(channels * self.gw, 8)

    def n(self, repeats: int) -> int:
        return max(round(repeats * self.gd), 1) if repeats > 1 else repeats


class Backbone(nn.Module):
    """CSPDarknet 骨干，输出 P3 / P4 / P5。"""

    def __init__(self, gw: float, gd: float, in_channels: int = 3) -> None:
        super().__init__()
        s = _Scaler(gw, gd)
        self.stem = Conv(in_channels, s.c(64), 6, 2, 2)                        # P1/2
        self.stage2 = nn.Sequential(                                           # P2/4
            Conv(s.c(64), s.c(128), 3, 2), C3(s.c(128), s.c(128), s.n(3)),
        )
        self.stage3 = nn.Sequential(                                           # P3/8
            Conv(s.c(128), s.c(256), 3, 2), C3(s.c(256), s.c(256), s.n(6)),
        )
        self.stage4 = nn.Sequential(                                           # P4/16
            Conv(s.c(256), s.c(512), 3, 2), C3(s.c(512), s.c(512), s.n(9)),
        )
        self.stage5 = nn.Sequential(                                           # P5/32
            Conv(s.c(512), s.c(1024), 3, 2), C3(s.c(1024), s.c(1024), s.n(3)),
            SPPF(s.c(1024), s.c(1024), 5),
        )
        self.out_channels = (s.c(256), s.c(512), s.c(1024))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        x = self.stage2(x)
        p3 = self.stage3(x)
        p4 = self.stage4(p3)
        p5 = self.stage5(p4)
        return p3, p4, p5


class Neck(nn.Module):
    """PANet：FPN 自顶向下 + PAN 自底向上，输出 N3 / N4 / N5。"""

    def __init__(self, ch3: int, ch4: int, ch5: int, gw: float, gd: float) -> None:
        super().__init__()
        s = _Scaler(gw, gd)
        c256, c512, c1024 = s.c(256), s.c(512), s.c(1024)
        self.up = nn.Upsample(scale_factor=2, mode="nearest")

        # 自顶向下（FPN）
        self.reduce5 = Conv(ch5, c512, 1, 1)
        self.c3_p4 = C3(c512 + ch4, c512, s.n(3), shortcut=False)
        self.reduce4 = Conv(c512, c256, 1, 1)
        self.c3_p3 = C3(c256 + ch3, c256, s.n(3), shortcut=False)

        # 自底向上（PAN）
        self.down3 = Conv(c256, c256, 3, 2)
        self.c3_n4 = C3(c256 + c256, c512, s.n(3), shortcut=False)
        self.down4 = Conv(c512, c512, 3, 2)
        self.c3_n5 = C3(c512 + c512, c1024, s.n(3), shortcut=False)

        self.out_channels = (c256, c512, c1024)

    def forward(
        self, p3: torch.Tensor, p4: torch.Tensor, p5: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x10 = self.reduce5(p5)
        x13 = self.c3_p4(torch.cat((self.up(x10), p4), dim=1))
        x14 = self.reduce4(x13)
        n3 = self.c3_p3(torch.cat((self.up(x14), p3), dim=1))
        n4 = self.c3_n4(torch.cat((self.down3(n3), x14), dim=1))
        n5 = self.c3_n5(torch.cat((self.down4(n4), x10), dim=1))
        return n3, n4, n5


class Detect(nn.Module):
    """检测头：每个尺度一个 1x1 卷积，输出 (tx, ty, tw, th, obj, cls...)。"""

    def __init__(
        self,
        nc: int,
        ch: Sequence[int],
        anchors: Sequence[Sequence[float]] = DEFAULT_ANCHORS,
        strides: Sequence[int] = DEFAULT_STRIDES,
    ) -> None:
        super().__init__()
        self.nc = nc
        self.no = nc + 5
        self.nl = len(anchors)
        self.na = len(anchors[0]) // 2
        if len(ch) != self.nl or len(strides) != self.nl:
            raise ValueError("ch / anchors / strides 的尺度数需一致")

        # anchors 以输入图像像素为单位，形状 (nl, na, 2)
        self.register_buffer("anchors", torch.tensor(anchors, dtype=torch.float32).view(self.nl, self.na, 2))
        self.register_buffer("stride", torch.tensor(strides, dtype=torch.float32))
        self.m = nn.ModuleList(nn.Conv2d(c, self.no * self.na, 1) for c in ch)
        self._init_biases()

    def _init_biases(self) -> None:
        """与官方一致：obj 偏置按每图约 8 个目标设定；cls 偏置按先验类别概率设定（多类时）。"""
        with torch.no_grad():
            for mi, s in zip(self.m, self.stride.tolist()):
                b = mi.bias.view(self.na, -1)
                b[:, 4] += math.log(8 / (640 / s) ** 2)
                if self.nc > 1:
                    b[:, 5:] += math.log(0.6 / (self.nc - 0.99999))

    @staticmethod
    def _grid(ny: int, nx: int, device: torch.device) -> torch.Tensor:
        yv, xv = torch.meshgrid(torch.arange(ny, device=device), torch.arange(nx, device=device), indexing="ij")
        return torch.stack((xv, yv), dim=2).view(1, 1, ny, nx, 2).float()

    def _decode(self, i: int, y: torch.Tensor) -> torch.Tensor:
        bs, na, ny, nx, _ = y.shape
        grid = self._grid(ny, nx, y.device)
        s = self.stride[i]
        y = y.sigmoid()
        xy = (y[..., 0:2] * 2 - 0.5 + grid) * s
        wh = (y[..., 2:4] * 2) ** 2 * self.anchors[i].view(1, na, 1, 1, 2)
        return torch.cat((xy, wh, y[..., 4:]), dim=-1).view(bs, -1, self.no)

    def forward(self, x: list[torch.Tensor]):
        raw: list[torch.Tensor] = []
        decoded: list[torch.Tensor] = []
        for i in range(self.nl):
            bs, _, ny, nx = x[i].shape
            y = self.m[i](x[i]).view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            raw.append(y)
            if not self.training:
                decoded.append(self._decode(i, y))
        if self.training:
            return raw
        return torch.cat(decoded, dim=1), raw


class YOLOv5(nn.Module):
    """YOLOv5 检测模型。

    Args:
        nc: 类别数（不含背景）。
        variant: 模型规模，n / s / m / l / x。
        anchors: 每个尺度的锚框（像素），默认官方 COCO 锚框。
        in_channels: 输入通道数。
        depth_multiple / width_multiple: 若指定则覆盖 variant 中的缩放系数，便于自定义结构。
    """

    def __init__(
        self,
        nc: int = 80,
        variant: str = "s",
        anchors: Sequence[Sequence[float]] = DEFAULT_ANCHORS,
        in_channels: int = 3,
        depth_multiple: float | None = None,
        width_multiple: float | None = None,
    ) -> None:
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"未知 variant: {variant}（可选 {', '.join(VARIANTS)}）")
        gd, gw = VARIANTS[variant]
        gd = gd if depth_multiple is None else depth_multiple
        gw = gw if width_multiple is None else width_multiple

        self.variant = variant
        self.backbone = Backbone(gw, gd, in_channels)
        self.neck = Neck(*self.backbone.out_channels, gw, gd)
        self.detect = Detect(nc, self.neck.out_channels, anchors)

    @property
    def nc(self) -> int:
        return self.detect.nc

    def forward(self, x: torch.Tensor):
        return self.detect(list(self.neck(*self.backbone(x))))
