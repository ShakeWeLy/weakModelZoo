"""YOLOv8 标准结构（Anchor-Free）：Backbone(C2f) - PANet Neck - Detect(DFL)。

训练::
    model.train()  -> dict(boxes, scores, feats)
                      boxes:  (bs, 4*reg_max, N)  四边距离分布 logits
                      scores: (bs, nc, N)         分类 logits
                      feats:  三尺度特征图列表（供造格子点）

推理::
    model.eval()   -> (pred, raw)
                      pred: (bs, N, 5+nc)，xywh 像素 + obj(=1) + cls，兼容现有 NMS
                      raw : 与训练相同的 dict
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn

from src.models.detection.YOLO.common import Conv, SPPF
from src.models.detection.YOLO.tal import dist2bbox, make_anchors
from src.models.detection.YOLO.yolov8_common import C2f

# variant -> (depth_multiple, width_multiple)；与 ultralytics YOLOv8 对齐
VARIANTS: dict[str, tuple[float, float]] = {
    "n": (0.33, 0.25),
    "s": (0.33, 0.50),
    "m": (0.67, 0.75),
    "l": (1.00, 1.00),
    "x": (1.00, 1.25),
}

DEFAULT_STRIDES: tuple[int, int, int] = (8, 16, 32)


def make_divisible(x: float, divisor: int = 8) -> int:
    return math.ceil(x / divisor) * divisor


class _Scaler:
    def __init__(self, gw: float, gd: float) -> None:
        self.gw, self.gd = gw, gd

    def c(self, channels: int) -> int:
        return make_divisible(channels * self.gw, 8)

    def n(self, repeats: int) -> int:
        return max(round(repeats * self.gd), 1) if repeats > 1 else repeats


class DFL(nn.Module):
    """Distribution Focal Loss 积分：把每边 reg_max 维分布变成期望距离。"""

    def __init__(self, c1: int = 16) -> None:
        super().__init__()
        self.c1 = c1
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, a = x.shape  # (bs, 4*reg_max, N)
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)


class Backbone(nn.Module):
    """YOLOv8 CSPDarknet 风格骨干，输出 P3 / P4 / P5。"""

    def __init__(self, gw: float, gd: float, in_channels: int = 3) -> None:
        super().__init__()
        s = _Scaler(gw, gd)
        self.stem = Conv(in_channels, s.c(64), 3, 2)                 # /2
        self.stage2 = nn.Sequential(                                 # /4
            Conv(s.c(64), s.c(128), 3, 2),
            C2f(s.c(128), s.c(128), s.n(3), True),
        )
        self.stage3 = nn.Sequential(                                 # /8  P3
            Conv(s.c(128), s.c(256), 3, 2),
            C2f(s.c(256), s.c(256), s.n(6), True),
        )
        self.stage4 = nn.Sequential(                                 # /16 P4
            Conv(s.c(256), s.c(512), 3, 2),
            C2f(s.c(512), s.c(512), s.n(6), True),
        )
        self.stage5 = nn.Sequential(                                 # /32 P5
            Conv(s.c(512), s.c(1024), 3, 2),
            C2f(s.c(1024), s.c(1024), s.n(3), True),
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
    """YOLOv8 PANet：FPN 自顶向下 + PAN 自底向上。"""

    def __init__(self, ch3: int, ch4: int, ch5: int, gw: float, gd: float) -> None:
        super().__init__()
        s = _Scaler(gw, gd)
        c3, c4, c5 = s.c(256), s.c(512), s.c(1024)
        self.up = nn.Upsample(scale_factor=2, mode="nearest")

        self.reduce5 = Conv(ch5, c4, 1, 1)
        self.c2f_p4 = C2f(c4 + ch4, c4, s.n(3), False)
        self.reduce4 = Conv(c4, c3, 1, 1)
        self.c2f_p3 = C2f(c3 + ch3, c3, s.n(3), False)

        self.down3 = Conv(c3, c3, 3, 2)
        self.c2f_n4 = C2f(c3 + c3, c4, s.n(3), False)
        self.down4 = Conv(c4, c4, 3, 2)
        self.c2f_n5 = C2f(c4 + c4, c5, s.n(3), False)

        self.out_channels = (c3, c4, c5)

    def forward(
        self, p3: torch.Tensor, p4: torch.Tensor, p5: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x5 = self.reduce5(p5)
        x4 = self.c2f_p4(torch.cat((self.up(x5), p4), 1))
        x3 = self.reduce4(x4)
        n3 = self.c2f_p3(torch.cat((self.up(x3), p3), 1))
        n4 = self.c2f_n4(torch.cat((self.down3(n3), x3), 1))
        n5 = self.c2f_n5(torch.cat((self.down4(n4), x5), 1))
        return n3, n4, n5


class Detect(nn.Module):
    """Anchor-Free 检测头：框分支 (4*reg_max) + 分类分支 (nc)，无 objectness。"""

    def __init__(
        self,
        nc: int,
        ch: Sequence[int],
        strides: Sequence[int] = DEFAULT_STRIDES,
        reg_max: int = 16,
    ) -> None:
        super().__init__()
        self.nc = nc
        self.nl = len(ch)
        self.reg_max = reg_max
        self.no = nc + reg_max * 4
        if len(strides) != self.nl:
            raise ValueError("ch / strides 尺度数需一致")
        self.register_buffer("stride", torch.tensor(strides, dtype=torch.float32))

        c2 = max(16, ch[0] // 4, reg_max * 4)
        c3 = max(ch[0], min(nc, 100))
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * reg_max, 1)) for x in ch
        )
        self.cv3 = nn.ModuleList(
            nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, nc, 1)) for x in ch
        )
        self.dfl = DFL(reg_max) if reg_max > 1 else nn.Identity()
        self._init_biases()

    def _init_biases(self) -> None:
        for a, b, s in zip(self.cv2, self.cv3, self.stride.tolist()):
            a[-1].bias.data[:] = 1.0
            b[-1].bias.data[: self.nc] = math.log(5 / self.nc / (640 / s) ** 2)

    def forward(self, x: list[torch.Tensor]):
        bs = x[0].shape[0]
        boxes = torch.cat([self.cv2[i](x[i]).view(bs, 4 * self.reg_max, -1) for i in range(self.nl)], dim=-1)
        scores = torch.cat([self.cv3[i](x[i]).view(bs, self.nc, -1) for i in range(self.nl)], dim=-1)
        raw = {"boxes": boxes, "scores": scores, "feats": x}
        if self.training:
            return raw
        return self._decode_for_nms(raw), raw

    def _decode_for_nms(self, raw: dict[str, torch.Tensor]) -> torch.Tensor:
        """解码为 (bs, N, 5+nc)：xywh + obj(=1) + cls，兼容 YOLOv5 风格 NMS。"""
        feats = raw["feats"]
        anchors, strides = make_anchors(feats, self.stride, 0.5)
        dist = self.dfl(raw["boxes"])  # (bs, 4, N)
        # dist2bbox 要 (bs, N, 4)；anchors (N,2)
        dbox = dist2bbox(dist.permute(0, 2, 1), anchors.unsqueeze(0), xywh=True) * strides
        cls = raw["scores"].permute(0, 2, 1).sigmoid()
        obj = torch.ones((*cls.shape[:2], 1), device=cls.device, dtype=cls.dtype)
        return torch.cat((dbox, obj, cls), dim=-1)


class YOLOv8(nn.Module):
    """YOLOv8 检测模型。

    Args:
        nc: 类别数。
        variant: n / s / m / l / x。
        in_channels: 输入通道。
        depth_multiple / width_multiple: 覆盖 variant 缩放。
        reg_max: DFL 离散 bins，默认 16。
    """

    def __init__(
        self,
        nc: int = 80,
        variant: str = "s",
        in_channels: int = 3,
        depth_multiple: float | None = None,
        width_multiple: float | None = None,
        reg_max: int = 16,
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
        self.detect = Detect(nc, self.neck.out_channels, DEFAULT_STRIDES, reg_max=reg_max)

    @property
    def nc(self) -> int:
        return self.detect.nc

    def forward(self, x: torch.Tensor):
        return self.detect(list(self.neck(*self.backbone(x))))
