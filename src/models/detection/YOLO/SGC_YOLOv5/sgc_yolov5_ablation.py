"""SGC-YOLOv5 消融变体（独立模型名，无开关）。

1. SGC_YOLOv5_NoCBAM      : 去掉输出前 CBAM
2. SGC_YOLOv5_C3Neck      : Neck 中 VOVGCSP -> 标准 C3（保留 GSConv / CBAM）
3. SGC_YOLOv5_StdEarly    : Backbone 浅层 stage2/3 用 Conv+C3，深层仍用 SGhost/SDGhost
4. SGC_YOLOv5_StdBackbone : Backbone 全标准 Conv+C3；Neck 用 VOVGCSPYolo（GSConv+CBAM）
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.detection.YOLO.common import C3, Conv, SPPF
from src.models.detection.YOLO.yolov5 import DEFAULT_ANCHORS, VARIANTS, Detect, _Scaler
from src.models.detection.YOLO.SGC_YOLOv5.sgc_yolov5 import (
    Backbone,
    CBAM,
    GSConv,
    Neck,
    SDGhost,
    SGhostConv,
    VOVGCSP,
)


def _parse_scales(
    variant: str,
    depth_multiple: float | None,
    width_multiple: float | None,
) -> tuple[float, float]:
    if variant not in VARIANTS:
        raise ValueError(f"未知 variant: {variant}（可选 {', '.join(VARIANTS)}）")
    gd, gw = VARIANTS[variant]
    gd = gd if depth_multiple is None else depth_multiple
    gw = gw if width_multiple is None else width_multiple
    return gd, gw


class NeckNoCBAM(nn.Module):
    """与 SGC Neck 相同，但输出前不加 CBAM。"""

    def __init__(self, ch3: int, ch4: int, ch5: int, gw: float, gd: float) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        s = _Scaler(gw, gd)
        c3, c4, c5 = ch3, ch4, ch5

        self.reduce5 = GSConv(c5, c4, 1, 1)
        self.c3_p4 = VOVGCSP(c4 + c4, c4, n=s.n(3))
        self.reduce4 = GSConv(c4, c3, 1, 1)
        self.c3_p3 = VOVGCSP(c3 + c3, c3, n=s.n(3))

        self.down3 = GSConv(c3, c3, 3, 2)
        self.c3_n4 = VOVGCSP(c3 + c3, c4, n=s.n(3))
        self.down4 = GSConv(c4, c4, 3, 2)
        self.c3_n5 = VOVGCSP(c4 + c4, c5, n=s.n(3))
        self.out_channels = (c3, c4, c5)

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


class NeckC3(nn.Module):
    """SGC Neck，但把 VOVGCSP 换成标准 C3；保留 GSConv 与 CBAM。"""

    def __init__(self, ch3: int, ch4: int, ch5: int, gw: float, gd: float) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        s = _Scaler(gw, gd)
        c3, c4, c5 = ch3, ch4, ch5

        self.reduce5 = GSConv(c5, c4, 1, 1)
        self.c3_p4 = C3(c4 + c4, c4, s.n(3), shortcut=False)
        self.reduce4 = GSConv(c4, c3, 1, 1)
        self.c3_p3 = C3(c3 + c3, c3, s.n(3), shortcut=False)

        self.down3 = GSConv(c3, c3, 3, 2)
        self.c3_n4 = C3(c3 + c3, c4, s.n(3), shortcut=False)
        self.down4 = GSConv(c4, c4, 3, 2)
        self.c3_n5 = C3(c4 + c4, c5, s.n(3), shortcut=False)

        self.out_channels = (c3, c4, c5)
        self.cbam_p3 = CBAM(c3)
        self.cbam_p4 = CBAM(c4)
        self.cbam_p5 = CBAM(c5)

    def forward(
        self, p3: torch.Tensor, p4: torch.Tensor, p5: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x10 = self.reduce5(p5)
        x13 = self.c3_p4(torch.cat((self.up(x10), p4), dim=1))
        x14 = self.reduce4(x13)
        n3 = self.c3_p3(torch.cat((self.up(x14), p3), dim=1))
        n4 = self.c3_n4(torch.cat((self.down3(n3), x14), dim=1))
        n5 = self.c3_n5(torch.cat((self.down4(n4), x10), dim=1))
        return self.cbam_p3(n3), self.cbam_p4(n4), self.cbam_p5(n5)


class BackboneStdEarly(nn.Module):
    """浅层 stage2/3 使用标准 Conv+C3；深层 stage4/5 仍用 SGhost+SDGhost。"""

    def __init__(self, gw: float = 1.0, gd: float = 1.0, in_channels: int = 3) -> None:
        super().__init__()
        s = _Scaler(gw, gd)
        self.stem = Conv(in_channels, s.c(64), 6, 2, 2)
        self.stage2 = nn.Sequential(  # P2/4：标准
            Conv(s.c(64), s.c(128), 3, 2),
            C3(s.c(128), s.c(128), s.n(3)),
        )
        self.stage3 = nn.Sequential(  # P3/8：标准（小目标关键层）
            Conv(s.c(128), s.c(256), 3, 2),
            C3(s.c(256), s.c(256), s.n(6)),
        )
        self.stage4 = nn.Sequential(  # P4/16：保留 SGC
            SGhostConv(s.c(256), s.c(512), 3, 2, 1),
            SDGhost(s.c(512), s.c(512), n=s.n(9)),
        )
        self.stage5 = nn.Sequential(  # P5/32：保留 SGC
            SGhostConv(s.c(512), s.c(1024), 3, 2, 1),
            SDGhost(s.c(1024), s.c(1024), n=s.n(3)),
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


class _SGCDetBase(nn.Module):
    """统一 Detect 接口。"""

    def __init__(self, backbone: nn.Module, neck: nn.Module, nc: int, anchors) -> None:
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.detect = Detect(nc, self.neck.out_channels, anchors)

    @property
    def nc(self) -> int:
        return self.detect.nc

    def forward(self, x: torch.Tensor):
        return self.detect(list(self.neck(*self.backbone(x))))


class SGC_YOLOv5_NoCBAM(_SGCDetBase):
    """消融 1：去掉输出前 CBAM，其余与 SGC_YOLOv5 相同。"""

    def __init__(
        self,
        nc: int = 80,
        variant: str = "s",
        anchors=DEFAULT_ANCHORS,
        in_channels: int = 3,
        depth_multiple: float | None = None,
        width_multiple: float | None = None,
    ) -> None:
        gd, gw = _parse_scales(variant, depth_multiple, width_multiple)
        backbone = Backbone(gw, gd, in_channels)
        neck = NeckNoCBAM(*backbone.out_channels, gw, gd)
        super().__init__(backbone, neck, nc, anchors)
        self.variant = variant


class SGC_YOLOv5_C3Neck(_SGCDetBase):
    """消融 2：Neck 中 VOVGCSP 换成标准 C3，保留 GSConv 与 CBAM。"""

    def __init__(
        self,
        nc: int = 80,
        variant: str = "s",
        anchors=DEFAULT_ANCHORS,
        in_channels: int = 3,
        depth_multiple: float | None = None,
        width_multiple: float | None = None,
    ) -> None:
        gd, gw = _parse_scales(variant, depth_multiple, width_multiple)
        backbone = Backbone(gw, gd, in_channels)
        neck = NeckC3(*backbone.out_channels, gw, gd)
        super().__init__(backbone, neck, nc, anchors)
        self.variant = variant


class SGC_YOLOv5_StdEarly(_SGCDetBase):
    """消融 3：浅层 Backbone（stage2/3）用标准 Conv+C3，深层与 Neck 保持 SGC。"""

    def __init__(
        self,
        nc: int = 80,
        variant: str = "s",
        anchors=DEFAULT_ANCHORS,
        in_channels: int = 3,
        depth_multiple: float | None = None,
        width_multiple: float | None = None,
    ) -> None:
        gd, gw = _parse_scales(variant, depth_multiple, width_multiple)
        backbone = BackboneStdEarly(gw, gd, in_channels)
        neck = Neck(*backbone.out_channels, gw, gd)
        super().__init__(backbone, neck, nc, anchors)
        self.variant = variant


class BackboneStdFull(nn.Module):
    """全阶段标准 YOLOv5 Backbone：Conv+C3（含 SPPF）。"""

    def __init__(self, gw: float = 1.0, gd: float = 1.0, in_channels: int = 3) -> None:
        super().__init__()
        s = _Scaler(gw, gd)
        self.stem = Conv(in_channels, s.c(64), 6, 2, 2)
        self.stage2 = nn.Sequential(
            Conv(s.c(64), s.c(128), 3, 2),
            C3(s.c(128), s.c(128), s.n(3)),
        )
        self.stage3 = nn.Sequential(
            Conv(s.c(128), s.c(256), 3, 2),
            C3(s.c(256), s.c(256), s.n(6)),
        )
        self.stage4 = nn.Sequential(
            Conv(s.c(256), s.c(512), 3, 2),
            C3(s.c(512), s.c(512), s.n(9)),
        )
        self.stage5 = nn.Sequential(
            Conv(s.c(512), s.c(1024), 3, 2),
            C3(s.c(1024), s.c(1024), s.n(3)),
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


class SGC_YOLOv5_StdBackbone(_SGCDetBase):
    """消融：Backbone 全标准 Conv+C3；Neck 用 VOVGCSPYolo（保留 GSConv / CBAM）。"""

    def __init__(
        self,
        nc: int = 80,
        variant: str = "s",
        anchors=DEFAULT_ANCHORS,
        in_channels: int = 3,
        depth_multiple: float | None = None,
        width_multiple: float | None = None,
        groups_ratio: int = 4,
        dw_kernel_size: int = 1,
        ratio: int = 2,
    ) -> None:
        gd, gw = _parse_scales(variant, depth_multiple, width_multiple)
        backbone = BackboneStdFull(gw, gd, in_channels)
        neck_dw = 3 if dw_kernel_size <= 1 else dw_kernel_size
        neck = Neck(
            *backbone.out_channels,
            gw,
            gd,
            ratio=ratio,
            groups_ratio=groups_ratio,
            dw_kernel_size=neck_dw,
            vov_style="yolo",
        )
        super().__init__(backbone, neck, nc, anchors)
        self.variant = variant
