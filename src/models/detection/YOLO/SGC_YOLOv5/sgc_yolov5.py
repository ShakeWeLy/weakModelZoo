
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.module.conv import DepthwiseConv, GhostConv, BaseConv
from src.models.detection.YOLO.common import Bottleneck as YoloBottleneck
from src.models.detection.YOLO.common import Conv, SPPF
from src.models.detection.YOLO.yolov5 import DEFAULT_ANCHORS, VARIANTS, Detect, _Scaler  # _Scaler: 与标准 YOLOv5 相同的缩放规则
from src.module.attention import CAM
from src.module.attention.SAM.sam import SAM


def _shuffle_groups(channels: int, groups_ratio: int) -> int:
    """ChannelShuffle 要求 ``channels % groups == 0``；按 ``channels // groups_ratio`` 向下取合法分组数。"""
    if groups_ratio < 1:
        raise ValueError(f"groups_ratio 必须 >= 1，当前 {groups_ratio}")
    g = max(1, channels // groups_ratio)
    while g > 1 and channels % g != 0:
        g -= 1
    return g


class SGhostConv(GhostConv):
    '''
    最后做额外的通道混洗
    '''
    def __init__(self,
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=None,
                ratio=2,
                groups_ratio=2,
                dw_kernel_size=3,
                act=True,
                bn=True,
                ):
        super().__init__(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                         stride=stride, padding=padding, ratio=ratio, dw_kernel_size=dw_kernel_size,
                         act=act, bn=bn)
        self.groups = _shuffle_groups(out_channels, groups_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = super().forward(x)
        return torch.nn.ChannelShuffle(self.groups)(x)


class GBS(GhostConv):
    '''
    Ghost + BN + SILU
    '''
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=None,
                 ratio=2,
                 groups_ratio=2,
                 dw_kernel_size=3,
                 act=F.silu,
                 bn=F.batch_norm,
                 ) -> None:
        # groups_ratio 仅 SGhostConv 的通道混洗使用，GhostConv 不需要
        super().__init__(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                         stride=stride, padding=padding, ratio=ratio, dw_kernel_size=dw_kernel_size,
                         act=act, bn=bn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x)


class DFCAttention(nn.Module):
    '''
    DFC注意力模块利用~固定权重~(可学习的)全连接层来创建具有全局感受野的注意力图，
    以捕获长距离空间位置的依赖关系从而增强S-Ghost 模块
    '''
    def __init__(self, channels, kernel_size=5):
        super().__init__()
        # 用深度卷积模拟水平/垂直方向的全连接层
        self.h_conv = nn.Conv2d(channels, channels, (1, kernel_size),
                                padding=(0, kernel_size//2), groups=channels)
        self.v_conv = nn.Conv2d(channels, channels, (kernel_size, 1),
                                padding=(kernel_size//2, 0), groups=channels)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        h_feat = self.h_conv(x)
        v_feat = self.v_conv(x)
        return self.sigmoid(h_feat * v_feat)



class Bottleneck(nn.Module):
    '''
    特征增强, 不改变size和channel。
    默认两层 SGhostConv 共用 kernel_size；若指定 mid_kernel_size，则变为 YOLO 式 1×1 → 3×3
    （第一层 kernel_size，第二层 mid_kernel_size）。
    '''
    def __init__(self,
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=None,
                ratio=2,
                groups_ratio=2,
                dw_kernel_size=1,
                act=F.silu,
                bn=F.batch_norm,
                mid_kernel_size: int | None = None,
                ) -> None:
        super().__init__()
        if in_channels != out_channels:
            raise ValueError("Bottleneck 要求 in_channels == out_channels（DFC 注意力作用在输入上，需与 x1 通道一致）")
        # YOLO 的 Bottleneck 都是 stride=1，下采样由单独的卷积层完成，这里不支持 stride != 1
        if stride != 1:
            raise ValueError("Bottleneck 只支持 stride=1，下采样请使用单独的卷积层")
        k2 = kernel_size if mid_kernel_size is None else mid_kernel_size
        self.SGhostConv1 = SGhostConv(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                      stride=stride, padding=padding, ratio=ratio, groups_ratio=groups_ratio,
                                      dw_kernel_size=dw_kernel_size, act=act, bn=bn)
        self.SGhostConv2 = SGhostConv(in_channels=out_channels, out_channels=out_channels, kernel_size=k2,
                                      stride=stride, padding=padding, ratio=ratio, groups_ratio=groups_ratio,
                                      dw_kernel_size=dw_kernel_size, act=act, bn=bn)
        # self.DWConv1 = BaseConv(out_channels, out_channels, kernel_size, stride, groups=out_channels, act=act, bn=bn)  # stride=2 分支专用，已注释
        self.DFCAttention = DFCAttention(in_channels)
        # 残差支路的深度卷积固定 3×3，不随主卷积 kernel 变化（主卷积已改为 1×1）
        self.DWConv2 = BaseConv(out_channels, out_channels, 3, stride, groups=out_channels, act=act, bn=bn)
        # self.Conv = BaseConv(in_channels, out_channels, kernel_size, stride, act=act, bn=bn)  # stride=2 分支专用，已注释

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.DWConv2(x)
        x1 = self.SGhostConv1(x)
        x2 = self.DFCAttention(x)
        x = x1*x2

        # if self.stride == 2:  # YOLO 无此结构，已注释
        #     x = self.DWConv1(x)
        #     res = self.Conv(res)
        x = self.SGhostConv2(x)
        return x + res


class SDGhost(nn.Module):
    """
    和SGhost结合, SGhostConv+SDGhost, 替换CSP1,2,3。

    - ``e=1``（默认）：GBS1/2 与 Bottleneck 均在 ``out_channels`` 上（满通道）。
    - ``e<1``（如 0.5）：YOLO C3 式窄通道 —
      GBS1(1×1) 降维到 ``c_=int(out*e)`` → n×Bottleneck → GBS2(1×1) 升维回 out。
    - ``bottleneck_type``: ``ghost`` 用 SGhost+DFC；``dense`` 用 YOLO 稠密 1×1→3×3 Bottleneck。
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 n=1,
                 kernel_size=1,
                 stride=1,
                 padding=None,
                 ratio=2,
                 groups_ratio=2,
                 dw_kernel_size=1,
                 act=F.silu,
                 bn=F.batch_norm,
                 mid_kernel_size: int | None = None,
                 e: float = 1.0,
                 bottleneck_type: str = "ghost",
                 ) -> None:
        super().__init__()
        if n < 1:
            raise ValueError("n 必须 >= 1")
        if e <= 0 or e > 1:
            raise ValueError(f"e 须在 (0, 1]，当前 {e}")
        btype = str(bottleneck_type).lower()
        if btype not in {"ghost", "dense"}:
            raise ValueError(f"bottleneck_type 须为 ghost/dense，当前 {bottleneck_type!r}")
        # 隐藏宽度；e<1 时对齐 Ghost ratio（偶数通道更稳）
        c_ = int(out_channels * e)
        c_ = max(ratio, c_ - (c_ % ratio))
        if c_ > out_channels:
            c_ = out_channels
        self.hidden_channels = c_

        # GBS1 降维/投影 → Bottleneck → GBS2 升维/投影（GBS 仍为 Ghost）
        self.GBS1 = GBS(in_channels, c_, kernel_size, stride, padding, ratio, groups_ratio, dw_kernel_size, act, bn)
        self.GBS2 = GBS(c_, out_channels, kernel_size, stride, padding, ratio, groups_ratio, dw_kernel_size, act, bn)
        if btype == "dense":
            self.bottleneck = nn.Sequential(*[
                YoloBottleneck(c_, c_, shortcut=True, e=1.0) for _ in range(n)
            ])
        else:
            self.bottleneck = nn.Sequential(*[
                Bottleneck(
                    c_, c_, kernel_size, stride, padding, ratio, groups_ratio,
                    dw_kernel_size, act, bn, mid_kernel_size=mid_kernel_size,
                )
                for _ in range(n)
            ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.GBS1(x)
        x = x + self.bottleneck(x)
        x = self.GBS2(x)
        return x


class GSConv(nn.Module):
    '''
    CSP4 = 一个GSConv+upsample到512
    '''
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=1,
                 stride=1,
                 ratio=2,
                 groups_ratio=2,
                 dw_kernel_size=3,
                 act=F.silu,
                 bn=F.batch_norm,
                ):
        super().__init__()
        # 输出 = 普通卷积分支(dw_channels) + 深度卷积分支(dw_channels)，只有 ratio=2 且 out 为偶数时才等于 out_channels
        if ratio != 2 or out_channels % 2 != 0:
            raise ValueError(f"GSConv 要求 ratio=2 且 out_channels 为偶数，当前 ratio={ratio}, out_channels={out_channels}")
        self.dw_channels = out_channels // ratio
        # shuffle 作用在 cat 后的 out_channels 上，分组数需整除 out_channels
        self.groups = _shuffle_groups(out_channels, groups_ratio * ratio)
        self.conv = BaseConv(in_channels, self.dw_channels, kernel_size, stride, act=act, bn=bn)
        self.DWconv = DepthwiseConv(self.dw_channels, dw_kernel_size, act=act, bn=bn)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = torch.cat([x, self.DWconv(x)], dim=1)
        return torch.nn.ChannelShuffle(self.groups)(x)

class GSBottleneck(nn.Module):
    '''
    通道数保持不变的 GS 瓶颈: GSConv -> GSConv + 残差，对应 YOLO C3 中的 Bottleneck，可堆叠。
    mid_kernel_size 非空时第二层用该核（YOLO 式 1×1 → 3×3）。
    '''
    def __init__(
        self,
        channels,
        kernel_size=1,
        ratio=2,
        groups_ratio=2,
        dw_kernel_size=3,
        act=F.silu,
        bn=F.batch_norm,
        mid_kernel_size: int | None = None,
    ):
        super().__init__()
        k2 = kernel_size if mid_kernel_size is None else mid_kernel_size
        self.GSConv1 = GSConv(channels, channels, kernel_size, 1, ratio, groups_ratio, dw_kernel_size, act, bn)
        self.GSConv2 = GSConv(channels, channels, k2, 1, ratio, groups_ratio, dw_kernel_size, act, bn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.GSConv2(self.GSConv1(x)) + x


class VOVGCSP(nn.Module):
    '''
    该结构将输入分为两部分，其中一部分经过卷积操作实现通道数缩减，
    有效地控制了计算复杂度；另一部分在改变通道数后经过GS bottleneck，进一步增强了特征的提取能力。
    再将这两部分输出拼接在一起，进行卷积操作以实现输出通道的调整。

    VOVGCSP + GSConv替换 CSP5,6,7,8
    '''
    def __init__(self,
                 in_channels,
                 out_channels,
                 n=1,
                 kernel_size=1,
                 stride=1,
                 ratio=2,
                 groups_ratio=2,
                 ghost_ratio=2,
                 dw_kernel_size=3,
                 act=F.silu,
                 bn=F.batch_norm,
                 mid_kernel_size: int | None = None,
                 ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.in_channels_half = in_channels//2
        self.in_channels_reduce = in_channels - self.in_channels_half
        self.in_channels_half_ghost = self.in_channels_half * ghost_ratio

        if n < 1:
            raise ValueError("n 必须 >= 1")
        self.conv_bottle_input = BaseConv(self.in_channels_half, self.in_channels_half_ghost, 1, stride, act=act, bn=bn)  # 调整通道数
        # 堆叠 n 个 GS 瓶颈（通道数保持 half_ghost），对应 YOLO C3 的 n 个 Bottleneck
        self.gs_bottlenecks = nn.Sequential(*[
            GSBottleneck(
                self.in_channels_half_ghost, kernel_size, ratio, groups_ratio, dw_kernel_size, act, bn,
                mid_kernel_size=mid_kernel_size,
            )
            for _ in range(n)
        ])
        # 投影回 in_half，便于与 x2 拼接
        self.conv_bottle_output = BaseConv(self.in_channels_half_ghost, self.in_channels_half, 1, stride, act=act, bn=bn)

        self.conv_reduce = BaseConv(self.in_channels_reduce, self.in_channels_reduce, 1, stride, act=act, bn=bn)  # 通道数缩减

        self.conv_out = BaseConv(in_channels, out_channels, 1, stride, act=act, bn=bn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = torch.split(x, [self.in_channels_half, self.in_channels_reduce], dim=1)
        x1 = self.conv_bottle_input(x1)
        x1 = self.gs_bottlenecks(x1)
        x1 = self.conv_bottle_output(x1)
        x2 = self.conv_reduce(x2)
        x = torch.cat([x1, x2], dim=1)
        return self.conv_out(x)


class VOVGCSPYolo(nn.Module):
    """YOLO C3 式 VOVGCSP：双路 1×1 投影（非通道切分），主路宽度 ``c_≈out/2``，
    堆叠 ``GSBottleneck`` 为 1×1→3×3 并保留残差。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n: int = 1,
        ratio: int = 2,
        groups_ratio: int = 2,
        dw_kernel_size: int = 3,
        act=F.silu,
        bn=F.batch_norm,
        e: float = 0.5,
    ) -> None:
        super().__init__()
        if n < 1:
            raise ValueError("n 必须 >= 1")
        if e <= 0 or e > 1:
            raise ValueError(f"e 须在 (0, 1]，当前 {e}")
        c_ = int(out_channels * e)
        c_ = max(ratio, c_ - (c_ % ratio))
        self.cv1 = BaseConv(in_channels, c_, 1, 1, act=act, bn=bn)
        self.cv2 = BaseConv(in_channels, c_, 1, 1, act=act, bn=bn)
        self.m = nn.Sequential(*[
            GSBottleneck(
                c_,
                kernel_size=1,
                ratio=ratio,
                groups_ratio=groups_ratio,
                dw_kernel_size=dw_kernel_size,
                act=act,
                bn=bn,
                mid_kernel_size=3,
            )
            for _ in range(n)
        ])
        self.cv3 = BaseConv(2 * c_, out_channels, 1, 1, act=act, bn=bn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class CBAM(nn.Module):
    '''
    CBAM注意力模块
    '''
    def __init__(self, channels: int, reduction_ratio: int = 16, spatial_kernel: int = 7):
        super().__init__()
        # 先通道注意力，再空间注意力（与 CBAM 论文一致）
        self.channel_attention = CAM(channels, reduction_ratio=reduction_ratio)
        self.spatial_attention = SAM(kernel_size=spatial_kernel)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.spatial_attention(self.channel_attention(x))


class Backbone(nn.Module):
    def __init__(
        self,
        gw: float = 1.0,
        gd: float = 1.0,
        in_channels: int = 3,
        *,
        ratio: int = 2,
        groups_ratio: int = 2,
        kernel_size: int = 1,
        dw_kernel_size: int = 1,
        mid_kernel_size: int | None = None,
        e: float = 1.0,
        bottleneck_type: str = "ghost",
    ) -> None:
        super().__init__()
        s = _Scaler(gw, gd)
        ghost_kw = dict(ratio=ratio, groups_ratio=groups_ratio, dw_kernel_size=dw_kernel_size)
        block_kw = dict(
            kernel_size=kernel_size,
            ratio=ratio,
            groups_ratio=groups_ratio,
            dw_kernel_size=dw_kernel_size,
            mid_kernel_size=mid_kernel_size,
            e=e,
            bottleneck_type=bottleneck_type,
        )
        self.stem = Conv(in_channels, s.c(64), 6, 2, 2)  # P1/2
        self.stage2 = nn.Sequential(  # P2/4
            SGhostConv(s.c(64), s.c(128), 3, 2, 1, **ghost_kw),
            SDGhost(s.c(128), s.c(128), n=s.n(3), **block_kw),
        )
        self.stage3 = nn.Sequential(  # P3/8
            SGhostConv(s.c(128), s.c(256), 3, 2, 1, **ghost_kw),
            SDGhost(s.c(256), s.c(256), n=s.n(6), **block_kw),
        )
        self.stage4 = nn.Sequential(  # P4/16
            SGhostConv(s.c(256), s.c(512), 3, 2, 1, **ghost_kw),
            SDGhost(s.c(512), s.c(512), n=s.n(9), **block_kw),
        )
        self.stage5 = nn.Sequential(  # P5/32
            SGhostConv(s.c(512), s.c(1024), 3, 2, 1, **ghost_kw),
            SDGhost(s.c(1024), s.c(1024), n=s.n(3), **block_kw),
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
    """PANet：FPN 自顶向下 + PAN 自底向上，输出 N3 / N4 / N5。

    ``vov_style``:
      - ``split``: 原 VOVGCSP（通道切分）
      - ``yolo``: VOVGCSPYolo（双 1×1 投影 + 窄通道 GSBottleneck 1×1→3×3）
    """
    def __init__(
        self,
        ch3: int,
        ch4: int,
        ch5: int,
        gw: float,
        gd: float,
        *,
        ratio: int = 2,
        groups_ratio: int = 2,
        kernel_size: int = 1,
        dw_kernel_size: int = 3,
        mid_kernel_size: int | None = None,
        vov_style: str = "split",
    ) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")  # ???为什么要上采样？

        s = _Scaler(gw, gd)
        # 通道数由 Backbone 输出推导（已按 gw 缩放），不再写死
        c3, c4, c5 = ch3, ch4, ch5
        gs_kw = dict(ratio=ratio, groups_ratio=groups_ratio, dw_kernel_size=dw_kernel_size)
        style = str(vov_style).lower()
        if style == "yolo":
            def _vov(cin: int, cout: int, n: int) -> nn.Module:
                return VOVGCSPYolo(
                    cin, cout, n=n, ratio=ratio, groups_ratio=groups_ratio,
                    dw_kernel_size=dw_kernel_size,
                )
        elif style == "split":
            vov_kw = dict(
                kernel_size=kernel_size,
                ratio=ratio,
                groups_ratio=groups_ratio,
                dw_kernel_size=dw_kernel_size,
                mid_kernel_size=mid_kernel_size,
            )

            def _vov(cin: int, cout: int, n: int) -> nn.Module:
                return VOVGCSP(cin, cout, n=n, **vov_kw)
        else:
            raise ValueError(f"未知 vov_style={vov_style!r}（可选 split / yolo）")

        self.reduce5 = GSConv(c5, c4, 1, 1, **gs_kw)
        self.c3_p4 = _vov(c4 + c4, c4, s.n(3))
        self.reduce4 = GSConv(c4, c3, 1, 1, **gs_kw)
        self.c3_p3 = _vov(c3 + c3, c3, s.n(3))

        self.down3 = GSConv(c3, c3, 3, 2, **gs_kw)
        self.c3_n4 = _vov(c3 + c3, c4, s.n(3))
        self.down4 = GSConv(c4, c4, 3, 2, **gs_kw)
        self.c3_n5 = _vov(c4 + c4, c5, s.n(3))

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
        n3 = self.cbam_p3(n3)
        n4 = self.cbam_p4(n4)
        n5 = self.cbam_p5(n5)
        return n3, n4, n5


class SGC_YOLOv5(nn.Module):
    """SGC-YOLOv5 检测模型：SGC Backbone + PANet Neck（输出前接 CBAM）+ 标准 Detect 头。

    接口与 ``YOLOv5`` 一致（``.detect`` / ``.nc``），可直接与 ``YOLOv5Loss`` 和训练工程配合使用。

    Args:
        nc: 类别数（不含背景）。
        variant: 模型规模，n / s / m / l / x。
        anchors: 每个尺度的锚框（像素），默认官方 COCO 锚框。
        in_channels: 输入通道数。
        depth_multiple / width_multiple: 若指定则覆盖 variant 中的缩放系数。
    """

    def __init__(
        self,
        nc: int = 80,
        variant: str = "s",
        anchors=DEFAULT_ANCHORS,
        in_channels: int = 3,
        depth_multiple: float | None = None,
        width_multiple: float | None = None,
        ratio: int = 2,
        groups_ratio: int = 2,
        kernel_size: int = 1,
        dw_kernel_size: int = 1,
        mid_kernel_size: int | None = None,
        e: float = 1.0,
        vov_style: str = "split",
        bottleneck_type: str = "ghost",
    ) -> None:
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"未知 variant: {variant}（可选 {', '.join(VARIANTS)}）")
        gd, gw = VARIANTS[variant]
        gd = gd if depth_multiple is None else depth_multiple
        gw = gw if width_multiple is None else width_multiple

        self.variant = variant
        ghost_kw = dict(
            ratio=ratio,
            groups_ratio=groups_ratio,
            kernel_size=kernel_size,
            dw_kernel_size=dw_kernel_size,
            mid_kernel_size=mid_kernel_size,
            e=e,
            bottleneck_type=bottleneck_type,
        )
        self.backbone = Backbone(gw, gd, in_channels, **ghost_kw)
        # Neck 的 GSConv 廉价支路默认应为 3；若显式传入 dw_kernel_size 则一并覆盖
        # Neck/VOVGCSP 本身已有通道拆分，此处不传 e（仅 SDGhost 做 CSP 窄通道）
        neck_dw = 3 if dw_kernel_size <= 1 else dw_kernel_size
        self.neck = Neck(
            *self.backbone.out_channels,
            gw,
            gd,
            ratio=ratio,
            groups_ratio=groups_ratio,
            kernel_size=kernel_size,
            dw_kernel_size=neck_dw,
            mid_kernel_size=mid_kernel_size,
            vov_style=vov_style,
        )
        self.detect = Detect(nc, self.neck.out_channels, anchors)

    @property
    def nc(self) -> int:
        return self.detect.nc

    def forward(self, x: torch.Tensor):
        return self.detect(list(self.neck(*self.backbone(x))))

