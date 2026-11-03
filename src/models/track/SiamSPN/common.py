"""SiamSPN++ 基础模块：Adjust 通道对齐、DepthwiseXCorr、DepthwiseRPN、MultiRPN。

对齐 pysot ``models/neck`` + ``models/head/rpn`` 中 SiamRPN++ 的实现。
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.track.SiamSPN.xcorr import xcorr_depthwise, xcorr_fast


class AdjustLayer(nn.Module):
    """1x1 降维到统一通道；模板特征若空间尺寸较小则裁中心区域。"""

    def __init__(self, in_channels: int, out_channels: int, center_size: int = 7) -> None:
        super().__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.center_size = center_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downsample(x)
        if x.size(3) < 20:
            l = (x.size(3) - self.center_size) // 2
            r = l + self.center_size
            x = x[:, :, l:r, l:r]
        return x


class AdjustAllLayer(nn.Module):
    """对多层特征分别做 AdjustLayer。"""

    def __init__(
        self,
        in_channels: Sequence[int],
        out_channels: Sequence[int],
        center_size: int = 7,
    ) -> None:
        super().__init__()
        if len(in_channels) != len(out_channels):
            raise ValueError("in_channels 与 out_channels 长度需一致")
        self.num = len(out_channels)
        self.layers = nn.ModuleList(
            [AdjustLayer(cin, cout, center_size) for cin, cout in zip(in_channels, out_channels)]
        )

    def forward(self, features: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        if len(features) != self.num:
            raise ValueError(f"期望 {self.num} 层特征，实际 {len(features)}")
        return [layer(f) for layer, f in zip(self.layers, features)]


class DepthwiseXCorr(nn.Module):
    """模板 / 搜索各自 3x3 卷积后做 depthwise 互相关，再接 1x1 头。"""

    def __init__(
        self,
        in_channels: int,
        hidden: int,
        out_channels: int,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        self.conv_kernel = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=kernel_size, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )
        self.conv_search = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=kernel_size, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, out_channels, kernel_size=1),
        )

    def forward(self, kernel: torch.Tensor, search: torch.Tensor) -> torch.Tensor:
        kernel = self.conv_kernel(kernel)
        search = self.conv_search(search)
        return self.head(xcorr_depthwise(search, kernel))


class DepthwiseRPN(nn.Module):
    """单层 Depthwise Siamese RPN：分类 + 回归。"""

    def __init__(self, anchor_num: int = 5, in_channels: int = 256, out_channels: int = 256) -> None:
        super().__init__()
        self.cls = DepthwiseXCorr(in_channels, out_channels, 2 * anchor_num)
        self.loc = DepthwiseXCorr(in_channels, out_channels, 4 * anchor_num)

    def forward(self, z_f: torch.Tensor, x_f: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.cls(z_f, x_f), self.loc(z_f, x_f)


class UPChannelRPN(nn.Module):
    """原始 SiamRPN 的升通道互相关头（可选，非 ++ 默认）。"""

    def __init__(self, anchor_num: int = 5, feature_in: int = 256) -> None:
        super().__init__()
        cls_out, loc_out = 2 * anchor_num, 4 * anchor_num
        self.template_cls_conv = nn.Conv2d(feature_in, feature_in * cls_out, kernel_size=3)
        self.template_loc_conv = nn.Conv2d(feature_in, feature_in * loc_out, kernel_size=3)
        self.search_cls_conv = nn.Conv2d(feature_in, feature_in, kernel_size=3)
        self.search_loc_conv = nn.Conv2d(feature_in, feature_in, kernel_size=3)
        self.loc_adjust = nn.Conv2d(loc_out, loc_out, kernel_size=1)

    def forward(self, z_f: torch.Tensor, x_f: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cls = xcorr_fast(self.search_cls_conv(x_f), self.template_cls_conv(z_f))
        loc = self.loc_adjust(xcorr_fast(self.search_loc_conv(x_f), self.template_loc_conv(z_f)))
        return cls, loc


class MultiRPN(nn.Module):
    """多层 DepthwiseRPN + 等权 / 可学习加权融合（SiamSPN++ / SiamRPN++）。"""

    def __init__(
        self,
        anchor_num: int,
        in_channels: Sequence[int],
        weighted: bool = True,
    ) -> None:
        super().__init__()
        self.weighted = weighted
        self.rpns = nn.ModuleList(
            [DepthwiseRPN(anchor_num, c, c) for c in in_channels]
        )
        if weighted:
            self.cls_weight = nn.Parameter(torch.ones(len(in_channels)))
            self.loc_weight = nn.Parameter(torch.ones(len(in_channels)))

    def forward(
        self,
        z_fs: Sequence[torch.Tensor],
        x_fs: Sequence[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cls_list, loc_list = [], []
        for rpn, z_f, x_f in zip(self.rpns, z_fs, x_fs):
            c, l = rpn(z_f, x_f)
            cls_list.append(c)
            loc_list.append(l)

        if self.weighted:
            cw = F.softmax(self.cls_weight, dim=0)
            lw = F.softmax(self.loc_weight, dim=0)
            cls = sum(c * w for c, w in zip(cls_list, cw))
            loc = sum(l * w for l, w in zip(loc_list, lw))
        else:
            cls = sum(cls_list) / len(cls_list)
            loc = sum(loc_list) / len(loc_list)
        return cls, loc
