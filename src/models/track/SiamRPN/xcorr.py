"""互相关算子：depthwise / group-conv 加速版，对齐 pysot ``core/xcorr.py``。"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def xcorr_fast(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """用 group conv 实现 batch 互相关。

    Args:
        x: 搜索特征 ``(B, C, H, W)``
        kernel: 模板核 ``(B, C_out, kH, kW)``，其中 ``C_out`` 可为 ``C`` 的倍数
    """
    batch = kernel.size(0)
    pk = kernel.view(-1, x.size(1), kernel.size(2), kernel.size(3))
    px = x.view(1, -1, x.size(2), x.size(3))
    po = F.conv2d(px, pk, groups=batch)
    return po.view(batch, -1, po.size(2), po.size(3))


def xcorr_depthwise(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """逐通道 depthwise 互相关（SiamRPN++ 默认）。

    Args:
        x: 搜索特征 ``(B, C, H, W)``
        kernel: 模板特征 ``(B, C, kH, kW)``
    """
    batch, channel = kernel.size(0), kernel.size(1)
    x = x.view(1, batch * channel, x.size(2), x.size(3))
    kernel = kernel.view(batch * channel, 1, kernel.size(2), kernel.size(3))
    out = F.conv2d(x, kernel, groups=batch * channel)
    return out.view(batch, channel, out.size(2), out.size(3))
