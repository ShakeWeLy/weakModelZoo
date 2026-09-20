"""2D Haar 离散小波变换（DWT），纯 PyTorch 实现，支持反向传播。"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _pad_to_even(x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
    _, _, height, width = x.shape
    pad_h = height % 2
    pad_w = width % 2
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
    return x, (pad_h, pad_w)


def haar_dwt(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    对输入特征做 2D Haar 小波分解。

    Args:
        x: [B, C, H, W]

    Returns:
        LL: 低频分量 [B, C, H/2, W/2]
        LH: 水平高频 [B, C, H/2, W/2]
        HL: 垂直高频 [B, C, H/2, W/2]
        HH: 对角高频 [B, C, H/2, W/2]
    """
    x, _ = _pad_to_even(x)

    x00 = x[:, :, 0::2, 0::2]
    x01 = x[:, :, 0::2, 1::2]
    x10 = x[:, :, 1::2, 0::2]
    x11 = x[:, :, 1::2, 1::2]

    ll = (x00 + x01 + x10 + x11) * 0.5
    lh = (x00 + x01 - x10 - x11) * 0.5
    hl = (x00 - x01 + x10 - x11) * 0.5
    hh = (x00 - x01 - x10 + x11) * 0.5
    return ll, lh, hl, hh


def haar_idwt(
    ll: torch.Tensor,
    lh: torch.Tensor,
    hl: torch.Tensor,
    hh: torch.Tensor,
) -> torch.Tensor:
    """2D Haar 逆小波变换，将四个子带重建为特征图。"""
    batch, channels, height, width = ll.shape
    out = torch.zeros(
        (batch, channels, height * 2, width * 2),
        device=ll.device,
        dtype=ll.dtype,
    )

    out[:, :, 0::2, 0::2] = (ll + lh + hl + hh) * 0.5
    out[:, :, 0::2, 1::2] = (ll + lh - hl - hh) * 0.5
    out[:, :, 1::2, 0::2] = (ll - lh + hl - hh) * 0.5
    out[:, :, 1::2, 1::2] = (ll - lh - hl + hh) * 0.5
    return out
