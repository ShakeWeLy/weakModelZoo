# 朴素 Swin Transformer 4D：在 3D 上再扩一维，特征布局 [B, T, H, W, D, C]
# 核心不变：局部窗口注意力 + cyclic shift 跨窗交互 + PatchMerging 下采样
# 注意：相对位置表 (2w-1)^4、窗口体积 ws^4，算力/显存比 3D 再高一个数量级

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def window_partition_4D(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """
    是什么：把 4D 特征切成互不重叠的超立方体窗口。
    为什么：全局自注意力是 O((THWD)^2)，窗口内注意力降到 O(窗口体积^2 * 窗口数)。
    做了什么：[B, T, H, W, D, C] -> [B * num_windows, ws^4, C]
    """
    batch_size, time, height, width, depth, channels = x.shape
    x = x.view(
        batch_size,
        time // window_size,
        window_size,
        height // window_size,
        window_size,
        width // window_size,
        window_size,
        depth // window_size,
        window_size,
        channels,
    )
    # 四个「窗口网格」轴靠前，四个「窗内坐标」轴靠后，再展平成 token 序列
    windows = x.permute(0, 1, 3, 5, 7, 2, 4, 6, 8, 9).contiguous()
    return windows.view(-1, window_size ** 4, channels)


def window_reverse_4D(
    windows: torch.Tensor,
    window_size: int,
    time: int,
    height: int,
    width: int,
    depth: int,
    batch_size: int,
) -> torch.Tensor:
    """
    是什么：window_partition_4D 的逆操作。
    为什么：窗口内算完注意力后要拼回完整 4D 网格，才能做残差 / 下一层。
    做了什么：[B * num_windows, ws^4, C] -> [B, T, H, W, D, C]
    """
    channels = windows.shape[-1]
    x = windows.view(
        batch_size,
        time // window_size,
        height // window_size,
        width // window_size,
        depth // window_size,
        window_size,
        window_size,
        window_size,
        window_size,
        channels,
    )
    # 与 partition 的 permute 互逆：交错还原 T/H/W/D
    x = x.permute(0, 1, 5, 2, 6, 3, 7, 4, 8, 9).contiguous()
    return x.view(batch_size, time, height, width, depth, channels)


class WindowAttention4D(nn.Module):
    """
    是什么：单个 4D 窗口内的多头自注意力（W-MSA / SW-MSA 共用）。
    为什么：相对位置偏置表大小为 (2w-1)^4，是 4D Swin 的主要参数/查表开销之一。
    做了什么：输入输出均为窗口 token 序列 [num_windows*B, ws^4, C]。
    """

    def __init__(
        self,
        embed_dim: int,
        window_size: int,
        num_heads: int,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = embed_dim // num_heads
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim 必须能被 num_heads 整除")
        self.scale = qk_scale or head_dim ** -0.5

        # 每轴相对偏移 -(w-1)~+(w-1) 共 2w-1 种；四维组合 (2w-1)^4
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) ** 4, num_heads)
        )

        coords = torch.stack(
            torch.meshgrid(
                torch.arange(window_size),
                torch.arange(window_size),
                torch.arange(window_size),
                torch.arange(window_size),
                indexing="ij",
            )
        )  # [4, w, w, w, w]
        coords_flatten = torch.flatten(coords, 1)  # [4, w^4]
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # [w^4, w^4, 4]

        # 平移非负后，用混合进制压成一维索引
        for axis in range(4):
            relative_coords[:, :, axis] += window_size - 1
        base = 2 * window_size - 1
        relative_coords[:, :, 0] *= base ** 3
        relative_coords[:, :, 1] *= base ** 2
        relative_coords[:, :, 2] *= base
        self.register_buffer(
            "relative_position_index",
            relative_coords.sum(-1),
            persistent=False,
        )

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, seq_len, channels = x.shape
        qkv = self.qkv(x).reshape(
            batch_size, seq_len, 3, self.num_heads, channels // self.num_heads
        ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv
        attn = (q * self.scale) @ k.transpose(-2, -1)

        bias = self.relative_position_bias_table[
            self.relative_position_index.reshape(-1)
        ].reshape(seq_len, seq_len, self.num_heads).permute(2, 0, 1)
        attn = attn + bias.unsqueeze(0)

        if attn_mask is not None:
            num_windows = attn_mask.shape[0]
            attn = attn.reshape(
                batch_size // num_windows, num_windows, self.num_heads, seq_len, seq_len
            )
            attn = (attn + attn_mask.unsqueeze(1).unsqueeze(0)).reshape(
                -1, self.num_heads, seq_len, seq_len
            )

        attn = self.attn_drop(F.softmax(attn, dim=-1))
        out = (attn @ v).transpose(1, 2).reshape(batch_size, seq_len, channels)
        return self.proj_drop(self.proj(out))


def build_shifted_window_attn_mask_4D(
    time: int,
    height: int,
    width: int,
    depth: int,
    window_size: int,
    shift_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    是什么：shifted window 的注意力掩码，shape [num_windows, ws^4, ws^4]。
    为什么：cyclic shift 会把原本不相邻的区域硬凑进同一窗口；
           mask 禁止这些「假邻居」互相 attention。
    做了什么：给超体素打 region id，同窗内 id 不同则填 -100。
    """
    img_mask = torch.zeros(1, time, height, width, depth, 1, device=device, dtype=dtype)
    # 每轴最多切 3 段 → 4D 最多 3^4=81 个 region
    axis_slices = (
        slice(0, -window_size),
        slice(-window_size, -shift_size),
        slice(-shift_size, None),
    )
    region_id = 0
    for time_slice in axis_slices:
        for height_slice in axis_slices:
            for width_slice in axis_slices:
                for depth_slice in axis_slices:
                    img_mask[:, time_slice, height_slice, width_slice, depth_slice, :] = (
                        region_id
                    )
                    region_id += 1

    mask_windows = window_partition_4D(img_mask, window_size).squeeze(-1)
    attn_mask = mask_windows.unsqueeze(2) - mask_windows.unsqueeze(1)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(
        attn_mask == 0, 0.0
    )
    return attn_mask


class WindowAttentionLayer4D(nn.Module):
    """
    是什么：普通 4D 窗口注意力（W-MSA），无 shift。
    为什么：偶数层先在固定窗口内聚合局部信息。
    做了什么：partition → WindowAttention4D → reverse；输入输出 [B, T, H, W, D, C]。
    """

    def __init__(self, embed_dim: int, window_size: int, num_heads: int, **kwargs):
        super().__init__()
        self.window_size = window_size
        self.attention = WindowAttention4D(embed_dim, window_size, num_heads, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, time, height, width, depth, _ = x.shape
        if (
            time % self.window_size != 0
            or height % self.window_size != 0
            or width % self.window_size != 0
            or depth % self.window_size != 0
        ):
            raise ValueError(
                f"T/H/W/D 必须能被 window_size 整除，当前为 "
                f"({time}, {height}, {width}, {depth}), window_size={self.window_size}"
            )
        windows = window_partition_4D(x, self.window_size)
        windows = self.attention(windows, attn_mask=None)
        return window_reverse_4D(
            windows, self.window_size, time, height, width, depth, batch_size
        )


class ShiftedWindowAttention4D(nn.Module):
    """
    是什么：4D 移位窗口注意力（SW-MSA）。
    为什么：纯窗口注意力窗口间不通信；四维 cyclic shift 让相邻窗口边界上的 token 进入同一窗。
    做了什么：roll 移位 → 带 mask 的窗口注意力 → reverse → roll 还原。
    """

    def __init__(
        self,
        embed_dim: int,
        window_size: int,
        num_heads: int,
        shift_size: int | None = None,
        **kwargs,
    ):
        super().__init__()
        self.window_size = window_size
        self.shift_size = shift_size if shift_size is not None else window_size // 2
        self.attention = WindowAttention4D(embed_dim, window_size, num_heads, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, time, height, width, depth, _ = x.shape
        if (
            time % self.window_size != 0
            or height % self.window_size != 0
            or width % self.window_size != 0
            or depth % self.window_size != 0
        ):
            raise ValueError(
                f"T/H/W/D 必须能被 window_size 整除，当前为 "
                f"({time}, {height}, {width}, {depth}), window_size={self.window_size}"
            )

        shifted_x = torch.roll(
            x,
            shifts=(-self.shift_size,) * 4,
            dims=(1, 2, 3, 4),
        )
        attn_mask = build_shifted_window_attn_mask_4D(
            time,
            height,
            width,
            depth,
            self.window_size,
            self.shift_size,
            x.device,
            x.dtype,
        )

        windows = window_partition_4D(shifted_x, self.window_size)
        windows = self.attention(windows, attn_mask=attn_mask)
        shifted_x = window_reverse_4D(
            windows, self.window_size, time, height, width, depth, batch_size
        )
        return torch.roll(
            shifted_x,
            shifts=(self.shift_size,) * 4,
            dims=(1, 2, 3, 4),
        )


class Mlp(nn.Module):
    """两层 MLP（Linear → GELU → Dropout → Linear → Dropout），作用在最后一维通道上。"""

    def __init__(self, embed_dim: int, mlp_ratio: float = 4.0, drop: float = 0.0):
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return self.drop(x)


class PatchMerging4D(nn.Module):
    """
    是什么：4D Patch Merging 下采样。
    为什么：类比 stride-2，扩大感受野并降分辨率；通道翻倍保留容量。
    做了什么：拼 2×2×2×2 邻域（16C）→ LN → Linear 到 2C；T/H/W/D 各减半。
    """

    def __init__(self, embed_dim: int, ratio: int = 2):
        super().__init__()
        if ratio != 2:
            raise ValueError("当前 PatchMerging4D 仅支持 ratio=2")
        self.embed_dim = embed_dim
        self.out_dim = embed_dim * 2
        self.norm = nn.LayerNorm(embed_dim * (ratio ** 4))
        self.reduction = nn.Linear(embed_dim * (ratio ** 4), self.out_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, time, height, width, depth, channels = x.shape
        if time % 2 != 0 or height % 2 != 0 or width % 2 != 0 or depth % 2 != 0:
            raise ValueError(
                f"T/H/W/D 必须为偶数，当前为 ({time}, {height}, {width}, {depth})"
            )
        if channels != self.embed_dim:
            raise ValueError(f"输入通道应为 {self.embed_dim}，当前为 {channels}")

        # 16 个相位，对应 2×2×2×2 超立方体的全部角点
        parts = []
        for t in (0, 1):
            for h in (0, 1):
                for w in (0, 1):
                    for d in (0, 1):
                        parts.append(x[:, t::2, h::2, w::2, d::2, :])
        x = torch.cat(parts, dim=-1)
        x = self.norm(x)
        return self.reduction(x)


class PatchExpanding4D(nn.Module):
    """
    是什么：PatchMerging4D 的逆过程（上采样）。
    为什么：解码器需要把 4D 分辨率抬回去。
    做了什么：Linear 扩通道 → 拆成 2×2×2×2 空间 → 通道减半。
    """

    def __init__(self, embed_dim: int, ratio: int = 2):
        super().__init__()
        if ratio != 2:
            raise ValueError("当前 PatchExpanding4D 仅支持 ratio=2")
        self.embed_dim = embed_dim
        self.out_dim = embed_dim // 2
        # 16 * out_dim = 8 * embed_dim
        self.expand = nn.Linear(embed_dim, embed_dim * (ratio ** 3), bias=False)
        self.norm = nn.LayerNorm(self.out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, time, height, width, depth, channels = x.shape
        if channels != self.embed_dim:
            raise ValueError(f"输入通道应为 {self.embed_dim}，当前为 {channels}")

        x = self.expand(x)
        x = x.view(batch_size, time, height, width, depth, 2, 2, 2, 2, self.out_dim)
        x = x.permute(0, 1, 5, 2, 6, 3, 7, 4, 8, 9).contiguous()
        x = x.view(
            batch_size,
            time * 2,
            height * 2,
            width * 2,
            depth * 2,
            self.out_dim,
        )
        return self.norm(x)


class DropPath(nn.Module):
    """Stochastic Depth：训练时按样本随机丢弃整条残差分支。"""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        return x.div(keep_prob) * random_tensor.floor()


class SwinTransformerBlock4D(nn.Module):
    """
    是什么：Swin 4D 基本块。
    为什么：Pre-Norm + 残差稳定深层训练；W-MSA / SW-MSA 交替实现局部+跨窗建模。
    做了什么：LN → W-MSA/SW-MSA → 残差 → LN → MLP → 残差。
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        window_size: int,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        attn_kwargs = dict(
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        if shift_size > 0:
            self.attn: nn.Module = ShiftedWindowAttention4D(
                embed_dim=embed_dim,
                window_size=window_size,
                num_heads=num_heads,
                shift_size=shift_size,
                **attn_kwargs,
            )
        else:
            self.attn = WindowAttentionLayer4D(
                embed_dim=embed_dim,
                window_size=window_size,
                num_heads=num_heads,
                **attn_kwargs,
            )
        self.drop_path = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = Mlp(embed_dim=embed_dim, mlp_ratio=mlp_ratio, drop=drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


if __name__ == "__main__":
    # 4D 体积涨得快，冒烟用小分辨率 + 小窗口
    x = torch.randn(1, 4, 4, 4, 4, 32)
    block = SwinTransformerBlock4D(embed_dim=32, num_heads=4, window_size=2, shift_size=0)
    shifted_block = SwinTransformerBlock4D(
        embed_dim=32, num_heads=4, window_size=2, shift_size=1
    )
    merge = PatchMerging4D(embed_dim=32)
    expand = PatchExpanding4D(embed_dim=64)

    print("block:", block(x).shape)
    print("shifted block:", shifted_block(x).shape)
    merged = merge(x)
    print("patch merge:", merged.shape)
    print("patch expand:", expand(merged).shape)
