# 朴素 Swin Transformer 3D：在 2D Swin 上把空间从 (H, W) 扩到 (H, W, D)
# 核心不变：局部窗口注意力 + cyclic shift 跨窗交互 + PatchMerging 下采样

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def window_partition_3D(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """
    是什么：把整幅 3D 特征图切成互不重叠的立方体窗口。
    为什么：全局自注意力是 O((HWD)^2)，窗口内注意力降到 O(窗口体积^2 * 窗口数)。
    做了什么：[B, H, W, D, C] -> [B * num_windows, ws^3, C]
    """
    batch_size, height, width, depth, channels = x.shape
    # 先按每个轴拆成「窗口网格 + 窗口内坐标」
    x = x.view(
        batch_size,
        height // window_size,
        window_size,
        width // window_size,
        window_size,
        depth // window_size,
        window_size,
        channels,
    )
    # 把三个窗口轴凑到前面，方便一次性展平成 token 序列
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
    return windows.view(-1, window_size ** 3, channels)


def window_reverse_3D(
    windows: torch.Tensor,
    window_size: int,
    height: int,
    width: int,
    depth: int,
    batch_size: int,
) -> torch.Tensor:
    """
    是什么：window_partition_3D 的逆操作。
    为什么：注意力在窗口内算完后，要拼回完整体素网格才能做残差 / 下一层。
    做了什么：[B * num_windows, ws^3, C] -> [B, H, W, D, C]
    """
    channels = windows.shape[-1]
    x = windows.view(
        batch_size,
        height // window_size,
        width // window_size,
        depth // window_size,
        window_size,
        window_size,
        window_size,
        channels,
    )
    # 与 partition 的 permute 互逆：交错还原 H/W/D
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()
    return x.view(batch_size, height, width, depth, channels)


class WindowAttention3D(nn.Module):
    """
    是什么：单个窗口内的多头自注意力（W-MSA / SW-MSA 共用）。
    为什么：3D 相对位置偏置表大小为 (2w-1)^3，比 2D 的 (2w-1)^2 大一个数量级。
    做了什么：输入输出均为窗口 token 序列 [num_windows*B, ws^3, C]。
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

        # 相对位置偏置表：相对偏移在每个轴上是 -(w-1)~+(w-1)，共 2w-1 种；
        # 三维组合共 (2w-1)^3 种，每个 head 一份可学习偏置。
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) ** 3, num_heads)
        )

        # 预计算「任意两 token 的相对坐标 -> 偏置表下标」
        coords = torch.stack(
            torch.meshgrid(
                torch.arange(window_size),
                torch.arange(window_size),
                torch.arange(window_size),
                indexing="ij",
            )
        )  # [3, w, w, w]
        coords_flatten = torch.flatten(coords, 1)  # [3, w^3]
        # relative_coords[axis, i, j] = coords_i - coords_j
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # token_i - token_j, 第一维的0,1,2 代表xyz, 第二维和第三维是两个token
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # [w^3, w^3, 3]  w^3就是3D窗口内的每一个位置的token, 只是被展平了, 第三维是三个轴的差值

        # 平移成非负，再用混合进制压成一维索引
        relative_coords[:, :, 0] += window_size - 1  # 最小为上面说的 -(w-1), 所以逐行加上
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 2] += window_size - 1

        relative_coords[:, :, 0] *= (2 * window_size - 1) ** 2  # 偏置表是一维的, 为了区分xyz是不同位置, 类似进位
        relative_coords[:, :, 1] *= 2 * window_size - 1
        self.register_buffer(
            "relative_position_index",
            relative_coords.sum(-1),
            persistent=False,
        )

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)  # 初始化

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, seq_len, channels = x.shape
        qkv = self.qkv(x).reshape(
            batch_size, seq_len, 3, self.num_heads, channels // self.num_heads
        ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv
        attn = (q * self.scale) @ k.transpose(-2, -1)

        # 查表得到可学习的相对位置偏置并加到注意力分数上
        bias = self.relative_position_bias_table[
            self.relative_position_index.reshape(-1)  # 展平, 相当于把相对坐标映射到偏置表的下标
        ].reshape(seq_len, seq_len, self.num_heads).permute(2, 0, 1)
        attn = attn + bias.unsqueeze(0)

        if attn_mask is not None:
            # mask: [num_windows, N, N]；同一 window 内 region 不同时禁止互看
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


def build_shifted_window_attn_mask_3D(
    height: int,
    width: int,
    depth: int,
    window_size: int,
    shift_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    是什么：shifted window 的注意力掩码，shape [num_windows, ws^3, ws^3]。
    为什么：cyclic shift 会把原本不相邻的区域硬凑进同一窗口；
           mask 禁止这些「假邻居」互相 attention，只保留真正相邻的交互。
    做了什么：给体素打 region id，同窗内 id 不同则填 -100（softmax 后≈0）。
    """
    img_mask = torch.zeros(1, height, width, depth, 1, device=device, dtype=dtype)
    # 每个轴被 shift 切成最多 3 段 → 3D 最多 27 个 region
    axis_slices = (
        slice(0, -window_size),
        slice(-window_size, -shift_size),
        slice(-shift_size, None),
    )
    region_id = 0
    for height_slice in axis_slices:
        for width_slice in axis_slices:
            for depth_slice in axis_slices:
                img_mask[:, height_slice, width_slice, depth_slice, :] = region_id
                region_id += 1

    mask_windows = window_partition_3D(img_mask, window_size).squeeze(-1)
    attn_mask = mask_windows.unsqueeze(2) - mask_windows.unsqueeze(1)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(
        attn_mask == 0, 0.0
    )
    return attn_mask


class WindowAttentionLayer3D(nn.Module):
    """
    是什么：普通窗口注意力（W-MSA），无 shift。
    为什么：作为 Swin block 的偶数层，先在固定窗口内聚合局部信息。
    做了什么：partition → WindowAttention3D → reverse，输入输出 [B, H, W, D, C]。
    """

    def __init__(self, embed_dim: int, window_size: int, num_heads: int, **kwargs):
        super().__init__()
        self.window_size = window_size
        self.attention = WindowAttention3D(embed_dim, window_size, num_heads, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, height, width, depth, _ = x.shape
        if (
            height % self.window_size != 0
            or width % self.window_size != 0
            or depth % self.window_size != 0
        ):
            raise ValueError(
                f"H/W/D 必须能被 window_size 整除，当前为 ({height}, {width}, {depth}), "
                f"window_size={self.window_size}"
            )
        windows = window_partition_3D(x, self.window_size)
        windows = self.attention(windows, attn_mask=None)
        return window_reverse_3D(
            windows, self.window_size, height, width, depth, batch_size
        )


class ShiftedWindowAttention3D(nn.Module):
    """
    是什么：移位窗口注意力（SW-MSA）。
    为什么：纯窗口注意力窗口间不通信；cyclic shift 让相邻窗口边界上的 token 进入同一窗。
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
        self.attention = WindowAttention3D(embed_dim, window_size, num_heads, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, height, width, depth, _ = x.shape
        if (
            height % self.window_size != 0
            or width % self.window_size != 0
            or depth % self.window_size != 0
        ):
            raise ValueError(
                f"H/W/D 必须能被 window_size 整除，当前为 ({height}, {width}, {depth}), "
                f"window_size={self.window_size}"
            )

        # 沿 H/W/D 同时 cyclic shift，使窗口边界错位
        shifted_x = torch.roll(
            x,
            shifts=(-self.shift_size, -self.shift_size, -self.shift_size),
            dims=(1, 2, 3),
        )
        attn_mask = build_shifted_window_attn_mask_3D(
            height,
            width,
            depth,
            self.window_size,
            self.shift_size,
            x.device,
            x.dtype,
        )

        windows = window_partition_3D(shifted_x, self.window_size)
        windows = self.attention(windows, attn_mask=attn_mask)
        shifted_x = window_reverse_3D(
            windows, self.window_size, height, width, depth, batch_size
        )
        return torch.roll(
            shifted_x,
            shifts=(self.shift_size, self.shift_size, self.shift_size),
            dims=(1, 2, 3),
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


class PatchMerging3D(nn.Module):
    """
    是什么：3D Patch Merging 下采样。
    为什么：类似 CNN 的 stride-2，扩大感受野并降低分辨率；通道翻倍以保留信息容量。
    做了什么：拼 2×2×2 邻域（8C）→ LN → Linear 到 2C；空间各轴减半。
    """

    def __init__(self, embed_dim: int, ratio: int = 2):
        super().__init__()
        if ratio != 2:
            raise ValueError("当前 PatchMerging3D 仅支持 ratio=2")
        self.embed_dim = embed_dim
        self.out_dim = embed_dim * 2
        # 8 个邻域 token 拼接后再投影
        self.norm = nn.LayerNorm(embed_dim * (ratio ** 3))
        self.reduction = nn.Linear(embed_dim * (ratio ** 3), self.out_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, height, width, depth, channels = x.shape
        if height % 2 != 0 or width % 2 != 0 or depth % 2 != 0:
            raise ValueError(f"H/W/D 必须为偶数，当前为 ({height}, {width}, {depth})")
        if channels != self.embed_dim:
            raise ValueError(f"输入通道应为 {self.embed_dim}，当前为 {channels}")

        # 8 个相位的子采样，对应 2×2×2 立方体的 8 个角点
        x0 = x[:, 0::2, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, 0::2, :]
        x3 = x[:, 1::2, 1::2, 0::2, :]
        x4 = x[:, 0::2, 0::2, 1::2, :]
        x5 = x[:, 1::2, 0::2, 1::2, :]
        x6 = x[:, 0::2, 1::2, 1::2, :]
        x7 = x[:, 1::2, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], dim=-1)
        x = self.norm(x)
        return self.reduction(x)


class PatchExpanding3D(nn.Module):
    """
    是什么：PatchMerging3D 的逆过程（上采样）。
    为什么：解码器 / U-Net 风格需要把分辨率抬回去。
    做了什么：Linear 扩通道 → 拆成 2×2×2 空间 → 通道减半。
    """

    def __init__(self, embed_dim: int, ratio: int = 2):
        super().__init__()
        if ratio != 2:
            raise ValueError("当前 PatchExpanding3D 仅支持 ratio=2")
        self.embed_dim = embed_dim
        self.out_dim = embed_dim // 2
        # 需要 8 * out_dim = 4 * embed_dim 的通道预算
        self.expand = nn.Linear(embed_dim, embed_dim * (ratio ** 2), bias=False)
        self.norm = nn.LayerNorm(self.out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, height, width, depth, channels = x.shape
        if channels != self.embed_dim:
            raise ValueError(f"输入通道应为 {self.embed_dim}，当前为 {channels}")

        x = self.expand(x)
        x = x.view(batch_size, height, width, depth, 2, 2, 2, self.out_dim)
        # 把三个「子像素」轴分别插进 H/W/D
        x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()
        x = x.view(batch_size, height * 2, width * 2, depth * 2, self.out_dim)
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


class SwinTransformerBlock3D(nn.Module):
    """
    是什么：Swin 3D 基本块。
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
            self.attn: nn.Module = ShiftedWindowAttention3D(
                embed_dim=embed_dim,
                window_size=window_size,
                num_heads=num_heads,
                shift_size=shift_size,
                **attn_kwargs,
            )
        else:
            self.attn = WindowAttentionLayer3D(
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
    x = torch.randn(2, 8, 8, 8, 64)
    block = SwinTransformerBlock3D(embed_dim=64, num_heads=4, window_size=4, shift_size=0)
    shifted_block = SwinTransformerBlock3D(
        embed_dim=64, num_heads=4, window_size=4, shift_size=2
    )
    merge = PatchMerging3D(embed_dim=64)
    expand = PatchExpanding3D(embed_dim=128)

    print("block:", block(x).shape)
    print("shifted block:", shifted_block(x).shape)
    merged = merge(x)
    print("patch merge:", merged.shape)
    print("patch expand:", expand(merged).shape)
