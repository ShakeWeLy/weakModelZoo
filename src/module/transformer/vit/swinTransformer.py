import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """[B, H, W, C] -> [B * num_windows, window_size * window_size, C]"""
    batch_size, height, width, channels = x.shape
    x = x.view(
        batch_size,
        height // window_size,
        window_size,
        width // window_size,
        window_size,
        channels,
    )
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return windows.view(-1, window_size * window_size, channels)


def window_reverse(
    windows: torch.Tensor,
    window_size: int,
    height: int,
    width: int,
    batch_size: int,
) -> torch.Tensor:
    """[B * num_windows, window_size * window_size, C] -> [B, H, W, C]"""
    channels = windows.shape[-1]
    x = windows.view(
        batch_size,
        height // window_size,
        width // window_size,
        window_size,
        window_size,
        channels,
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return x.view(batch_size, height, width, channels)


class WindowAttention(nn.Module):
    """在局部窗口内做多头自注意力。输入输出均为 [B, H, W, C]。"""

    def __init__(self, embed_dim: int, window_size: int, num_heads: int,
                 qkv_bias: bool = True, qk_scale: float | None = None,
                 attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = embed_dim // num_heads
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim 必须能被 num_heads 整除")
        self.scale = qk_scale or head_dim ** -0.5
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) ** 2, num_heads)
        )
        coords = torch.stack(torch.meshgrid(
            torch.arange(window_size), torch.arange(window_size), indexing="ij"
        ))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        self.register_buffer("relative_position_index", relative_coords.sum(-1), persistent=False)
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
            attn = attn.reshape(batch_size // num_windows, num_windows,
                                self.num_heads, seq_len, seq_len)
            attn = (attn + attn_mask.unsqueeze(1).unsqueeze(0)).reshape(
                -1, self.num_heads, seq_len, seq_len
            )
        attn = self.attn_drop(F.softmax(attn, dim=-1))
        out = (attn @ v).transpose(1, 2).reshape(batch_size, seq_len, channels)
        return self.proj_drop(self.proj(out))


def build_shifted_window_attn_mask(
    height: int,
    width: int,
    window_size: int,
    shift_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    构建 shifted window 的注意力掩码，shape: [num_windows, ws*ws, ws*ws]。
    Shift 是为了让不同窗口之间能交换信息；Mask 是为了防止「被 shift 硬凑进同一窗、但在原图上并不相邻」的 token 互相 attention
    """
    img_mask = torch.zeros(1, height, width, 1, device=device, dtype=dtype)
    height_slices = (
        slice(0, -window_size),
        slice(-window_size, -shift_size),
        slice(-shift_size, None),
    )
    width_slices = (
        slice(0, -window_size),
        slice(-window_size, -shift_size),
        slice(-shift_size, None),
    )
    region_id = 0
    for height_slice in height_slices:
        for width_slice in width_slices:
            img_mask[:, height_slice, width_slice, :] = region_id
            region_id += 1

    mask_windows = window_partition(img_mask, window_size).squeeze(-1)
    attn_mask = mask_windows.unsqueeze(2) - mask_windows.unsqueeze(1)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, 0.0)
    return attn_mask


class WindowAttentionLayer(nn.Module):
    """对 [B, H, W, C] 做 W-MSA（无 cyclic shift）。"""

    def __init__(self, embed_dim: int, window_size: int, num_heads: int, **kwargs):
        super().__init__()
        self.window_size = window_size
        self.attention = WindowAttention(embed_dim, window_size, num_heads, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, height, width, _ = x.shape
        if height % self.window_size != 0 or width % self.window_size != 0:
            raise ValueError(
                f"H/W 必须能被 window_size 整除，当前为 ({height}, {width}), window_size={self.window_size}"
            )
        windows = window_partition(x, self.window_size)
        windows = self.attention(windows, attn_mask=None)
        return window_reverse(windows, self.window_size, height, width, batch_size)


class ShiftedWindowAttention(nn.Module):
    """Swin 的 Shifted Window Attention：先 cyclic shift，窗口内注意力，再 shift 回来。"""

    def __init__(self, embed_dim: int, window_size: int, num_heads: int,
                 shift_size: int | None = None, **kwargs):
        super().__init__()
        self.window_size = window_size
        self.shift_size = shift_size if shift_size is not None else window_size // 2
        self.attention = WindowAttention(embed_dim, window_size, num_heads, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, height, width, _ = x.shape
        if height % self.window_size != 0 or width % self.window_size != 0:
            raise ValueError(
                f"H/W 必须能被 window_size 整除，当前为 ({height}, {width}), window_size={self.window_size}"
            )

        shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        attn_mask = build_shifted_window_attn_mask(
            height, width, self.window_size, self.shift_size, x.device, x.dtype
        )
        num_windows = attn_mask.shape[0]
        attn_mask = attn_mask.unsqueeze(0).repeat(batch_size, 1, 1, 1).view(
            batch_size * num_windows, self.window_size * self.window_size, self.window_size * self.window_size
        )

        windows = window_partition(shifted_x, self.window_size)
        windows = self.attention(windows, attn_mask=attn_mask)
        shifted_x = window_reverse(windows, self.window_size, height, width, batch_size)
        return torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))


class Mlp(nn.Module):
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


class PatchMerging(nn.Module):
    """将 2×2 邻域 patch 合并，空间分辨率减半、通道数翻倍。输入输出均为 [B, H, W, C]。"""

    def __init__(self, embed_dim: int, ratio: int = 2):
        super().__init__()
        if ratio != 2:
            raise ValueError("当前 PatchMerging 仅支持 ratio=2")
        self.embed_dim = embed_dim
        self.out_dim = embed_dim * 2
        self.norm = nn.LayerNorm(embed_dim * ratio ** 2)
        self.reduction = nn.Linear(embed_dim * ratio ** 2, self.out_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, height, width, channels = x.shape
        if height % 2 != 0 or width % 2 != 0:
            raise ValueError(f"H/W 必须为偶数，当前为 ({height}, {width})")
        if channels != self.embed_dim:
            raise ValueError(f"输入通道应为 {self.embed_dim}，当前为 {channels}")

        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], dim=-1)
        x = self.norm(x)
        x = self.reduction(x)
        return x


class PatchExpanding(nn.Module):
    """Patch Expanding：空间分辨率 x2、通道数减半。输入输出均为 [B, H, W, C]。"""

    def __init__(self, embed_dim: int, ratio: int = 2):
        super().__init__()
        if ratio != 2:
            raise ValueError("当前 PatchExpanding 仅支持 ratio=2")
        self.embed_dim = embed_dim
        self.out_dim = embed_dim // 2
        self.expand = nn.Linear(embed_dim, embed_dim * ratio, bias=False)
        self.norm = nn.LayerNorm(self.out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, height, width, channels = x.shape
        if channels != self.embed_dim:
            raise ValueError(f"输入通道应为 {self.embed_dim}，当前为 {channels}")

        x = self.expand(x)
        x = x.view(batch_size, height, width, 2, 2, self.out_dim)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(batch_size, height * 2, width * 2, self.out_dim)
        return self.norm(x)


class FinalPatchExpanding(nn.Module):
    """最终上采样至原始分辨率。输入输出均为 [B, H, W, C]。"""

    def __init__(self, embed_dim: int, patch_size: int = 4):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.out_dim = embed_dim // patch_size
        self.expand = nn.Linear(embed_dim, embed_dim * patch_size, bias=False)
        self.norm = nn.LayerNorm(self.out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, height, width, channels = x.shape
        if channels != self.embed_dim:
            raise ValueError(f"输入通道应为 {self.embed_dim}，当前为 {channels}")

        x = self.expand(x)
        x = x.view(
            batch_size,
            height,
            width,
            self.patch_size,
            self.patch_size,
            self.out_dim,
        )
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(
            batch_size,
            height * self.patch_size,
            width * self.patch_size,
            self.out_dim,
        )
        return self.norm(x)


class DropPath(nn.Module):
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


class SwinTransformerBlock(nn.Module):
    """Swin Transformer Block: LN → W-MSA/SW-MSA → 残差 → LN → MLP → 残差。"""

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
        if shift_size > 0:
            self.attn: nn.Module = ShiftedWindowAttention(
                embed_dim=embed_dim,
                window_size=window_size,
                num_heads=num_heads,
                shift_size=shift_size,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                attn_drop=attn_drop, proj_drop=drop,
            )
        else:
            self.attn = WindowAttentionLayer(
                embed_dim=embed_dim,
                window_size=window_size,
                num_heads=num_heads,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                attn_drop=attn_drop, proj_drop=drop,
            )
        self.drop_path = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = Mlp(embed_dim=embed_dim, mlp_ratio=mlp_ratio, drop=drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


if __name__ == "__main__":
    x = torch.randn(2, 8, 8, 64)
    block = SwinTransformerBlock(embed_dim=64, num_heads=4, window_size=4, shift_size=0)
    shifted_block = SwinTransformerBlock(embed_dim=64, num_heads=4, window_size=4, shift_size=2)
    merge = PatchMerging(embed_dim=64)

    print("block:", block(x).shape)
    print("shifted block:", shifted_block(x).shape)
    print("patch merge:", merge(x).shape)
