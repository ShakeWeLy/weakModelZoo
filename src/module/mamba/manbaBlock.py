"""
VMamba / VM-UNet building blocks.

VSSBlock replaces the DoubleConv stack in UNet: LayerNorm + SS2D (2D selective scan) + residual.
Reference: VM-UNet (https://arxiv.org/abs/2402.02491), VMamba.
"""

from __future__ import annotations

import math
from functools import partial
from typing import Callable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn as _selective_scan_fn
except ImportError:
    _selective_scan_fn = None


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        return x.div(keep_prob) * random_tensor


def trunc_normal_(tensor: torch.Tensor, mean: float = 0.0, std: float = 1.0) -> torch.Tensor:
    return nn.init.trunc_normal_(tensor, mean=mean, std=std)


def selective_scan_torch(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor | None = None,
    delta_bias: torch.Tensor | None = None,
    delta_softplus: bool = True,
) -> torch.Tensor:
    dtype_in = u.dtype
    batch, kcdim, length = u.shape
    _, num_groups, state_dim, _ = B.shape
    channel_dim = kcdim // num_groups

    if delta_bias is not None:
        delta = delta + delta_bias[..., None]
    if delta_softplus:
        delta = F.softplus(delta)

    u = u.float()
    delta = delta.float()
    A = A.float()
    B = B.float()
    C = C.float()

    B = B.view(batch, num_groups, 1, state_dim, length).repeat(1, 1, channel_dim, 1, 1)
    B = B.view(batch, kcdim, state_dim, length)
    C = C.view(batch, num_groups, 1, state_dim, length).repeat(1, 1, channel_dim, 1, 1)
    C = C.view(batch, kcdim, state_dim, length)

    delta_a = torch.exp(torch.einsum("bdl,dn->bdln", delta, A))
    delta_b_u = torch.einsum("bdl,bdnl,bdl->bdln", delta, B, u)

    state = A.new_zeros((batch, kcdim, state_dim))
    outputs = []
    for index in range(length):
        state = delta_a[:, :, index, :] * state + delta_b_u[:, :, index, :]
        outputs.append(torch.einsum("bdn,bdn->bd", state, C[:, :, :, index]))
    y = torch.stack(outputs, dim=2)

    if D is not None:
        y = y + u * D.unsqueeze(-1)
    return y.to(dtype=dtype_in)


def selective_scan(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor | None = None,
    delta_bias: torch.Tensor | None = None,
    delta_softplus: bool = True,
) -> torch.Tensor:
    if _selective_scan_fn is not None and u.is_cuda:
        return _selective_scan_fn(
            u,
            delta,
            A,
            B,
            C,
            D,
            z=None,
            delta_bias=delta_bias,
            delta_softplus=delta_softplus,
            return_last_state=False,
        )
    return selective_scan_torch(
        u,
        delta,
        A,
        B,
        C,
        D,
        delta_bias=delta_bias,
        delta_softplus=delta_softplus,
    )


class SS2D(nn.Module):
    """2D Selective Scan module used inside VSSBlock."""

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 3,
        expand: float = 2.0,
        dt_rank: str | int = "auto",
        dropout: float = 0.0,
        conv_bias: bool = True,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else int(dt_rank)

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias)
        self.conv2d = nn.Conv2d(
            self.d_inner,
            self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
        )
        self.act = nn.SiLU()

        x_proj = [
            nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
            for _ in range(4)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([layer.weight for layer in x_proj], dim=0))
        del x_proj

        dt_projs = [self._dt_init(self.dt_rank, self.d_inner) for _ in range(4)]
        self.dt_projs_weight = nn.Parameter(torch.stack([layer.weight for layer in dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([layer.bias for layer in dt_projs], dim=0))
        del dt_projs

        self.A_logs = self._A_log_init(self.d_state, self.d_inner, copies=4, merge=True)
        self.Ds = self._D_init(self.d_inner, copies=4, merge=True)

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    @staticmethod
    def _dt_init(dt_rank: int, d_inner: int) -> nn.Linear:
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True)
        dt_init_std = dt_rank**-0.5
        nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)

        dt = torch.exp(
            torch.rand(d_inner) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
        ).clamp(min=1e-4)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True
        return dt_proj

    @staticmethod
    def _A_log_init(d_state: int, d_inner: int, copies: int = 1, merge: bool = True) -> nn.Parameter:
        a = torch.arange(1, d_state + 1, dtype=torch.float32).view(1, -1).repeat(d_inner, 1).contiguous()
        a_log = torch.log(a)
        if copies > 1:
            a_log = a_log[None].repeat(copies, 1, 1).contiguous()
            if merge:
                a_log = a_log.flatten(0, 1)
        param = nn.Parameter(a_log)
        param._no_weight_decay = True
        return param

    @staticmethod
    def _D_init(d_inner: int, copies: int = 1, merge: bool = True) -> nn.Parameter:
        d = torch.ones(d_inner)
        if copies > 1:
            d = d[None].repeat(copies, 1).contiguous()
            if merge:
                d = d.flatten(0, 1)
        param = nn.Parameter(d)
        param._no_weight_decay = True
        return param

    def forward_core(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, _, height, width = x.shape
        length = height * width
        num_dirs = 4

        x_hwwh = torch.stack(
            [x.view(batch, -1, length), x.transpose(2, 3).contiguous().view(batch, -1, length)],
            dim=1,
        ).view(batch, 2, -1, length)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
        dts, bs, cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)

        xs = xs.float().view(batch, -1, length)
        dts = dts.contiguous().float().view(batch, -1, length)
        bs = bs.float().view(batch, num_dirs, -1, length)
        cs = cs.float().view(batch, num_dirs, -1, length)
        ds = self.Ds.float().view(-1)
        a_s = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_bias = self.dt_projs_bias.float().view(-1)

        out_y = selective_scan(
            xs,
            dts,
            a_s,
            bs,
            cs,
            ds,
            delta_bias=dt_bias,
            delta_softplus=True,
        ).view(batch, num_dirs, -1, length)

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(batch, 2, -1, length)
        wh_y = torch.transpose(out_y[:, 1].view(batch, -1, width, height), 2, 3).contiguous().view(batch, -1, length)
        inv_wh_y = torch.transpose(inv_y[:, 1].view(batch, -1, width, height), 2, 3).contiguous().view(batch, -1, length)
        return out_y[:, 0], inv_y[:, 0], wh_y, inv_wh_y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, height, width, channels = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        y1, y2, y3, y4 = self.forward_core(x)
        y = y1 + y2 + y3 + y4
        y = y.transpose(1, 2).contiguous().view(batch, height, width, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        return self.dropout(self.out_proj(y))


class VSSBlock(nn.Module):
    """Visual State Space block: replaces DoubleConv as the feature extractor."""

    def __init__(
        self,
        hidden_dim: int,
        drop_path: float = 0.0,
        norm_layer: Callable[..., nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0.0,
        d_state: int = 16,
    ) -> None:
        super().__init__()
        self.norm = norm_layer(hidden_dim)
        self.op = SS2D(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.drop_path(self.op(self.norm(x)))


class PatchEmbed2D(nn.Module):
    def __init__(
        self,
        patch_size: int = 4,
        in_chans: int = 3,
        embed_dim: int = 96,
        norm_layer: Callable[..., nn.Module] | None = nn.LayerNorm,
    ) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer is not None else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x).permute(0, 2, 3, 1)
        return self.norm(x)


class PatchMerging2D(nn.Module):
    def __init__(self, dim: int, norm_layer: Callable[..., nn.Module] = nn.LayerNorm) -> None:
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, height, width, channels = x.shape
        shape_fix = [-1, -1]
        if (width % 2 != 0) or (height % 2 != 0):
            shape_fix = [height // 2, width // 2]

        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]

        if shape_fix[0] > 0:
            x0 = x0[:, : shape_fix[0], : shape_fix[1], :]
            x1 = x1[:, : shape_fix[0], : shape_fix[1], :]
            x2 = x2[:, : shape_fix[0], : shape_fix[1], :]
            x3 = x3[:, : shape_fix[0], : shape_fix[1], :]

        x = torch.cat([x0, x1, x2, x3], dim=-1)
        x = x.view(batch, height // 2, width // 2, 4 * channels)
        x = self.norm(x)
        return self.reduction(x)


class PatchExpand2D(nn.Module):
    def __init__(self, dim: int, dim_scale: int = 2, norm_layer: Callable[..., nn.Module] = nn.LayerNorm) -> None:
        super().__init__()
        self.dim = dim * 2
        self.dim_scale = dim_scale
        self.expand = nn.Linear(self.dim, dim_scale * self.dim, bias=False)
        self.norm = norm_layer(self.dim // dim_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, height, width, channels = x.shape
        x = self.expand(x)
        x = x.view(
            batch,
            height,
            width,
            self.dim_scale,
            self.dim_scale,
            channels // self.dim_scale,
        )
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(batch, height * self.dim_scale, width * self.dim_scale, channels // self.dim_scale)
        return self.norm(x)


class FinalPatchExpand2D(nn.Module):
    def __init__(self, dim: int, dim_scale: int = 4, norm_layer: Callable[..., nn.Module] = nn.LayerNorm) -> None:
        super().__init__()
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, dim_scale * dim, bias=False)
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, height, width, channels = x.shape
        x = self.expand(x)
        x = x.view(
            batch,
            height,
            width,
            self.dim_scale,
            self.dim_scale,
            channels // self.dim_scale,
        )
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(batch, height * self.dim_scale, width * self.dim_scale, channels // self.dim_scale)
        return self.norm(x)


class VSSLayer(nn.Module):
    """Encoder stage: several VSSBlocks followed by optional downsampling."""

    def __init__(
        self,
        dim: int,
        depth: int,
        drop_path: Sequence[float] | float = 0.0,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        downsample: Callable[..., nn.Module] | None = None,
        use_checkpoint: bool = False,
        d_state: int = 16,
    ) -> None:
        super().__init__()
        self.use_checkpoint = use_checkpoint
        if isinstance(drop_path, (int, float)):
            drop_path = [drop_path] * depth

        self.blocks = nn.ModuleList(
            [
                VSSBlock(
                    hidden_dim=dim,
                    drop_path=drop_path[index],
                    norm_layer=norm_layer,
                    d_state=d_state,
                )
                for index in range(depth)
            ]
        )
        self.downsample = downsample(dim=dim, norm_layer=norm_layer) if downsample is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class VSSLayerUp(nn.Module):
    """Decoder stage: optional upsampling followed by several VSSBlocks."""

    def __init__(
        self,
        dim: int,
        depth: int,
        drop_path: Sequence[float] | float = 0.0,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        upsample: Callable[..., nn.Module] | None = None,
        use_checkpoint: bool = False,
        d_state: int = 16,
    ) -> None:
        super().__init__()
        self.use_checkpoint = use_checkpoint
        if isinstance(drop_path, (int, float)):
            drop_path = [drop_path] * depth

        self.blocks = nn.ModuleList(
            [
                VSSBlock(
                    hidden_dim=dim,
                    drop_path=drop_path[index],
                    norm_layer=norm_layer,
                    d_state=d_state,
                )
                for index in range(depth)
            ]
        )
        self.upsample = upsample(dim=dim, norm_layer=norm_layer) if upsample is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.upsample is not None:
            x = self.upsample(x)
        for block in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        return x


def init_vmamba_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.bias, 0)
        nn.init.constant_(module.weight, 1.0)
