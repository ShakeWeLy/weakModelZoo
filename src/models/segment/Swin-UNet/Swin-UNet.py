"""
Swin-UNet: 以 Swin Transformer Block 替换 UNet 中的卷积块。

编码器：Swin Block + PatchMerging 下采样
解码器：PatchExpanding 上采样 + skip concat + Swin Block
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from src.module.transformer.vit.swinTransformer import (
    FinalPatchExpanding,
    PatchExpanding,
    PatchMerging,
    SwinTransformerBlock,
)


@dataclass
class SwinUNetConfig:
    in_channels: int = 3
    out_channels: int = 1
    embed_dim: int = 96
    depths: tuple[int, ...] = (2, 2, 2, 2)
    depths_decoder: tuple[int, ...] = (2, 2, 2, 2)
    num_heads: tuple[int, ...] = (3, 6, 12, 24)
    window_size: int = 7
    patch_size: int = 4
    mlp_ratio: float = 4.0


def _build_swin_blocks(
    embed_dim: int,
    depth: int,
    num_heads: int,
    window_size: int,
    mlp_ratio: float,
) -> nn.Sequential:
    blocks = []
    for index in range(depth):
        shift_size = 0 if index % 2 == 0 else window_size // 2
        blocks.append(
            SwinTransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=shift_size,
                mlp_ratio=mlp_ratio,
            )
        )
    return nn.Sequential(*blocks)


class SwinPatchEmbed(nn.Module):
    """将图像切分为 patch 并映射到 embed_dim，输出 [B, H, W, C]。"""

    def __init__(self, in_channels: int, embed_dim: int, patch_size: int = 4):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, height, width = x.shape
        if height % self.patch_size != 0 or width % self.patch_size != 0:
            raise ValueError(
                f"输入尺寸 ({height}, {width}) 必须能被 patch_size={self.patch_size} 整除"
            )
        x = self.proj(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        return self.norm(x)


class SwinUNetEncoderStage(nn.Module):
    """编码阶段：Swin Block + 可选 PatchMerging。"""

    def __init__(
        self,
        embed_dim: int,
        depth: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float,
        downsample: bool = True,
    ):
        super().__init__()
        self.blocks = _build_swin_blocks(
            embed_dim, depth, num_heads, window_size, mlp_ratio
        )
        self.downsample = PatchMerging(embed_dim) if downsample else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.blocks(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class SwinUNetDecoderStage(nn.Module):
    """解码阶段：skip concat + 可选 PatchExpanding + Swin Block。"""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        depth: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float,
        upsample: bool = True,
        use_skip: bool = True,
    ):
        super().__init__()
        self.use_skip = use_skip
        if use_skip:
            self.concat_proj = nn.Linear(in_dim * 2, in_dim)
        self.upsample = PatchExpanding(in_dim) if upsample else None
        self.blocks = _build_swin_blocks(
            out_dim, depth, num_heads, window_size, mlp_ratio
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None = None) -> torch.Tensor:
        if skip is not None:
            x = self.concat_proj(torch.cat([x, skip], dim=-1))
        if self.upsample is not None:
            x = self.upsample(x)
        return self.blocks(x)


class SwinUNet(nn.Module):
    """
    U 型分割网络，用 Swin Transformer Block 替换 DoubleConv。

    相比经典 UNet：
    - DoubleConv -> SwinTransformerBlock
    - MaxPool / ConvTranspose -> PatchMerging / PatchExpanding
    - skip 连接为通道拼接（concat）
    """

    def __init__(self, config: SwinUNetConfig | None = None, **kwargs):
        super().__init__()
        if config is None:
            config = SwinUNetConfig(**kwargs)
        elif kwargs:
            config = SwinUNetConfig(**{**config.__dict__, **kwargs})

        depths = tuple(config.depths)
        depths_decoder = tuple(config.depths_decoder)
        num_heads = tuple(config.num_heads)
        if len(depths) != 4 or len(depths_decoder) != 4 or len(num_heads) != 4:
            raise ValueError("depths、depths_decoder、num_heads 均需包含 4 个阶段的值")

        self.in_channels = config.in_channels
        self.out_channels = config.out_channels
        self.patch_size = config.patch_size
        self.num_stages = len(depths)

        dims = [config.embed_dim * (2 ** index) for index in range(self.num_stages)]
        dims_decoder = list(reversed(dims))

        self.patch_embed = SwinPatchEmbed(
            in_channels=config.in_channels,
            embed_dim=dims[0],
            patch_size=config.patch_size,
        )

        self.encoders = nn.ModuleList()
        for stage in range(self.num_stages):
            self.encoders.append(
                SwinUNetEncoderStage(
                    embed_dim=dims[stage],
                    depth=depths[stage],
                    num_heads=num_heads[stage],
                    window_size=config.window_size,
                    mlp_ratio=config.mlp_ratio,
                    downsample=stage < self.num_stages - 1,
                )
            )

        self.decoders = nn.ModuleList()
        for stage in range(self.num_stages):
            in_dim = dims_decoder[0] if stage <= 1 else dims_decoder[stage - 1]
            out_dim = dims_decoder[stage] if stage > 0 else dims_decoder[0]
            self.decoders.append(
                SwinUNetDecoderStage(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    depth=depths_decoder[stage],
                    num_heads=num_heads[self.num_stages - 1 - stage],
                    window_size=config.window_size,
                    mlp_ratio=config.mlp_ratio,
                    upsample=stage > 0,
                    use_skip=stage > 0,
                )
            )

        self.final_up = FinalPatchExpanding(
            embed_dim=dims_decoder[-1],
            patch_size=config.patch_size,
        )
        self.final_conv = nn.Conv2d(
            dims_decoder[-1] // config.patch_size,
            config.out_channels,
            kernel_size=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_list: list[torch.Tensor] = []
        x = self.patch_embed(x)

        for encoder in self.encoders:
            skip_list.append(x)
            x = encoder(x)

        for index, decoder in enumerate(self.decoders):
            skip = skip_list[-index] if index > 0 else None
            x = decoder(x, skip)

        x = self.final_up(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return self.final_conv(x)


if __name__ == "__main__":
    model = SwinUNet(
        in_channels=3,
        out_channels=2,
        embed_dim=96,
        depths=(2, 2, 2, 2),
        depths_decoder=(2, 2, 2, 2),
    )
    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    print(f"input:  {tuple(x.shape)}")
    print(f"output: {tuple(y.shape)}")
