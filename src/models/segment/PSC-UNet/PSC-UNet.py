"""
PSC-UNet: 并行 Swin-Transformer + ResNet-34 编码，FF 融合，DFCM / WBEM 增强。
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.module.conv import BaseConv
from src.module.conv.SnakeConv.SnakeConv import SnakeConv
from src.module.transformer.vit.swinTransformer import PatchMerging, SwinTransformerBlock
from src.module.wavelet import haar_dwt
from src.models.segment.unet.unet import DoubleConv, OutConv


def _nchw_to_bhwc(x: torch.Tensor) -> torch.Tensor:
    return x.permute(0, 2, 3, 1).contiguous()


def _bhwc_to_nchw(x: torch.Tensor) -> torch.Tensor:
    return x.permute(0, 3, 1, 2).contiguous()


class FF(nn.Module):
    """双分支特征融合：拼接 -> 通道注意力 -> 投影回 stage 维度。"""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(dim*2, dim*2),  # 因为进行了concat
            nn.Sigmoid()  # 如果是注意力, 需要sigmoid激活函数
        )
        self.proj = BaseConv(dim * 2, dim, kernel_size=1, stride=1, padding=0)  # 返回正常维度输出

    def forward(self, conv_x: torch.Tensor, swin_x: torch.Tensor) -> torch.Tensor:
        x = torch.cat([conv_x, swin_x], dim=1)
        batch, channels, _, _ = x.shape
        att = self.mlp(self.gap(x).view(batch, channels)).view(batch, channels, 1, 1)  # view: [B, C, W, H] -> [B, C](进行通道注意力) -> [B, C, W, H]
        return self.proj(x * att)


class DFCM(nn.Module):
    """细节信息捕捉模块：标准卷积 + x/y 方向动态蛇形卷积。"""

    def __init__(self, dim: int, fuse: bool = True) -> None:
        super().__init__()
        self.conv = BaseConv(dim, dim, kernel_size=3, stride=1, padding=1)
        self.dsconv_x = SnakeConv(dim, dim, kernel_size=3, morph=0)
        self.dsconv_y = SnakeConv(dim, dim, kernel_size=3, morph=1)
        self.fuse = BaseConv(dim * 3, dim, kernel_size=1, stride=1, padding=0) if fuse else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_base = self.conv(x)
        x_x = self.dsconv_x(x)
        x_y = self.dsconv_y(x)
        return self.fuse(torch.cat([x_base, x_x, x_y], dim=1))


class WBEM(nn.Module):
    """边界感知增强模块：DWT 分解四频带 -> 上采样 -> 残差融合。"""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.fuse = BaseConv(dim * 4, dim, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ll, lh, hl, hh = haar_dwt(x)
        f_spatial = torch.cat([ll, lh, hl, hh], dim=1)
        f_spatial = F.interpolate(
            f_spatial,
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return x + self.fuse(f_spatial)


class ResBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = BaseConv(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = BaseConv(channels, channels, kernel_size=3, stride=1, padding=1, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(self.conv2(self.conv1(x)) + x)


class ResNetBranch(nn.Module):
    """
    ResNet-34 风格分支：
    7x7 卷积 + 3x3 卷积 -> 4 个 stage，stage 间 stride=2 的 3x3 下采样，通道逐层 ×2。
    """

    def __init__(self, in_channels: int, base_dim: int, blocks_per_stage: Sequence[int] = (3, 4, 6, 3)) -> None:
        super().__init__()
        dims = [base_dim, base_dim * 2, base_dim * 4, base_dim * 8]
        self.stem = nn.Sequential(
            BaseConv(in_channels, base_dim, kernel_size=7, stride=2, padding=3),
            BaseConv(base_dim, base_dim, kernel_size=3, stride=2, padding=1),
        )
        self.stages = nn.ModuleList()
        in_dim = base_dim
        for stage_index, (out_dim, num_blocks) in enumerate(zip(dims, blocks_per_stage)):
            downsample = stage_index > 0
            self.stages.append(self._make_stage(in_dim, out_dim, num_blocks, downsample))
            in_dim = out_dim

    @staticmethod
    def _make_stage(in_channels: int, out_channels: int, num_blocks: int, downsample: bool) -> nn.Sequential:
        layers: list[nn.Module] = []
        if downsample:
            layers.append(BaseConv(in_channels, out_channels, kernel_size=3, stride=2, padding=1))
        elif in_channels != out_channels:
            layers.append(BaseConv(in_channels, out_channels, kernel_size=1, stride=1, padding=0))
        for _ in range(num_blocks):
            layers.append(ResBlock(out_channels))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.stem(x)
        features: list[torch.Tensor] = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        return features


class SwinStage(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: int,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock(
                    embed_dim=dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if block_index % 2 == 0 else window_size // 2,
                )
                for block_index in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class SwinBranch(nn.Module):
    """
    Swin-Transformer 分支：
    4x4 patch embedding -> 4 个 stage，stage 间 PatchMerging 2x 下采样，通道逐层 ×2。
    """

    def __init__(
        self,
        in_channels: int,
        base_dim: int,
        patch_size: int = 4,
        depths: Sequence[int] = (2, 2, 2, 2),
        num_heads: Sequence[int] = (2, 4, 8, 16),
        window_size: int = 7,
    ) -> None:
        super().__init__()
        if len(depths) != 4 or len(num_heads) != 4:
            raise ValueError("depths 和 num_heads 必须各包含 4 个 stage 配置")

        self.patch_embed = nn.Conv2d(in_channels, base_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(base_dim)

        dims = [base_dim, base_dim * 2, base_dim * 4, base_dim * 8]
        self.stages = nn.ModuleList(
            [
                SwinStage(dim=dims[index], depth=depths[index], num_heads=num_heads[index], window_size=window_size)
                for index in range(4)
            ]
        )
        self.merges = nn.ModuleList([PatchMerging(embed_dim=dims[index]) for index in range(3)])

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.patch_embed(x)
        x = _nchw_to_bhwc(x)
        x = self.norm(x)

        features: list[torch.Tensor] = []
        for stage_index, stage in enumerate(self.stages):
            x = stage(x)
            features.append(_bhwc_to_nchw(x))
            if stage_index < len(self.merges):
                x = self.merges[stage_index](x)
        return features


class UpNoSkip(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.up(x))


class FusionUp(nn.Module):
    """先与 skip 拼接，再 2x 上采样并卷积。"""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels + skip_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x, skip], dim=1)
        return self.conv(self.up(x))


class PSC_UNet(nn.Module):
    """
    并行双分支 U-Net：
    ResNet-34 分支 + Swin 分支 -> FF -> DFCM（前三 stage）
    Swin 最深层 -> WBEM 瓶颈 -> U-Net 解码。
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base_dim: int = 64,
        swin_depths: Sequence[int] = (2, 2, 2, 2),
    ) -> None:
        super().__init__()
        dims = [base_dim, base_dim * 2, base_dim * 4, base_dim * 8]

        self.resnet_branch = ResNetBranch(in_channels, base_dim)
        self.swin_branch = SwinBranch(in_channels, base_dim, patch_size=4, depths=swin_depths)

        self.ff_modules = nn.ModuleList([FF(dim) for dim in dims[:3]])
        self.dfcms = nn.ModuleList([DFCM(dim) for dim in dims[:3]])
        self.wbem = WBEM(dims[3])

        self.up1 = UpNoSkip(dims[3], dims[2])
        self.up2 = FusionUp(dims[2], dims[2], dims[1])
        self.up3 = FusionUp(dims[1], dims[1], dims[0])
        self.up4 = FusionUp(dims[0], dims[0], dims[0])
        self.up5 = UpNoSkip(dims[0], dims[0])
        self.outc = OutConv(dims[0], out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        resnet_feats = self.resnet_branch(x)
        swin_feats = self.swin_branch(x)

        x1 = self.dfcms[0](self.ff_modules[0](resnet_feats[0], swin_feats[0]))
        x2 = self.dfcms[1](self.ff_modules[1](resnet_feats[1], swin_feats[1]))
        x3 = self.dfcms[2](self.ff_modules[2](resnet_feats[2], swin_feats[2]))

        x5 = self.wbem(swin_feats[3])

        x = self.up1(x5)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.up5(x)
        return self.outc(x)


if __name__ == "__main__":
    model = PSC_UNet(in_channels=3, out_channels=2, base_dim=32)
    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    print(f"input:  {tuple(x.shape)}")
    print(f"output: {tuple(y.shape)}")
