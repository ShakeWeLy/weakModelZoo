"""
VM-UNet: Vision Mamba UNet for segmentation.

Replaces UNet's DoubleConv blocks with VSSBlock (SS2D + residual), and uses
patch embedding / merging / expanding for down- and up-sampling.

Reference: https://arxiv.org/abs/2402.02491
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from src.module.mamba.manbaBlock import (
    FinalPatchExpand2D,
    PatchEmbed2D,
    PatchExpand2D,
    PatchMerging2D,
    VSSLayer,
    VSSLayerUp,
    init_vmamba_weights,
)


class VMUNet(nn.Module):
    """
    Asymmetric encoder-decoder UNet built from VSSBlocks.

    Compared with the classic UNet in ``unet.py``:
    - ``DoubleConv`` -> ``VSSBlock`` (global 2D selective scan instead of local 3x3 conv)
    - ``MaxPool`` / ``ConvTranspose`` -> ``PatchMerging2D`` / ``PatchExpand2D``
    - skip connections are additive (same as the VM-UNet paper)
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        hidden_channels: int = 96,
        depths: Sequence[int] = (2, 2, 2, 2),
        depths_decoder: Sequence[int] = (2, 2, 2, 2),
        patch_size: int = 4,
        d_state: int = 16,
        drop_path_rate: float = 0.1,
        use_checkpoint: bool = False,
        repeat_gray_to_rgb: bool = True,
    ) -> None:
        super().__init__()
        if len(depths) != 4 or len(depths_decoder) != 4:
            raise ValueError("depths and depths_decoder must each contain 4 stage values.")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.repeat_gray_to_rgb = repeat_gray_to_rgb
        self.num_stages = len(depths)

        dims = [hidden_channels * (2 ** index) for index in range(self.num_stages)]
        dims_decoder = list(reversed(dims))
        self.dims = dims
        embed_in_chans = 3 if (repeat_gray_to_rgb and in_channels == 1) else in_channels

        self.patch_embed = PatchEmbed2D(
            patch_size=patch_size,
            in_chans=embed_in_chans,
            embed_dim=dims[0],
            norm_layer=nn.LayerNorm,
        )

        encoder_drop_path = torch.linspace(0, drop_path_rate, sum(depths)).tolist()
        decoder_drop_path = torch.linspace(0, drop_path_rate, sum(depths_decoder)).tolist()[::-1]

        self.layers = nn.ModuleList()
        for stage in range(self.num_stages):
            self.layers.append(
                VSSLayer(
                    dim=dims[stage],
                    depth=depths[stage],
                    drop_path=encoder_drop_path[sum(depths[:stage]) : sum(depths[: stage + 1])],
                    downsample=PatchMerging2D if stage < self.num_stages - 1 else None,
                    use_checkpoint=use_checkpoint,
                    d_state=d_state,
                )
            )

        self.layers_up = nn.ModuleList()
        for stage in range(self.num_stages):
            self.layers_up.append(
                VSSLayerUp(
                    dim=dims_decoder[stage],
                    depth=depths_decoder[stage],
                    drop_path=decoder_drop_path[sum(depths_decoder[:stage]) : sum(depths_decoder[: stage + 1])],
                    upsample=PatchExpand2D if stage > 0 else None,
                    use_checkpoint=use_checkpoint,
                    d_state=d_state,
                )
            )

        self.final_up = FinalPatchExpand2D(dim=dims_decoder[-1], dim_scale=patch_size, norm_layer=nn.LayerNorm)
        self.final_conv = nn.Conv2d(dims_decoder[-1] // patch_size, out_channels, kernel_size=1)
        self.apply(init_vmamba_weights)

    def _prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        if self.repeat_gray_to_rgb and x.size(1) == 1:
            return x.repeat(1, 3, 1, 1)
        return x

    def forward_features(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        skip_list: list[torch.Tensor] = []
        x = self.patch_embed(x)
        for layer in self.layers:
            skip_list.append(x)
            x = layer(x)
        return x, skip_list

    def forward_features_up(self, x: torch.Tensor, skip_list: list[torch.Tensor]) -> torch.Tensor:
        for index, layer_up in enumerate(self.layers_up):
            if index == 0:
                x = layer_up(x)
            else:
                x = layer_up(x + skip_list[-index])
        return x

    def forward_final(self, x: torch.Tensor) -> torch.Tensor:
        x = self.final_up(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return self.final_conv(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._prepare_input(x)
        x, skip_list = self.forward_features(x)
        x = self.forward_features_up(x, skip_list)
        return self.forward_final(x)


if __name__ == "__main__":
    model = VMUNet(in_channels=1, out_channels=2, hidden_channels=32, depths=(1, 1, 1, 1), depths_decoder=(1, 1, 1, 1))
    x = torch.randn(1, 1, 256, 256)
    y = model(x)
    print(f"input:  {tuple(x.shape)}")
    print(f"output: {tuple(y.shape)}")
