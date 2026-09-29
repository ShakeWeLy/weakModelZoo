import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.segment.unet.unet import DoubleConv, Down, OutConv, Up
from src.module.transformer.positionEmbedding.positionEmbedding import LearnablePositionalEmbedding
from src.module.transformer.transformerBlock import TransformerDecoderBlock


class TransUNet(nn.Module):
    """UNet 编码器 + Transformer 瓶颈增强 + UNet 解码器。

    CNN 编码器提取多尺度特征后，将最深层特征展平为序列，
    经 12 层 TransformerDecoderBlock 做全局特征增强，再还原为空间特征送入解码器。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        image_size: int = 224,
        transformer_depth: int = 12,
        num_heads: int = 8,
        ffn_hidden_channels: int | None = None,
    ) -> None:
        super().__init__()
        self.bottleneck_channels = hidden_channels * 16
        bottleneck_spatial = image_size // 16
        num_tokens = bottleneck_spatial ** 2

        self.inc = DoubleConv(in_channels, hidden_channels)
        self.down1 = Down(hidden_channels, hidden_channels * 2)
        self.down2 = Down(hidden_channels * 2, hidden_channels * 4)
        self.down3 = Down(hidden_channels * 4, hidden_channels * 8)
        self.down4 = Down(hidden_channels * 8, self.bottleneck_channels)

        self.pos_embed = LearnablePositionalEmbedding(num_tokens, self.bottleneck_channels)
        ffn_hidden = ffn_hidden_channels or self.bottleneck_channels * 4
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerDecoderBlock(self.bottleneck_channels, num_heads, ffn_hidden)
                for _ in range(transformer_depth)
            ]
        )
        self.transformer_norm = nn.LayerNorm(self.bottleneck_channels)

        self.up1 = Up(self.bottleneck_channels, hidden_channels * 8)
        self.up2 = Up(hidden_channels * 8, hidden_channels * 4)
        self.up3 = Up(hidden_channels * 4, hidden_channels * 2)
        self.up4 = Up(hidden_channels * 2, hidden_channels)
        self.outc = OutConv(hidden_channels, out_channels)

    def _add_pos_embed(self, x: torch.Tensor) -> torch.Tensor:
        """为序列特征添加位置编码，空间尺寸与训练时不一致时插值位置编码。"""
        pos_embed = self.pos_embed.pos_embed
        if pos_embed.shape[1] == x.shape[1]:
            return self.pos_embed(x)

        gs_old = int(pos_embed.shape[1] ** 0.5)
        gs_new = int(x.shape[1] ** 0.5)
        pos = pos_embed.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
        pos = F.interpolate(pos, size=(gs_new, gs_new), mode="bilinear", align_corners=False)
        pos = pos.flatten(2).transpose(1, 2)
        return x + pos

    def _enhance_with_transformer(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = x.shape
        x_seq = x.flatten(2).transpose(1, 2)
        x_seq = self._add_pos_embed(x_seq)
        for block in self.transformer_blocks:
            x_seq = block(x_seq)
        x_seq = self.transformer_norm(x_seq)
        return x_seq.transpose(1, 2).reshape(batch_size, channels, height, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x5 = self._enhance_with_transformer(x5)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)


if __name__ == "__main__":
    model = TransUNet(in_channels=1, out_channels=1, hidden_channels=64, image_size=224)
    print(model)

    def print_shape_hook(name: str):
        def hook(_module: nn.Module, _inputs: tuple, output: torch.Tensor) -> None:
            if isinstance(output, torch.Tensor):
                print(f"{name:<40} -> {tuple(output.shape)}")

        return hook

    for name, module in model.named_modules():
        if len(list(module.children())) == 0:
            module.register_forward_hook(print_shape_hook(name))

    x = torch.randn(1, 1, 224, 224)
    print(x.shape)
    y = model(x)
    print("最终输出:", tuple(y.shape))
