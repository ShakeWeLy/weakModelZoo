import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Down3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.max_pool = nn.MaxPool3d(2)
        self.conv = DoubleConv3D(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.max_pool(self.conv(x))


class Up3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose3d(
            in_channels,
            in_channels // 2,
            kernel_size=2,
            stride=2,
        )
        self.conv = DoubleConv3D(in_channels, out_channels)

    def forward(self, x_input: torch.Tensor, x_skip: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x_input)
        diff_d = x_skip.size(2) - x1.size(2)
        diff_h = x_skip.size(3) - x1.size(3)
        diff_w = x_skip.size(4) - x1.size(4)

        x1 = F.pad(
            x1,
            [
                diff_w // 2,
                diff_w - diff_w // 2,
                diff_h // 2,
                diff_h - diff_h // 2,
                diff_d // 2,
                diff_d - diff_d // 2,
            ],
        )
        return self.conv(torch.cat([x_skip, x1], dim=1))


class OutConv3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet3D(nn.Module):
    """3D U-Net for volumetric inputs shaped as (B, C, D, H, W)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
    ) -> None:
        super().__init__()
        self.inc = DoubleConv3D(in_channels, hidden_channels)
        self.down1 = Down3D(hidden_channels, hidden_channels * 2)
        self.down2 = Down3D(hidden_channels * 2, hidden_channels * 4)
        self.down3 = Down3D(hidden_channels * 4, hidden_channels * 8)
        self.down4 = Down3D(hidden_channels * 8, hidden_channels * 16)
        self.up1 = Up3D(hidden_channels * 16, hidden_channels * 8)
        self.up2 = Up3D(hidden_channels * 8, hidden_channels * 4)
        self.up3 = Up3D(hidden_channels * 4, hidden_channels * 2)
        self.up4 = Up3D(hidden_channels * 2, hidden_channels)
        self.outc = OutConv3D(hidden_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)
