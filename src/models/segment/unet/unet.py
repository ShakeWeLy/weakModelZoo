import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.max_pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.max_pool(self.conv(x))

class Up(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels,
            in_channels // 2,
            kernel_size=2,
            stride=2,
        )
        self.conv = DoubleConv(in_channels, out_channels)
    def forward(self, x_input: torch.Tensor, x_skip: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x_input)
        # !对齐奇数维度
        diff_y = x_skip.size(2) - x1.size(2)
        diff_x = x_skip.size(3) - x1.size(3)
        x1 = F.pad(
            x1,
            [
                diff_x // 2,
                diff_x - diff_x // 2,
                diff_y // 2,
                diff_y - diff_y // 2,
            ],
        )
        x = torch.cat([x_skip, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            # DoubleConv(in_channels, out_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=(1,1),)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, hidden_channels: int):
        super().__init__()
        self.inc = DoubleConv(in_channels, hidden_channels)
        self.down1 = Down(hidden_channels, hidden_channels * 2)
        self.down2 = Down(hidden_channels * 2, hidden_channels * 4)
        self.down3 = Down(hidden_channels * 4, hidden_channels * 8)
        self.down4 = Down(hidden_channels * 8, hidden_channels * 16)
        self.up1 = Up(hidden_channels * 16, hidden_channels * 8)
        self.up2 = Up(hidden_channels * 8, hidden_channels * 4)
        self.up3 = Up(hidden_channels * 4, hidden_channels * 2)
        self.up4 = Up(hidden_channels * 2, hidden_channels)

        self.outc = OutConv(hidden_channels, out_channels)
    
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


if __name__ == "__main__":
    model = UNet(in_channels=1, out_channels=1, hidden_channels=64)
    print(model)
    def print_shape_hook(name):
        def hook(module, inputs, output):
            if isinstance(output, torch.Tensor):
                print(f"{name:<40} -> {tuple(output.shape)}")
        return hook


    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # 只打印叶子层
            module.register_forward_hook(print_shape_hook(name))

    x = torch.randn(1, 1,572, 572)
    print(x.shape)
    y = model(x)

    print("最终输出:", tuple(y.shape))