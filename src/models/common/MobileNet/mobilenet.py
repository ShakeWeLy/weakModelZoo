import torch
import torch.nn as nn

from src.module.conv import BaseConv, DepthwiseConv, PointwiseConv


def _make_divisible(value: float, divisor: int = 8, min_value: int | None = None) -> int:
    """将通道数调整为 divisor 的倍数，避免过小通道。"""
    if min_value is None:
        min_value = divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    if new_value < 0.9 * value:
        new_value += divisor
    return new_value


class MobileNetV1Block(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()
        relu = nn.ReLU(inplace=True)
        self.conv1 = DepthwiseConv(
            in_channels, kernel_size=3, stride=stride, padding=1, act=relu
        )
        self.conv2 = PointwiseConv(in_channels, out_channels, act=nn.ReLU(inplace=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.conv1(x))


class MobileNetV1(nn.Module):
    # (out_channels, stride)
    _cfg = (
        (64, 1),
        (128, 2),
        (128, 1),
        (256, 2),
        (256, 1),
        (512, 2),
        *[(512, 1)] * 5,
        (1024, 2),
        (1024, 1),
    )

    def __init__(self, num_classes: int = 1000, width_mult: float = 1.0) -> None:
        super().__init__()
        in_channels = _make_divisible(32 * width_mult)
        self.conv = BaseConv(
            3, in_channels, 3, 2, act=nn.ReLU(inplace=True)
        )  # [B, 3, 224, 224] -> [B, C, 112, 112]

        layers: list[nn.Module] = []
        for out_channels, stride in self._cfg:
            out_channels = _make_divisible(out_channels * width_mult)
            layers.append(MobileNetV1Block(in_channels, out_channels, stride))
            in_channels = out_channels
        self.features = nn.Sequential(*layers)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


class MobileNetV2Bottleneck(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        expand_ratio: int,
    ) -> None:
        super().__init__()
        if stride not in (1, 2):
            raise ValueError(f"Invalid stride: {stride}")

        hidden_dim = int(round(in_channels * expand_ratio))
        self.use_residual = stride == 1 and in_channels == out_channels

        layers: list[nn.Module] = []
        # expand（t=1 时跳过）
        if expand_ratio != 1:
            layers.append(
                PointwiseConv(in_channels, hidden_dim, act=nn.ReLU6(inplace=True))
            )
        layers.extend(
            [
                DepthwiseConv(
                    hidden_dim,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    act=nn.ReLU6(inplace=True),
                ),
                # linear bottleneck：project 后不加激活
                PointwiseConv(hidden_dim, out_channels, act=False),
            ]
        )
        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        if self.use_residual:
            return x + out
        return out


class MobileNetV2(nn.Module):
    # (expand_ratio, out_channels, num_blocks, stride)
    _cfg = (
        (1, 16, 1, 1),
        (6, 24, 2, 2),
        (6, 32, 3, 2),
        (6, 64, 4, 2),
        (6, 96, 3, 1),
        (6, 160, 3, 2),
        (6, 320, 1, 1),
    )

    def __init__(self, num_classes: int = 1000, width_mult: float = 1.0) -> None:
        super().__init__()
        in_channels = _make_divisible(32 * width_mult)
        last_channels = _make_divisible(1280 * width_mult) if width_mult > 1.0 else 1280

        self.conv_1 = BaseConv(
            3, in_channels, 3, 2, act=nn.ReLU6(inplace=True)
        )  # [B, 3, 224, 224] -> [B, C, 112, 112]

        layers: list[nn.Module] = []
        for expand_ratio, out_channels, num_blocks, stride in self._cfg:
            out_channels = _make_divisible(out_channels * width_mult)
            for i in range(num_blocks):
                layers.append(
                    MobileNetV2Bottleneck(
                        in_channels,
                        out_channels,
                        stride if i == 0 else 1,
                        expand_ratio,
                    )
                )
                in_channels = out_channels
        self.features = nn.Sequential(*layers)

        self.conv_last = PointwiseConv(
            in_channels, last_channels, act=nn.ReLU6(inplace=True)
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(last_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_1(x)
        x = self.features(x)
        x = self.conv_last(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


