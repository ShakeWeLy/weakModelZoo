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
        self.conv1 = DepthwiseConv(
            in_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            act=nn.ReLU(inplace=True),
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


class SqueezeExcite(nn.Module):
    """MobileNetV3 用的 SE：GAP → 1x1 降维 → ReLU → 1x1 升维 → h-sigmoid。"""

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        squeeze = _make_divisible(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, squeeze, 1)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(squeeze, channels, 1)
        self.gate = nn.Hardsigmoid(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.gate(self.fc2(self.act(self.fc1(self.pool(x)))))
        return x * scale


class MobileNetV3Block(nn.Module):
    """V3 倒残差：可选 SE、可选 h-swish，DW 核可为 3/5。"""

    def __init__(
        self,
        in_channels: int,
        expand_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        use_se: bool,
        use_hs: bool,
    ) -> None:
        super().__init__()
        if stride not in (1, 2):
            raise ValueError(f"Invalid stride: {stride}")
        if kernel_size not in (3, 5):
            raise ValueError(f"Invalid kernel_size: {kernel_size}")

        self.use_residual = stride == 1 and in_channels == out_channels

        def _act() -> nn.Module:
            return nn.Hardswish(inplace=True) if use_hs else nn.ReLU(inplace=True)

        layers: list[nn.Module] = []
        if expand_channels != in_channels:
            layers.append(PointwiseConv(in_channels, expand_channels, act=_act()))

        # DW 先不做激活，SE 之后再激活（与常见实现一致）
        layers.append(
            DepthwiseConv(
                expand_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                act=False,
            )
        )
        if use_se:
            layers.append(SqueezeExcite(expand_channels))
        layers.append(_act())
        layers.append(PointwiseConv(expand_channels, out_channels, act=False))
        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        if self.use_residual:
            return x + out
        return out


class MobileNetV3(nn.Module):
    """
    MobileNetV3-Large / Small。
    cfg 每项: (kernel, expand, out, use_se, use_hs, stride)
    """

    _cfgs = {
        "large": (
            (3, 16, 16, False, False, 1),
            (3, 64, 24, False, False, 2),
            (3, 72, 24, False, False, 1),
            (5, 72, 40, True, False, 2),
            (5, 120, 40, True, False, 1),
            (5, 120, 40, True, False, 1),
            (3, 240, 80, False, True, 2),
            (3, 200, 80, False, True, 1),
            (3, 184, 80, False, True, 1),
            (3, 184, 80, False, True, 1),
            (3, 480, 112, True, True, 1),
            (3, 672, 112, True, True, 1),
            (5, 672, 160, True, True, 2),
            (5, 960, 160, True, True, 1),
            (5, 960, 160, True, True, 1),
        ),
        "small": (
            (3, 16, 16, True, False, 2),
            (3, 72, 24, False, False, 2),
            (3, 88, 24, False, False, 1),
            (5, 96, 40, True, True, 2),
            (5, 240, 40, True, True, 1),
            (5, 240, 40, True, True, 1),
            (5, 120, 48, True, True, 1),
            (5, 144, 48, True, True, 1),
            (5, 288, 96, True, True, 2),
            (5, 576, 96, True, True, 1),
            (5, 576, 96, True, True, 1),
        ),
    }
    # (last_conv_channels, classifier_channels, last_conv_use_se)
    _head = {
        "large": (960, 1280, False),
        "small": (576, 1024, True),
    }

    def __init__(
        self,
        variant: str = "large",
        num_classes: int = 1000,
        width_mult: float = 1.0,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        variant = variant.lower()
        if variant not in self._cfgs:
            raise ValueError(f"variant must be 'large' or 'small', got {variant!r}")

        in_channels = _make_divisible(16 * width_mult)
        last_conv_c, classifier_c, last_se = self._head[variant]
        last_conv_c = (
            _make_divisible(last_conv_c * width_mult)
            if width_mult > 1.0
            else last_conv_c
        )
        classifier_c = (
            _make_divisible(classifier_c * width_mult)
            if width_mult > 1.0
            else classifier_c
        )

        layers: list[nn.Module] = [
            BaseConv(3, in_channels, 3, 2, act=nn.Hardswish(inplace=True))
        ]
        for kernel, expand, out_c, use_se, use_hs, stride in self._cfgs[variant]:
            expand_c = _make_divisible(expand * width_mult)
            out_c = _make_divisible(out_c * width_mult)
            layers.append(
                MobileNetV3Block(
                    in_channels, expand_c, out_c, kernel, stride, use_se, use_hs
                )
            )
            in_channels = out_c

        # 最后一层 1x1；Small 按论文在此处加 SE
        head: list[nn.Module] = [PointwiseConv(in_channels, last_conv_c, act=False)]
        if last_se:
            head.append(SqueezeExcite(last_conv_c))
        head.append(nn.Hardswish(inplace=True))
        layers.extend(head)
        self.features = nn.Sequential(*layers)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(last_conv_c, classifier_c),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(classifier_c, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def mobilenet_v3_large(
    num_classes: int = 1000, width_mult: float = 1.0, dropout: float = 0.2
) -> MobileNetV3:
    return MobileNetV3("large", num_classes, width_mult, dropout)


def mobilenet_v3_small(
    num_classes: int = 1000, width_mult: float = 1.0, dropout: float = 0.2
) -> MobileNetV3:
    return MobileNetV3("small", num_classes, width_mult, dropout)
