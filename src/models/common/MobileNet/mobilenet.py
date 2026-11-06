import torch
import torch.nn as nn
import torch.nn.functional as F

from src.module.conv import BaseConv, DepthwiseConv, PointwiseConv


class MobileNetV1Block(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()
        self.conv1 = DepthwiseConv(in_channels, kernel_size=3, stride=stride, padding=1)
        self.conv2 = PointwiseConv(in_channels, out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.relu(x)
        return x



class MobileNetV1(nn.Module):
    def __init__(self, num_classes: int = 1000, width_mult: float = 1.0) -> None:
        super().__init__()
        self.width_mult = width_mult

        self.conv = BaseConv(3, 32, 3, 2)  # [batch_size, 3, 224, 224] -> [batch_size, 32, 112, 112]
        self.features_1 = MobileNetV1Block(32, 64, 1)  # [batch_size, 32, 112, 112] -> [batch_size, 64, 112, 112]
        self.features_2 = MobileNetV1Block(64, 128, 2)  # [batch_size, 64, 112, 112] -> [batch_size, 128, 56, 56]
        self.features_3 = nn.Sequential(
            MobileNetV1Block(128, 128, 1),  # [batch_size, 128, 56, 56] -> [batch_size, 128, 56, 56]
            MobileNetV1Block(128, 256, 2),  # [batch_size, 128, 56, 56] -> [batch_size, 256, 28, 28]
        )
        self.features_4 = nn.Sequential(
            MobileNetV1Block(256, 256, 1),  # [batch_size, 256, 28, 28] -> [batch_size, 256, 28, 28]
            MobileNetV1Block(256, 512, 2),  # [batch_size, 256, 28, 28] -> [batch_size, 512, 14, 14]
        )
        self.features_5 = nn.Sequential(
            *[MobileNetV1Block(512, 512, 1) for _ in range(5)],
        )  # [batch_size, 512, 14, 14] -> [batch_size, 512, 14, 14]
        self.features_6 = nn.Sequential(
            MobileNetV1Block(512, 1024, 2),
        )  # [batch_size, 512, 14, 14] -> [batch_size, 1024, 7, 7]
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=self.width_mult, mode='bilinear', align_corners=False)
        x = self.conv(x)
        x = self.features_1(x)
        x = self.features_2(x)
        x = self.features_3(x)
        x = self.features_4(x)
        x = self.features_5(x)
        x = self.features_6(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x



class MobileNetV2Bottleneck(nn.Module):
    def __init__(self, in_channels: int, channels_scale: int,out_channels: int, stride: int) -> None:
        super().__init__()
        dim_chanels = in_channels*channels_scale
        if stride not in (1,2):
            raise ValueError(f"Invalid stride: {stride}")
        self.stride =stride
        self.bottleneck = nn.Sequential(
            PointwiseConv(in_channels, dim_chanels),
            nn.ReLU(inplace=True),
            DepthwiseConv(dim_chanels, kernel_size=3, stride=stride, padding=1),
            nn.Linear(out_channels, out_channels),
            PointwiseConv(dim_chanels, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.stride == 2:
            return self.bottleneck(x)
        return x + self.bottleneck(x)

class MobileNetV2(nn.Module):
    def __init__(self, num_classes: int = 1000, width_mult: float = 1.0) -> None:
        super().__init__()
        self.width_mult = width_mult

        self.conv_1 = BaseConv(3, 32, 3, 2)  # [batch_size, 3, 224, 224] -> [batch_size, 32, 112, 112]
        self.features_2 = MobileNetV2Bottleneck(32, 1, 16, 1)  # [batch_size, 32, 112, 112] -> [batch_size, 16, 112, 112]
        self.features_3 = nn.Sequential(
            MobileNetV2Bottleneck(16, 6, 16, 2), 
            MobileNetV2Bottleneck(16, 6, 24, 1),  # [batch_size, 16, 112, 112] -> [batch_size, 24, 56, 56]
        )
        self.features_4 = nn.Sequential(
            MobileNetV2Bottleneck(24, 6, 24, 2),
            MobileNetV2Bottleneck(24, 6, 24, 1),
            MobileNetV2Bottleneck(24, 6, 32, 1), # [batch_size, 24, 56, 56] -> [batch_size, 32, 28, 28]
        )
        self.features_5 = nn.Sequential(
            MobileNetV2Bottleneck(32, 6, 32, 2),
            MobileNetV2Bottleneck(32, 6, 32, 1),
            MobileNetV2Bottleneck(32, 6, 32, 1),
            MobileNetV2Bottleneck(32, 6, 64, 1),
        )  # [batch_size, 32, 28, 28] -> [batch_size, 64, 14, 14]
        self.features_6 = nn.Sequential(
            MobileNetV2Bottleneck(64, 6, 64, 2),
            MobileNetV2Bottleneck(64, 6, 64, 1),
            MobileNetV2Bottleneck(64, 6, 64, 1),
            MobileNetV2Bottleneck(64, 6, 96, 1),
        )  # [batch_size, 64, 14, 14] -> [batch_size, 96, 14, 14]
        self.features_7 = nn.Sequential(
            MobileNetV2Bottleneck(96, 6, 96, 2),
            MobileNetV2Bottleneck(96, 6, 96, 1),
            MobileNetV2Bottleneck(96, 6, 96, 1),
            MobileNetV2Bottleneck(96, 6, 160, 1),
        )  # [batch_size, 96, 14, 14] -> [batch_size, 160, 7, 7]
        self.features_8 = nn.Sequential(
            MobileNetV2Bottleneck(160, 6, 320, 1),
        )  # [batch_size, 160, 7, 7] -> [batch_size, 320, 7, 7]
        self.pwConv_9 = PointwiseConv(320, 1280)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1280, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=self.width_mult, mode='bilinear', align_corners=False)
        x = self.conv_1(x)
        x = self.features_2(x)
        x = self.features_3(x)
        x = self.features_4(x)
        x = self.features_5(x)
        x = self.features_6(x)
        x = self.features_7(x)
        x = self.features_8(x)
        x = self.pwConv_9(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x