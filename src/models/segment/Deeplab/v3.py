import torch
import torch.nn as nn
import torch.nn.functional as F

from src.module.conv import BaseConv
from src.models.common.ResNet import ResNet18


class ASPP(nn.Module):
    def __init__(self, in_channels: int = 2048, out_channels: int = 256) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=6, dilation=6)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=12, dilation=12)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.conv4 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=18, dilation=18)
        self.bn4 = nn.BatchNorm2d(out_channels)

        # self.pool = nn.AdaptiveAvgPool2d(1)  # ??? WH will be 1
        self.pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        # self.bn5 = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x2 = self.conv2(x)
        x2 = self.bn2(x2)
        x3 = self.conv3(x)
        x3 = self.bn3(x3)
        x4 = self.conv4(x)
        x4 = self.bn4(x4)
        print("x4: ", x4.shape)
        x5 = self.pool(x)
        x5 = self.conv5(x5)
        print("x5: ", x5.shape)
        # x5 = self.bn5(x5)  # !if B = 1, but H=1 and W=1, so BN will be error
        x5 = F.relu(x5)
        x = torch.cat([x1, x2, x3, x4, x5], dim=1)
        x = F.relu(x)
        return x


class Decoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1x1 = BaseConv(
            64,
            48,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
            bn=True,
            act=False,  # !no activation, because it will be concatenated with low_level_features
        )
        self.conv3x3 = BaseConv(
            256+48,
            512,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            bn=True,
            act=nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, low_level_features: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=low_level_features.size()[2:], mode='bilinear', align_corners=True)
        low_level_features = self.conv1x1(low_level_features)
        x = torch.cat([x, low_level_features], dim=1)
        x = self.conv3x3(x)
        return x



class DeeplabV3(nn.Module):
    def __init__(self, num_classes: int = 1000) -> None:
        super().__init__()
        self.backbone = ResNet18()
        self.aspp = ASPP(in_channels=512, out_channels=256)
        self.conv1x1 = BaseConv(
            5*256,
            256,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
            bn=True,
            act=nn.ReLU(inplace=True),
        )
        self.decoder = Decoder()
        self.classifier = nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.size()[2:]
        x1 = self.backbone.layer1(x)  # [N, 64, H/4, W/4]
        x2 = self.backbone.layer2(x1)  # [N, 64, H/8, W/8]
        x3 = self.backbone.layer3(x2)  # [N, 128, H/16, W/16]
        x4 = self.backbone.layer4(x3)  # [N, 256, H/32, W/32]
        x5 = self.backbone.layer5(x4)  # [N, 512, H/64, W/64]
        x = self.aspp(x5)  # [N, 256*5, H/64, W/64]
        x = self.conv1x1(x)  # [N, 256, H/64, W/64]
        x = self.decoder(x, x2)  # [N, 256, H/8, W/8]
        print("x: ", x.shape)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)  # /8 upsample to WH
        print("x: ", x.shape)
        x = self.classifier(x)  # [N, num_classes, H, W]
        print("x: ", x.shape)
        return x


if __name__ == "__main__":
    # python -m src.models.segment.Deeplab.v3
    model = DeeplabV3()
    out = model(torch.randn(2, 3, 224, 224))
    print(out.shape)