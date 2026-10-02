import torch
import torch.nn as nn

from src.module.conv import BaseConv


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ) -> None:
        super().__init__()
        self.conv1 = BaseConv(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
            bn=True,
            act=nn.ReLU(inplace=True),
        )
        self.conv2 = BaseConv(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            bias=False,
            bn=True,
            act=False,
        )
        # 仅在下采样或通道变化时使用 1×1；与 torchvision ResNet 一致，shortcut 上无 ReLU
        self.downsample: nn.Module | None = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = BaseConv(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                padding=0,
                bias=False,
                bn=True,
                act=False,
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.downsample(x) if self.downsample is not None else x
        out = self.conv2(self.conv1(x))
        out = self.relu(out + identity)
        return out


class ResNet18(nn.Module):
    def __init__(self, num_classes: int = 1000) -> None:
        super().__init__()
        self.layer1 = BaseConv(
            in_channels=3,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
            bn=True,
            act=nn.ReLU(inplace=True),
        )
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer2 = nn.Sequential(
            BasicBlock(64, 64, stride=1),
            BasicBlock(64, 64, stride=1),
        )
        self.layer3 = nn.Sequential(
            BasicBlock(64, 128, stride=2),
            BasicBlock(128, 128, stride=1),
        )
        self.layer4 = nn.Sequential(
            BasicBlock(128, 256, stride=2),
            BasicBlock(256, 256, stride=1),
        )
        self.layer5 = nn.Sequential(
            BasicBlock(256, 512, stride=2),
            BasicBlock(512, 512, stride=1),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.maxpool1(self.layer1(x))  # 224 -> 56
        x = self.layer2(x)  # 56
        x = self.layer3(x)  # 28
        x = self.layer4(x)  # 14
        x = self.layer5(x)  # 7
        x = self.avgpool(x)
        return self.fc(x.flatten(1))


if __name__ == "__main__":
    model = ResNet18(num_classes=1000)
    out = model(torch.randn(1, 3, 224, 224))
    print(out.shape)
