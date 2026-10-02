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
        # self.layer1 = BaseConv(
        #     in_channels=3,
        #     out_channels=64,
        #     kernel_size=7,
        #     stride=2,
        #     padding=3,
        #     bias=False,
        #     bn=True,
        #     act=nn.ReLU(inplace=True),
        # )
        # self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = nn.Sequential(
            BaseConv(
                in_channels=3,
                out_channels=64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
                bn=True,
                act=nn.ReLU(inplace=True),
            ),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
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
        # x = self.maxpool1(self.layer1(x))  # 224 -> 56
        x = self.layer1(x)  # 224 -> 56
        x = self.layer2(x)  # 56
        x = self.layer3(x)  # 28
        x = self.layer4(x)  # 14
        x = self.layer5(x)  # 7
        x = self.avgpool(x)
        return self.fc(x.flatten(1))


class ResNet34(nn.Module):
    def __init__(self, num_classes: int = 1000) -> None:
        super().__init__()
        self.in_channels = [64, 64, 128, 256]  # !input channels for each layer in first block
        self.out_channels = [64, 128, 256, 512]
        self.repeat_layers = [3, 4, 6, 3]
        self.layer1 = nn.Sequential(
            BaseConv(3, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        for i in range(len(self.repeat_layers)):
            setattr(self, f"layer{i+2}", nn.Sequential(
                BasicBlock(self.in_channels[i], self.out_channels[i], stride=1 if i == 0 else 2),
                *[BasicBlock(self.out_channels[i], self.out_channels[i], stride=1) for _ in range(self.repeat_layers[i]-1)]
            ))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.avgpool(x)
        return self.fc(x.flatten(1))


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1) -> None:
        super().__init__()
        self.conv1 = BaseConv(in_channels, out_channels, kernel_size=1, stride=1, padding=0, act=nn.ReLU(inplace=True))
        self.conv2 = BaseConv(out_channels, out_channels, kernel_size=3, stride=stride, padding=padding, act=nn.ReLU(inplace=True))
        self.conv3 = BaseConv(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, padding=0, act=False)

        self.downsample: nn.Module | None = None
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.downsample = BaseConv(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, padding=0, act=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.downsample(x) if self.downsample is not None else x
        out = self.conv3(self.conv2(self.conv1(x)))
        out = self.relu(out + identity)
        return out


class ResNet50(nn.Module):
    def __init__(self, num_classes: int = 1000) -> None:
        super().__init__()
        self.in_channels = [64, 256, 512, 1024]
        self.out_channels = [64, 128, 256, 512]
        self.repeat_layers = [3, 4, 6, 3]
        self.layer1 = nn.Sequential(
            BaseConv(3, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        for i in range(len(self.repeat_layers)):
            setattr(self, f"layer{i+2}", nn.Sequential(
                Bottleneck(self.in_channels[i], self.out_channels[i], stride=1 if i == 0 else 2),
                *[Bottleneck(self.out_channels[i]*4, self.out_channels[i], stride=1) for _ in range(self.repeat_layers[i]-1)]  # !inchanels should be out_channels * 4, because of the expansion factor by the last block
            ))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.avgpool(x)
        return self.fc(x.flatten(1))


class ResNet101(nn.Module):
    def __init__(self, num_classes: int = 1000) -> None:
        super().__init__()
        self.in_channels = [64, 256, 512, 1024]
        self.out_channels = [64, 128, 256, 512]
        self.repeat_layers = [3, 4, 23, 3]
        self.layer1 = nn.Sequential(
            BaseConv(3, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        for i in range(len(self.repeat_layers)):
            setattr(self, f"layer{i+2}", nn.Sequential(
                Bottleneck(self.in_channels[i], self.out_channels[i], stride=1 if i == 0 else 2),
                *[Bottleneck(self.out_channels[i]*4, self.out_channels[i], stride=1) for _ in range(self.repeat_layers[i]-1)]
            ))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.avgpool(x)
        return self.fc(x.flatten(1))


class ResNet152(nn.Module):
    def __init__(self, num_classes: int = 1000) -> None:
        super().__init__()
        self.in_channels = [64, 256, 512, 1024]
        self.out_channels = [64, 128, 256, 512]
        self.repeat_layers = [3, 4, 30, 3]
        self.layer1 = nn.Sequential(
            BaseConv(3, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        for i in range(len(self.repeat_layers)):
            setattr(self, f"layer{i+2}", nn.Sequential(
                Bottleneck(self.in_channels[i], self.out_channels[i], stride=1 if i == 0 else 2),
                *[Bottleneck(self.out_channels[i]*4, self.out_channels[i], stride=1) for _ in range(self.repeat_layers[i]-1)]
            ))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.avgpool(x)
        return self.fc(x.flatten(1))


if __name__ == "__main__":
    # python -m src.models.common.ResNet.resnet18
    # model = ResNet18(num_classes=1000)
    # out = model(torch.randn(1, 3, 224, 224))
    # print(out.shape)

    # model = ResNet34(num_classes=1000)
    # print(model)
    # out = model(torch.randn(1, 3, 224, 224))
    # print(out.shape)

    # model = ResNet50(num_classes=1000)
    # print(model)
    # out = model(torch.randn(1, 3, 224, 224))
    # print(out.shape)

    # model = ResNet101(num_classes=1000)
    # print(model)
    # out = model(torch.randn(1, 3, 224, 224))
    # print(out.shape)

    model = ResNet152(num_classes=1000)
    print(model)
    print("total parameters: ", sum(p.numel() for p in model.parameters()))
    print("trainable parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
    print("total flops(M): ", sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6)
    out = model(torch.randn(1, 3, 224, 224))
    print(out.shape)