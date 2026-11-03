"""跟踪专用 ResNet（atrous）：layer3/4 用 dilation 保持较大特征图。

对齐 pysot ``models/backbone/resnet_atrous.py``，默认输出 layer2/3/4 供多层 RPN 使用。
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, dilation: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        bias=False,
        dilation=dilation,
    )


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        padding = 2 - stride
        if downsample is not None and dilation > 1:
            dilation = dilation // 2
            padding = dilation
        if dilation > 1:
            padding = dilation

        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=padding, bias=False, dilation=dilation
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class ResNetAtrous(nn.Module):
    """ResNet-50 atrous 骨干。

    ``used_layers`` 索引约定（与 pysot 一致）::
        0: stem(relu 后), 1: layer1, 2: layer2, 3: layer3, 4: layer4
    SiamSPN++ 默认 ``(2, 3, 4)``，通道分别为 512 / 1024 / 2048。
    """

    def __init__(
        self,
        layers: Sequence[int] = (3, 4, 6, 3),
        used_layers: Sequence[int] = (2, 3, 4),
    ) -> None:
        super().__init__()
        self.inplanes = 64
        self.used_layers = list(used_layers)

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, layers[0])
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=1, dilation=2) if 3 in used_layers else None
        self.layer4 = self._make_layer(512, layers[3], stride=1, dilation=4) if 4 in used_layers else None

        self._init_weights()
        self.out_channels = self._infer_out_channels()

    def _make_layer(
        self,
        planes: int,
        blocks: int,
        stride: int = 1,
        dilation: int = 1,
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * Bottleneck.expansion:
            if stride == 1 and dilation == 1:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * Bottleneck.expansion, 1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes * Bottleneck.expansion),
                )
            else:
                dd = dilation // 2 if dilation > 1 else 1
                padding = dd if dilation > 1 else 0
                downsample = nn.Sequential(
                    nn.Conv2d(
                        self.inplanes,
                        planes * Bottleneck.expansion,
                        kernel_size=3,
                        stride=stride,
                        bias=False,
                        padding=padding,
                        dilation=dd,
                    ),
                    nn.BatchNorm2d(planes * Bottleneck.expansion),
                )

        layers = [Bottleneck(self.inplanes, planes, stride, downsample, dilation=dilation)]
        self.inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes, dilation=dilation))
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _infer_out_channels(self) -> tuple[int, ...]:
        mapping = {0: 64, 1: 256, 2: 512, 3: 1024, 4: 2048}
        return tuple(mapping[i] for i in self.used_layers)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.conv1(x)
        x = self.bn1(x)
        x0 = self.relu(x)
        x = self.maxpool(x0)

        p1 = self.layer1(x)
        p2 = self.layer2(p1)
        p3 = self.layer3(p2) if self.layer3 is not None else p2
        p4 = self.layer4(p3) if self.layer4 is not None else p3

        feats = [x0, p1, p2, p3, p4]
        return [feats[i] for i in self.used_layers]


def resnet50(used_layers: Sequence[int] = (2, 3, 4)) -> ResNetAtrous:
    return ResNetAtrous(layers=(3, 4, 6, 3), used_layers=used_layers)
