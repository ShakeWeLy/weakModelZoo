"""torchvision 官方 ResNet 编码器：ImageNet 预训练 + 多尺度特征。"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
from torchvision.models import (
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
    ResNet101_Weights,
    ResNet152_Weights,
    resnet18,
    resnet34,
    resnet50,
    resnet101,
    resnet152,
)

ResNetName = Literal["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]

_BUILDERS = {
    "resnet18": (resnet18, ResNet18_Weights),
    "resnet34": (resnet34, ResNet34_Weights),
    "resnet50": (resnet50, ResNet50_Weights),
    "resnet101": (resnet101, ResNet101_Weights),
    "resnet152": (resnet152, ResNet152_Weights),
}

# layer1..4 输出通道（Bottleneck 系列为 256/512/1024/2048）
_OUT_CHANNELS: dict[str, tuple[int, int, int, int]] = {
    "resnet18": (64, 128, 256, 512),
    "resnet34": (64, 128, 256, 512),
    "resnet50": (256, 512, 1024, 2048),
    "resnet101": (256, 512, 1024, 2048),
    "resnet152": (256, 512, 1024, 2048),
}

# 项目内默认权重文件名（相对 pretrained_weights/resnet/）
_LOCAL_WEIGHT_FILES: dict[str, str] = {
    "resnet18": "resnet18-f37072fd.pth",
    "resnet34": "resnet34-b627a593.pth",
}


class TorchvisionResNetEncoder(nn.Module):
    """官方 ResNet stem + layer1..4，``forward_features`` 返回四级特征图。

    空间尺寸（输入 H×W）：H/4, H/8, H/16, H/32。
    默认加载 ImageNet 权重且 **不冻结**（可继续训练）。

    ``weights_path`` 优先；否则若 ``pretrained=True`` 则用 torchvision 在线权重。
    """

    def __init__(
        self,
        name: ResNetName = "resnet34",
        *,
        pretrained: bool = True,
        in_channels: int = 3,
        weights_path: str | Path | None = None,
    ) -> None:
        super().__init__()
        if name not in _BUILDERS:
            raise ValueError(f"不支持的 ResNet: {name!r}，可选 {tuple(_BUILDERS)}")
        builder, weights_enum = _BUILDERS[name]

        path = Path(weights_path) if weights_path else None
        if path is not None and str(path).strip() and path.is_file():
            net = builder(weights=None)
            state = torch.load(path, map_location="cpu", weights_only=True)
            _, unexpected = net.load_state_dict(state, strict=False)
            if unexpected:
                raise RuntimeError(f"加载 {path} 出现 unexpected keys: {unexpected[:8]}")
            # fc 头不参与编码，缺失可忽略
            loaded = True
        elif pretrained:
            net = builder(weights=weights_enum.DEFAULT)
            loaded = True
        else:
            net = builder(weights=None)
            loaded = False

        self.name = name
        self.out_channels = _OUT_CHANNELS[name]
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4

        if in_channels != 3:
            self._adapt_stem_in_channels(in_channels, pretrained=loaded)

    def _adapt_stem_in_channels(self, in_channels: int, *, pretrained: bool) -> None:
        old: nn.Conv2d = self.stem[0]  # type: ignore[assignment]
        new = nn.Conv2d(
            in_channels,
            old.out_channels,
            kernel_size=old.kernel_size,
            stride=old.stride,
            padding=old.padding,
            bias=False,
        )
        with torch.no_grad():
            if pretrained and in_channels == 1:
                new.weight.copy_(old.weight.mean(dim=1, keepdim=True))
            elif pretrained and in_channels > 3:
                new.weight[:, :3].copy_(old.weight)
                nn.init.kaiming_normal_(new.weight[:, 3:], mode="fan_out", nonlinearity="relu")
            else:
                nn.init.kaiming_normal_(new.weight, mode="fan_out", nonlinearity="relu")
        self.stem[0] = new

    def forward_features(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.stem(x)
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        return [c1, c2, c3, c4]

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        return self.forward_features(x)
