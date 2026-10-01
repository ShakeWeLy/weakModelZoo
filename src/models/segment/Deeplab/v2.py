import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.segment.Deeplab.deeplab import DeeplabV1

# 每路 ASPP 分支输出通道；四路 concat 后为 ASPP_BRANCH_CHANNELS * 4
ASPP_BRANCH_CHANNELS = 1024


class ASPP(nn.Module):
    """四路并行空洞卷积，输出 concat 后的特征图（不含类别维）。"""

    def __init__(self, in_channels: int, branch_channels: int) -> None:
        super().__init__()
        self.aspp_rates = [6, 12, 18, 24]
        self.aspp_conv = nn.ModuleList()
        for rate in self.aspp_rates:
            self.aspp_conv.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        branch_channels,
                        kernel_size=3,
                        padding=rate,
                        dilation=rate,
                    ),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(branch_channels, branch_channels, kernel_size=1),
                    nn.ReLU(inplace=True),
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branches = [block(x) for block in self.aspp_conv]
        return torch.cat(branches, dim=1)


class DeeplabV2(DeeplabV1):
    def __init__(self, num_classes: int = 21) -> None:
        super().__init__(num_classes)
        self.aspp = ASPP(in_channels=256, branch_channels=ASPP_BRANCH_CHANNELS)
        # 覆盖 V1 的 fc8：ASPP concat 为 4*branch_channels，再压到 num_classes
        aspp_out_channels = ASPP_BRANCH_CHANNELS * len(self.aspp.aspp_rates)
        self.backbone.fc8 = nn.Conv2d(aspp_out_channels, num_classes, kernel_size=1)
        # V2 不用 fn6/fn7，保留模块避免误走 V1 路径时仍占位（可选，仅占参数名）
        self.backbone.fn6 = nn.Identity()
        self.backbone.fn7 = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s0 = x
        x = self.backbone.layer1(x)
        x = self.backbone.maxpool1(x)
        s1 = x
        x = self.backbone.layer2(x)
        x = self.backbone.maxpool2(x)
        s2 = x
        x = self.backbone.layer3(x)
        x = self.backbone.maxpool3(x)
        s3 = x
        x = self.backbone.layer4(x)
        x = self.backbone.maxpool4(x)
        s4 = x
        x = self.backbone.layer5(x)
        x = self.backbone.maxpool5(x)

        main_score = self.backbone.fc8(self.aspp(x))

        features = [s0, s1, s2, s3, s4]
        skip_out: torch.Tensor | None = None
        for i, f in enumerate(features):
            score = self.skip_conv[i](f)
            if score.shape[-2:] != main_score.shape[-2:]:
                score = F.interpolate(
                    score,
                    size=main_score.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            skip_out = score if skip_out is None else skip_out + score

        return main_score + skip_out


if __name__ == "__main__":
    model = DeeplabV2(num_classes=21)
    y = model(torch.randn(2, 3, 224, 224))
    print(y.shape)
