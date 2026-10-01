import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.common.VGG import VGG16


class Backbone(VGG16):
    def __init__(self, num_classes: int = 21) -> None:
        super().__init__(num_classes)
        # 分割：不再下采样，保持 layer4 输出的 28×28（224 输入）
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.maxpool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.layer5 = nn.Sequential(
            # 1×1 不用 dilation，避免无意义的空洞参数
            nn.Conv2d(512, 256, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, dilation=2, padding=2),  # [B, 256, 28, 28]
            nn.ReLU(inplace=True),
        )
        # layer5 输出 256 通道；原 fn6 写 512 会与主分支 forward 不一致
        self.fn6 = nn.Conv2d(256, 4096, kernel_size=3, stride=1, padding=4, dilation=4)
        self.fn7 = nn.Conv2d(4096, 4096, kernel_size=1, stride=1)
        # 覆盖父类 Linear fc8：主分支输出与 skip 同形状的 score map
        self.fc8 = nn.Conv2d(4096, num_classes, kernel_size=1, stride=1)


class DeeplabV1(nn.Module):
    def __init__(self, num_classes: int = 21) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.skip_conv_stride = [8, 4, 2, 1, 1]
        self.skip_channels = [3, 64, 128, 256, 512]
        self.backbone = Backbone(num_classes)
        self.skip_conv = nn.ModuleList()
        for i in range(len(self.skip_conv_stride)):
            self.skip_conv.append(
                nn.Sequential(
                    # 各 stage 通道不同，先统一压到 128 再出类别
                    nn.Conv2d(
                        self.skip_channels[i],
                        128,
                        kernel_size=3,
                        stride=self.skip_conv_stride[i],
                        padding=1,
                    ),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=0.5),
                    nn.Conv2d(128, 128, kernel_size=1, stride=1),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=0.5),
                    nn.Conv2d(128, num_classes, kernel_size=1, stride=1),
                )
            )

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
        x = self.backbone.fn6(x)
        x = F.relu(x)
        x = self.backbone.fn7(x)
        x = F.relu(x)
        main_score = self.backbone.fc8(x)

        features = [s0, s1, s2, s3, s4]
        skip_out: torch.Tensor | None = None
        for i, f in enumerate(features):
            score = self.skip_conv[i](f)
            # stride 卷积偶发 off-by-one 时，与主分支 score 对齐后再相加
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
    model = DeeplabV1(num_classes=21)
    y = model(torch.randn(2, 3, 224, 224))
    print(y.shape)