import torch
import torch.nn as nn
import torch.nn.functional as F


class ASPP(nn.Module):
    """
    ASPP 空洞空间金字塔池化 (DeepLab, TPAMI 2017 / 2018)。

    一共 5 个分支:
        branch1: 1x1 卷积
        branch2: 3x3 空洞卷积, 膨胀率 rate1
        branch3: 3x3 空洞卷积, 膨胀率 rate2
        branch4: 3x3 空洞卷积, 膨胀率 rate3
        branch5: 图像级池化 (全局平均池化 -> 1x1 卷积 -> 上采样回原尺寸)

    5 个分支拼接后, 用 1x1 卷积投影回 out_channels。

    注意: branch5 含有 BatchNorm, 训练时 batch size 为 1 会报错, 请使用 batch size > 1。

    Args:
        in_channels: 输入通道数
        out_channels: 输出通道数
        atrous_rates: 三个空洞卷积分支的膨胀率, 必须恰好 3 个, 默认 (6, 12, 18)
        dropout: 投影之后的 Dropout 概率, 默认 0.5
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        atrous_rates: tuple[int, int, int] = (6, 12, 18),
        dropout: float = 0.5,
    ) -> None:
        super().__init__()

        rate1, rate2, rate3 = atrous_rates

        # branch1: 1x1 卷积
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        # branch2: 3x3 空洞卷积, padding 等于 dilation, 所以空间尺寸不变
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=rate1, dilation=rate1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        # branch3: 3x3 空洞卷积
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=rate2, dilation=rate2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        # branch4: 3x3 空洞卷积
        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=rate3, dilation=rate3, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        # branch5: 图像级池化, 先全局平均池化到 1x1, 再用 1x1 卷积处理
        self.image_pool = nn.AdaptiveAvgPool2d(1)
        self.branch5_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        # 拼接 5 个分支, 通道数为 5 * out_channels, 投影回 out_channels
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4 = self.branch4(x)

        pooled = self.image_pool(x)
        pooled = self.branch5_conv(pooled)
        height = x.shape[2]
        width = x.shape[3]
        out5 = F.interpolate(pooled, size=(height, width), mode="bilinear", align_corners=False)

        out = torch.cat([out1, out2, out3, out4, out5], dim=1)
        return self.project(out)
