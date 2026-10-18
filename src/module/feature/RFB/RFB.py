import torch
import torch.nn as nn


class RFB(nn.Module):
    """
    RFB 感受野模块 (Receptive Field Block, ECCV 2018, Liu et al.)。

    思路: 借鉴 Inception 的多分支结构, 但每个分支末尾使用空洞卷积 (dilation = rate),
    用较小的卷积核获得较大的感受野, 并模拟人眼视网膜中不同偏心率的感受野大小。

    结构 (对应论文 Figure 3(a)):
        branch0: 1x1 卷积 -> 3x3 空洞卷积 (rate=1)
        branch1: 1x1 卷积 -> 3x3 卷积     -> 3x3 空洞卷积 (rate=3)
        branch2: 1x1 卷积 -> 5x5 卷积     -> 3x3 空洞卷积 (rate=5)
        shortcut: 1x1 卷积 (残差连接)

    计算过程:
        cat(branch0, branch1, branch2) -> 1x1 卷积 (线性融合) -> * scale -> + shortcut -> ReLU

    Args:
        in_channels: 输入通道数, 必须不小于 map_reduce
        out_channels: 输出通道数
        scale: 线性融合分支的缩放系数, 与官方实现一致, 默认 0.1
        map_reduce: 1x1 降维的倍数, 中间通道数 = in_channels // map_reduce, 默认 8
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scale: float = 0.1,
        map_reduce: int = 8,
    ) -> None:
        super().__init__()

        inter_channels = in_channels // map_reduce
        if inter_channels < 1:
            raise ValueError("in_channels 必须不小于 map_reduce")

        self.scale = scale

        # ---------- branch0: 1x1 -> 3x3 (rate=1) ----------
        self.branch0_reduce = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
        )
        self.branch0_dilated = nn.Sequential(
            nn.Conv2d(inter_channels, out_channels, kernel_size=3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        # ---------- branch1: 1x1 -> 3x3 -> 3x3 (rate=3) ----------
        self.branch1_reduce = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
        )
        self.branch1_conv = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
        )
        self.branch1_dilated = nn.Sequential(
            nn.Conv2d(inter_channels, out_channels, kernel_size=3, padding=3, dilation=3, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        # ---------- branch2: 1x1 -> 5x5 -> 3x3 (rate=5) ----------
        self.branch2_reduce = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
        )
        self.branch2_conv = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
        )
        self.branch2_dilated = nn.Sequential(
            nn.Conv2d(inter_channels, out_channels, kernel_size=3, padding=5, dilation=5, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        # ---------- 拼接后的 1x1 线性融合 (不使用 ReLU, 与论文一致) ----------
        self.linear = nn.Sequential(
            nn.Conv2d(out_channels * 3, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        # ---------- shortcut: 1x1 卷积 (不使用 ReLU) ----------
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # branch0
        b0_reduced = self.branch0_reduce(x)
        b0_out = self.branch0_dilated(b0_reduced)

        # branch1
        b1_reduced = self.branch1_reduce(x)
        b1_mid = self.branch1_conv(b1_reduced)
        b1_out = self.branch1_dilated(b1_mid)

        # branch2
        b2_reduced = self.branch2_reduce(x)
        b2_mid = self.branch2_conv(b2_reduced)
        b2_out = self.branch2_dilated(b2_mid)

        # 拼接 -> 线性融合 -> 缩放
        concat = torch.cat([b0_out, b1_out, b2_out], dim=1)
        linear_out = self.linear(concat) * self.scale

        # 残差连接 + ReLU
        shortcut_out = self.shortcut(x)
        return self.relu(linear_out + shortcut_out)
