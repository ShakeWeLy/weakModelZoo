import torch.nn as nn
import torch
import torch.nn.functional as F

from src.module.conv import DWConv, BaseConv
from src.module.feature.LEM import LEM
from src.module.feature.sam3 import SAM3FeatureExtractor
from src.models.common.ResNet import ResNet34


class UCB_Module(nn.Module):
    """2× 上采样 + DWConv，对齐相邻 skip 的分辨率。"""

    def __init__(self, in_channels, out_channels):
        super(UCB_Module, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.dw = DWConv(in_channels, out_channels)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        # DW 之后通道已是 out_channels；1×1 用 padding=0
        self.conv = BaseConv(out_channels, out_channels, kernel_size=1, padding=0, act=False, bn=False)

    def forward(self, x):
        x = self.up(x)
        x = self.dw(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv(x)
        return x


class FFB_Module(nn.Module):
    def __init__(self, in_channels, dim):
        super(FFB_Module, self).__init__()
        # in_channels = C_res + C_sam（cat 之后的通道数）
        self.fc = nn.Sequential(
            nn.Linear(in_channels, dim),
            nn.Linear(dim, in_channels),
        )

    def forward(self, c, t):
        # ! 这里要保证 t 和 c 的尺寸相同
        if t.shape[-2:] != c.shape[-2:]:
            t = F.interpolate(t, size=c.shape[-2:], mode="bilinear", align_corners=False)
        res = torch.cat([c, t], dim=1)  # (B, C, H, W)
        # Linear 吃最后一维特征：(B,C,1,1) -> flatten -> (B,C)
        w = self.fc(F.adaptive_max_pool2d(res, 1).flatten(1))
        w = torch.sigmoid(w).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        return w * res


class LCM_Module(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(LCM_Module, self).__init__()
        self.hidden_channels = hidden_channels
        self.fc = nn.Linear(in_channels, hidden_channels * 3)

        self.ds_conv_q = DWConv(hidden_channels, hidden_channels)
        self.ds_conv_k = DWConv(hidden_channels, hidden_channels)
        self.ds_conv_v = DWConv(hidden_channels, hidden_channels)
        self.linear = nn.Linear(hidden_channels, in_channels)

    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape

        def to_map(t: torch.Tensor):
            return t.transpose(1, 2).reshape(B, self.hidden_channels, H, W)

        def to_seq(t: torch.Tensor):
            return t.reshape(B, self.hidden_channels, -1).transpose(1, 2)

        x_seq = x.reshape(B, C, -1).transpose(1, 2)
        q, k, v = torch.split(self.fc(x_seq), self.hidden_channels, dim=-1)

        q = to_seq(self.ds_conv_q(to_map(q)))
        k = to_seq(self.ds_conv_k(to_map(k)))
        v = to_seq(self.ds_conv_v(to_map(v)))
        attention = torch.matmul(q, k.transpose(-2, -1)) / (self.hidden_channels**0.5)
        attention = torch.softmax(attention, dim=-1)
        x = torch.matmul(attention, v)
        x = self.linear(x)
        # (B, HW, C) -> (B, C, H, W)
        x = x.transpose(1, 2).reshape(B, C, H, W)
        return x


class SLR_Unet(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels):
        super(SLR_Unet, self).__init__()
        self.resnet = ResNet34()
        # ResNet34 stage: 64/128/256/512；SAM3 neck 每级均为 256
        self.resnet_channels = (64, 128, 256, 512)
        self.sam_channels = SAM3FeatureExtractor.NECK_CHANNELS  # 256
        self.fused_channels = tuple(c + self.sam_channels for c in self.resnet_channels)
        # 320, 384, 512, 768

        self.transformer = SAM3FeatureExtractor(
            mode="neck",
            checkpoint_path=None,  # 换成真实 sam3.pt，或 load_from_hf=True
            freeze=True,
        )

        f1, f2, f3, f4 = self.fused_channels
        # 解码器按 skip 通道递进：UCB 输出对齐下一层 fused，再 cat
        self.lem = LEM(f4, f4)
        self.ucb_3 = UCB_Module(f4, f3)          # 768 -> 512
        self.ucb_2 = UCB_Module(f3 * 2, f2)      # 1024 -> 384
        self.ucb_1 = UCB_Module(f2 * 2, f1)      # 768 -> 320

        self.ffb_1 = FFB_Module(f1, hidden_channels)
        self.ffb_2 = FFB_Module(f2, hidden_channels)
        self.ffb_3 = FFB_Module(f3, hidden_channels)

        self.lcm_4 = LCM_Module(f3, hidden_channels)       # ucb_3
        self.lcm_3 = LCM_Module(f2, hidden_channels)       # ucb_2
        self.lcm_2 = LCM_Module(f1, hidden_channels)       # ucb_1
        self.lcm_1 = LCM_Module(f1 * 2, hidden_channels)   # up_f1 = cat(ucb_1, fused0)

        self.head = nn.Conv2d(f3 + f2 + f1 + f1 * 2, out_channels, kernel_size=1)

    @staticmethod
    def _align_cat(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if a.shape[-2:] != b.shape[-2:]:
            b = F.interpolate(b, size=a.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat([a, b], dim=1)

    def forward(self, x: torch.Tensor):
        c_feats = self.resnet.forward_features(x)  # list[4]
        t_feats = self.transformer(x).neck_pyramid  # list[4], C=256
        assert t_feats is not None

        ffbs = (self.ffb_1, self.ffb_2, self.ffb_3)  # 最深层用 LEM，不用 ffb_4
        fused = [ffb(c, t) for ffb, c, t in zip(ffbs, c_feats, t_feats)]

        t4 = t_feats[3]
        if t4.shape[-2:] != c_feats[3].shape[-2:]:
            t4 = F.interpolate(t4, size=c_feats[3].shape[-2:], mode="bilinear", align_corners=False)
        fused_4 = torch.cat([c_feats[3], t4], dim=1)  # 512 + 256

        up_f4 = self.lem(fused_4)
        ucb_3 = self.ucb_3(up_f4)
        up_f3 = self._align_cat(ucb_3, fused[2])

        ucb_2 = self.ucb_2(up_f3)
        up_f2 = self._align_cat(ucb_2, fused[1])

        ucb_1 = self.ucb_1(up_f2)
        up_f1 = self._align_cat(ucb_1, fused[0])

        lcm_4 = self.lcm_4(ucb_3)
        lcm_3 = self.lcm_3(ucb_2)
        lcm_2 = self.lcm_2(ucb_1)
        lcm_1 = self.lcm_1(up_f1)

        # 多尺度 LCM 空间尺寸不同，统一到最高分辨率再 cat
        target_hw = lcm_1.shape[-2:]
        lcm_feats = [
            F.interpolate(f, size=target_hw, mode="bilinear", align_corners=False)
            if f.shape[-2:] != target_hw
            else f
            for f in (lcm_4, lcm_3, lcm_2, lcm_1)
        ]
        x = torch.cat(lcm_feats, dim=1)
        return self.head(x)
