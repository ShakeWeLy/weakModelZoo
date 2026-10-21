from pathlib import Path

import torch.nn as nn
import torch
import torch.nn.functional as F

from src.module.conv import DSConv, BaseConv
from src.module.feature.LEM import LEM
from src.module.feature.sam3 import SAM3FeatureExtractor
from src.models.common.ResNet import TorchvisionResNetEncoder

_REPO_ROOT = Path(__file__).resolve().parents[4]


def _resolve_weight_path(path: str | None) -> str | None:
    if path is None:
        return None
    text = str(path).strip()
    if not text:
        return None
    p = Path(text)
    if not p.is_absolute():
        p = _REPO_ROOT / p
    return str(p.resolve())


class UCB_Module(nn.Module):
    """2× 上采样 + DWConv，对齐相邻 skip 的分辨率。"""

    def __init__(self, in_channels, out_channels):
        super(UCB_Module, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.dw = DSConv(in_channels, out_channels)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
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
        self.fc = nn.Sequential(
            nn.Linear(in_channels, dim),
            nn.Linear(dim, in_channels),
        )

    def forward(self, c, t):
        if t.shape[-2:] != c.shape[-2:]:
            t = F.interpolate(t, size=c.shape[-2:], mode="bilinear", align_corners=False)
        res = torch.cat([c, t], dim=1)
        w = self.fc(F.adaptive_max_pool2d(res, 1).flatten(1))
        w = torch.sigmoid(w).unsqueeze(-1).unsqueeze(-1)
        return w * res


class LCM_Module(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(LCM_Module, self).__init__()
        self.hidden_channels = hidden_channels
        self.fc = nn.Linear(in_channels, hidden_channels * 3)

        self.ds_conv_q = DSConv(hidden_channels, hidden_channels)
        self.ds_conv_k = DSConv(hidden_channels, hidden_channels)
        self.ds_conv_v = DSConv(hidden_channels, hidden_channels)
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
        x = x.transpose(1, 2).reshape(B, C, H, W)
        return x


class SLR_Unet(nn.Module):
    """双编码器 SLR-UNet：官方 ResNet（预训练可训）+ SAM3 neck（预训练冻结）。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int = 64,
        *,
        resnet_name: str = "resnet34",
        resnet_pretrained: bool = True,
        resnet_weights: str | None = None,
        sam_checkpoint: str | None = None,
        sam_load_from_hf: bool = False,
        sam_auto_download: bool = True,
        sam_hf_version: str = "sam3",
        freeze_sam: bool = True,
        sam_resolution: int = 1008,
    ):
        super(SLR_Unet, self).__init__()
        resnet_weights = _resolve_weight_path(resnet_weights)
        # 官方 ResNet：优先本地；缺失且 pretrained=True 时 torchvision 自动下载
        self.resnet = TorchvisionResNetEncoder(
            name=resnet_name,  # type: ignore[arg-type]
            pretrained=resnet_pretrained,
            in_channels=in_channels,
            weights_path=resnet_weights,
        )
        self.resnet_channels = self.resnet.out_channels
        self.sam_channels = SAM3FeatureExtractor.NECK_CHANNELS
        self.fused_channels = tuple(c + self.sam_channels for c in self.resnet_channels)

        # SAM3：本地优先；缺失则 ModelScope 自动下载（可关），再退回 HF
        ckpt = _resolve_weight_path(sam_checkpoint)
        if ckpt is not None and not Path(ckpt).is_file():
            if sam_auto_download:
                import importlib.util

                script = _REPO_ROOT / "pretrained_weights" / "download_sam3.py"
                spec = importlib.util.spec_from_file_location("slr_download_sam3", script)
                if spec is None or spec.loader is None:
                    raise ImportError(f"无法加载自动下载脚本: {script}")
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                ckpt = str(mod.ensure_sam3_checkpoint(ckpt))
            elif sam_load_from_hf:
                print(f"[pretrained] SAM 本地缺失 {ckpt}，改用 HuggingFace 下载")
                ckpt = None
            else:
                raise FileNotFoundError(
                    f"SAM 权重不存在: {ckpt}；可设 sam_auto_download=true "
                    "或运行 python pretrained_weights/download_sam3.py"
                )
        self.transformer = SAM3FeatureExtractor(
            mode="neck",
            checkpoint_path=ckpt,
            load_from_hf=bool(sam_load_from_hf) and ckpt is None,
            hf_version=sam_hf_version,  # type: ignore[arg-type]
            resolution=sam_resolution,
            freeze=freeze_sam,
        )

        f1, f2, f3, f4 = self.fused_channels
        self.lem = LEM(f4, f4)
        self.ucb_3 = UCB_Module(f4, f3)
        self.ucb_2 = UCB_Module(f3 * 2, f2)
        self.ucb_1 = UCB_Module(f2 * 2, f1)

        self.ffb_1 = FFB_Module(f1, hidden_channels)
        self.ffb_2 = FFB_Module(f2, hidden_channels)
        self.ffb_3 = FFB_Module(f3, hidden_channels)

        self.lcm_4 = LCM_Module(f3, hidden_channels)
        self.lcm_3 = LCM_Module(f2, hidden_channels)
        self.lcm_2 = LCM_Module(f1, hidden_channels)
        self.lcm_1 = LCM_Module(f1 * 2, hidden_channels)

        self.head = nn.Conv2d(f3 + f2 + f1 + f1 * 2, out_channels, kernel_size=1)

    @staticmethod
    def _align_cat(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if a.shape[-2:] != b.shape[-2:]:
            b = F.interpolate(b, size=a.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat([a, b], dim=1)

    def forward(self, x: torch.Tensor):
        input_hw = x.shape[-2:]
        c_feats = self.resnet.forward_features(x)
        t_feats = self.transformer(x).neck_pyramid
        assert t_feats is not None

        ffbs = (self.ffb_1, self.ffb_2, self.ffb_3)
        fused = [ffb(c, t) for ffb, c, t in zip(ffbs, c_feats, t_feats)]

        t4 = t_feats[3]
        if t4.shape[-2:] != c_feats[3].shape[-2:]:
            t4 = F.interpolate(t4, size=c_feats[3].shape[-2:], mode="bilinear", align_corners=False)
        fused_4 = torch.cat([c_feats[3], t4], dim=1)

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

        target_hw = lcm_1.shape[-2:]
        lcm_feats = [
            F.interpolate(f, size=target_hw, mode="bilinear", align_corners=False)
            if f.shape[-2:] != target_hw
            else f
            for f in (lcm_4, lcm_3, lcm_2, lcm_1)
        ]
        logits = self.head(torch.cat(lcm_feats, dim=1))
        # 对齐到输入分辨率，便于与全尺寸 label 计算 loss
        if logits.shape[-2:] != input_hw:
            logits = F.interpolate(logits, size=input_hw, mode="bilinear", align_corners=False)
        return logits
