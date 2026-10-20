import torch.nn as nn
import torch

from src.module.conv import DWConv, BaseConv
from src.module.feature.LEM import LEM
from src.module.feature.sam3 import SAM3FeatureExtractor
from src.models.common.ResNet import ResNet34


class UCB_Module(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UCB_Module, self).__init__()
        self.up_1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dw = DWConv(in_channels, out_channels)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = BaseConv(in_channels, out_channels, kernel_size=1, padding=1, act=self.relu, bn=False)  # ? 什么时候需要bn?

    def forward(self, x):
        x = self.up_1(x)
        x = self.up_2(x)
        x = self.dw(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv(x)
        return x


class FFB_Module(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FFB_Module, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=1, padding=1)  # ? 不确定
        self.fc = nn.Linear(in_channels, out_channels)

    def forward(self, c, t):
        res = torch.cat([c, t], dim=1)
        x = self.maxpool(res)
        x = self.fc(x)
        x = torch.add(x, res)
        return x


class LCM_Module(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(LCM_Module, self).__init__()
        self.hidden_channels = hidden_channels
        self.fc = nn.Linear(in_channels, hidden_channels*3)
        
        self.ds_conv_q = DWConv(hidden_channels, hidden_channels)
        self.ds_conv_k = DWConv(hidden_channels, hidden_channels)
        self.ds_conv_v = DWConv(hidden_channels, hidden_channels)
        self.linear = nn.Linear(hidden_channels, in_channels)

    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape

        def to_map(t: torch.Tensor):
            return t.transpose(1, 2).contiguous().view(B, self.hidden_channels, H, W)

        def to_seq(t: torch.Tensor):
            return t.contiguous().view(B, self.hidden_channels, -1).transpose(1, 2)
        
        x_seq = x.view(B, C, -1).transpose(1, 2)
        q, k, v = torch.split(self.fc(x_seq), self.hidden_channels, dim=-1)

        q = to_seq(self.ds_conv_q(to_map(q)))  
        k = to_seq(self.ds_conv_k(to_map(k)))
        v = to_seq(self.ds_conv_v(to_map(v)))
        attention = torch.matmul(q, k.transpose(-2, -1)) / (self.hidden_channels**0.5)
        attention = torch.softmax(attention, dim=-1)
        x = torch.matmul(attention, v)
        x = self.linear(x)
        x = x.view(B, C, H, W).transpose(1, 2).contiguous()
        return x


class SLR_Unet(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels):
        super(SLR_Unet, self).__init__()
        self.resnet = ResNet34()
        # ckpt = torch.load("resnet34.pth", map_location="cpu")  # 占位

        self.transformer = SAM3FeatureExtractor(
                                                mode="neck",
                                                checkpoint_path="path/to/sam3.pt",  # 或 load_from_hf=True
                                                freeze=True,   # load 后 requires_grad=False，并 eval()
                                            )
            

        self.ucb_1 = UCB_Module(in_channels, hidden_channels)
        self.ucb_2 = UCB_Module(hidden_channels, hidden_channels)
        self.ucb_3 = UCB_Module(hidden_channels, hidden_channels)

        self.lcm_1 = LCM_Module(hidden_channels, hidden_channels)
        self.lcm_2 = LCM_Module(hidden_channels, hidden_channels)
        self.lcm_3 = LCM_Module(hidden_channels, hidden_channels)
        self.lcm_4 = LCM_Module(hidden_channels, hidden_channels)

        self.ffb_1 = FFB_Module(hidden_channels, hidden_channels)
        self.ffb_2 = FFB_Module(hidden_channels, hidden_channels)
        self.ffb_3 = FFB_Module(hidden_channels, hidden_channels)
        self.ffb_4 = FFB_Module(hidden_channels, hidden_channels)

        self.lem = LEM(in_channels, hidden_channels)

    def forward(self, x: torch.Tensor):
        x = self.resnet.layer1(x)   # stem: 224→56, C=64（一般不用作 skip）
        c1, c2, c3, c4 = self.resnet.forward_features(x)