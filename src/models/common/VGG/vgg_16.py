import torch
import torch.nn as nn
import torch.nn.functional as F

from src.module.conv import BaseConv

class VGG16(nn.Module):
    def __init__(self, num_classes=1000):
        super(VGG16, self).__init__()
        self.layer1 = nn.Sequential(
            BaseConv(3, 64, kernel_size=3, padding=1, act=nn.ReLU(inplace=True), bn=False),
            BaseConv(64, 64, kernel_size=3, padding=1, act=nn.ReLU(inplace=True), bn=False),
        )
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer2 = nn.Sequential(
            BaseConv(64, 128, kernel_size=3, padding=1, act=nn.ReLU(inplace=True), bn=False),
            BaseConv(128, 128, kernel_size=3, padding=1, act=nn.ReLU(inplace=True), bn=False),
        )
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer3 = nn.Sequential(
            BaseConv(128, 256, kernel_size=3, padding=1, act=nn.ReLU(inplace=True), bn=False),
            BaseConv(256, 256, kernel_size=3, padding=1, act=nn.ReLU(inplace=True), bn=False),
            BaseConv(256, 256, kernel_size=3, padding=1, act=nn.ReLU(inplace=True), bn=False),
        )
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer4 = nn.Sequential(
            BaseConv(256, 512, kernel_size=3, padding=1, act=nn.ReLU(inplace=True), bn=False),
            BaseConv(512, 512, kernel_size=3, padding=1, act=nn.ReLU(inplace=True), bn=False),
            BaseConv(512, 512, kernel_size=3, padding=1, act=nn.ReLU(inplace=True), bn=False),
        )
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer5 = nn.Sequential(
            BaseConv(512, 512, kernel_size=3, padding=1, act=nn.ReLU(inplace=True), bn=False),
            BaseConv(512, 512, kernel_size=3, padding=1, act=nn.ReLU(inplace=True), bn=False),
            BaseConv(512, 512, kernel_size=3, padding=1, act=nn.ReLU(inplace=True), bn=False),
        )
        self.maxpool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p=0.5)
        self.fc6 = nn.Linear(512 * 7 * 7, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, num_classes)

    def forward(self, x):
        b, c, h, w = x.size()
        if h != 224 or w != 224:
            raise ValueError(f"Input size must be 224x224, but got {h}x{w}")
        x = self.layer1(x)
        x = self.maxpool1(x)
        x = self.layer2(x)
        x = self.maxpool2(x)
        x = self.layer3(x)
        x = self.maxpool3(x)
        x = self.layer4(x)
        x = self.maxpool4(x)
        x = self.layer5(x)
        x = self.maxpool5(x)
        # x = x.view(x.size(0), -1)
        x = x.flatten(start_dim=1)  # [B, C, H, W] -> [B, C * H * W]
        x = self.fc6(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc7(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc8(x)
        return x