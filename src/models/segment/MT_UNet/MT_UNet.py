"""官方 MT-UNet 封装（Dootmaan/MT-UNet）：网络实现直接复用仓库根目录下的 ``MT-UNet/model/MTUNet.py``。

接口对齐 project1_2610：``in_channels`` / ``out_channels`` 由框架注入。
MT-UNet 的 stem 固定按 3 通道构建，单通道输入在 ``MTUNet.forward`` 内复制为 3 通道。
网络结构中写死了 224 输入（位置编码 784 = 28×28），因此 ``image_size`` 必须为 224。
本模型从零训练，不加载任何预训练权重。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch.nn as nn

_REPO_ROOT = Path(__file__).resolve().parents[4]
_OFFICIAL_MODEL_FILE = _REPO_ROOT / "MT-UNet" / "model" / "MTUNet.py"
_MODULE_NAME = "mt_unet_official_model"


def _load_official_mtunet():
    module = sys.modules.get(_MODULE_NAME)
    if module is None:
        if not _OFFICIAL_MODEL_FILE.is_file():
            raise FileNotFoundError(f"未找到官方 MT-UNet 实现: {_OFFICIAL_MODEL_FILE}")
        spec = importlib.util.spec_from_file_location(_MODULE_NAME, _OFFICIAL_MODEL_FILE)
        if spec is None or spec.loader is None:
            raise ImportError(f"无法加载官方 MT-UNet: {_OFFICIAL_MODEL_FILE}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[_MODULE_NAME] = module
        spec.loader.exec_module(module)
    return module.MTUNet


class MTUNetOfficial(nn.Module):
    """官方 MT-UNet，输出 ``out_channels`` 类 logits。"""

    def __init__(self, in_channels: int = 1, out_channels: int = 9) -> None:
        super().__init__()
        if in_channels not in (1, 3):
            raise ValueError(f"MT-UNet 仅支持 1 或 3 通道输入，当前为 {in_channels}")
        MTUNet = _load_official_mtunet()
        self.net = MTUNet(out_ch=out_channels)

    def forward(self, x):
        return self.net(x)
