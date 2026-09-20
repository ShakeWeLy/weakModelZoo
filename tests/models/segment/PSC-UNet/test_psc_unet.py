import importlib.util
from pathlib import Path

import torch


def _load_psc_unet():
    module_path = Path("src/models/segment/PSC-UNet/PSC-UNet.py")
    spec = importlib.util.spec_from_file_location("psc_unet_module", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.PSC_UNet


def test_psc_unet_forward_preserves_spatial_size():
    model = _load_psc_unet()(in_channels=3, out_channels=4, base_dim=16)
    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    assert y.shape == (1, 4, 224, 224)
