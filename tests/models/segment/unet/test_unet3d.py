import torch
import torch.nn as nn

from src.models.segment.unet.unet3d import UNet3D


def test_unet3d_forward_preserves_volume_size():
    model = UNet3D(in_channels=1, out_channels=3, hidden_channels=2)
    x = torch.randn(1, 1, 17, 19, 21)

    output = model(x)

    assert output.shape == (1, 3, 17, 19, 21)


def test_unet3d_uses_3d_layers():
    model = UNet3D(in_channels=1, out_channels=2, hidden_channels=2)

    assert isinstance(model.down1.max_pool, nn.MaxPool3d)
    assert isinstance(model.up1.up, nn.ConvTranspose3d)
    assert isinstance(model.outc.conv, nn.Conv3d)
