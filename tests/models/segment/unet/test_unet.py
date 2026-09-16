import torch
import torch.nn as nn

from src.models.segment.unet.unet import Down, UNet, Up


def test_unet_forward_preserves_spatial_size():
    model = UNet(in_channels=1, out_channels=3, hidden_channels=4)
    x = torch.randn(2, 1, 65, 67)

    output = model(x)

    assert output.shape == (2, 3, 65, 67)


def test_unet_uses_2d_layers():
    model = UNet(in_channels=1, out_channels=2, hidden_channels=4)

    assert isinstance(model.down1.max_pool, nn.MaxPool2d)
    assert isinstance(model.up1.up, nn.ConvTranspose2d)


def test_up_aligns_odd_spatial_dimensions():
    layer = Up(in_channels=16, out_channels=8)
    x_input = torch.randn(1, 16, 8, 9)
    x_skip = torch.randn(1, 8, 17, 19)

    output = layer(x_input, x_skip)

    assert output.shape == (1, 8, 17, 19)
