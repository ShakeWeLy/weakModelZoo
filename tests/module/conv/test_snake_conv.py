import torch

from src.module.conv import SnakeConv


def test_snake_conv_x_axis_preserves_spatial_size():
    layer = SnakeConv(16, 16, kernel_size=3, morph=0)
    x = torch.randn(2, 16, 32, 32)
    y = layer(x)
    assert y.shape == (2, 16, 32, 32)


def test_snake_conv_y_axis_preserves_spatial_size():
    layer = SnakeConv(8, 8, kernel_size=3, morph=1)
    x = torch.randn(1, 8, 24, 24)
    y = layer(x)
    assert y.shape == (1, 8, 24, 24)


def test_snake_conv_without_offset_runs():
    layer = SnakeConv(4, 4, kernel_size=3, morph=0, if_offset=False)
    x = torch.randn(1, 4, 16, 16)
    y = layer(x)
    assert y.shape == (1, 4, 16, 16)
