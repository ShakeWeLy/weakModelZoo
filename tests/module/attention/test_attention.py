import pytest
import torch
import torch.nn as nn

from src.module.attention.CAM.cam import CAM
from src.module.attention.SAM.sam import SAM


def test_cam_returns_channel_reweighted_features():
    layer = CAM(in_channels=8, reduction_ratio=4)
    x = torch.randn(2, 8, 16, 16)

    output = layer(x)

    assert output.shape == x.shape
    assert torch.isfinite(output).all()


def test_cam_supports_disabled_and_custom_activation():
    x = torch.randn(2, 8, 8, 8)

    without_activation = CAM(8, act=False)
    custom_activation = CAM(8, act=nn.LeakyReLU(0.1))

    assert without_activation(x).shape == x.shape
    assert custom_activation(x).shape == x.shape


@pytest.mark.parametrize("kernel_size", [3, 7])
def test_sam_returns_spatially_reweighted_features(kernel_size):
    layer = SAM(kernel_size=kernel_size)
    x = torch.randn(2, 8, 16, 16)

    output = layer(x)

    assert output.shape == x.shape
    assert layer.conv.block[0].in_channels == 2
    assert layer.conv.block[0].out_channels == 1
    assert torch.isfinite(output).all()


def test_sam_rejects_unsupported_kernel_size():
    with pytest.raises(ValueError):
        SAM(kernel_size=5)
