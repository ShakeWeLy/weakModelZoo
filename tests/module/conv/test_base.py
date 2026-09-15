import pytest
import torch
import torch.nn as nn

from src.module.conv import BaseConv, DepthwiseConv


def test_base_conv_with_custom_bn_and_activation():
    layer = BaseConv(
        3,
        32,
        3,
        bn=nn.BatchNorm2d(32, eps=1e-3, momentum=0.03),
        act=nn.LeakyReLU(0.1, inplace=True),
    )

    output = layer(torch.randn(2, 3, 16, 16))

    assert output.shape == (2, 32, 16, 16)
    assert isinstance(layer.block[1], nn.BatchNorm2d)
    assert layer.block[1].eps == pytest.approx(1e-3)
    assert layer.block[1].momentum == pytest.approx(0.03)
    assert isinstance(layer.block[2], nn.LeakyReLU)
    assert layer.block[2].negative_slope == pytest.approx(0.1)


@pytest.mark.parametrize(
    ("bn", "act", "bn_type", "act_type"),
    [
        (True, True, nn.BatchNorm2d, nn.SiLU),
        (False, False, nn.Identity, nn.Identity),
    ],
)
def test_base_conv_default_options(bn, act, bn_type, act_type):
    layer = BaseConv(3, 8, 3, bn=bn, act=act)

    output = layer(torch.randn(2, 3, 8, 8))

    assert output.shape == (2, 8, 8, 8)
    if bn:
        assert isinstance(layer.block[1], bn_type)
    if act:
        assert isinstance(layer.block[-1], act_type)


def test_base_conv_rejects_invalid_groups():
    with pytest.raises(ValueError):
        BaseConv(3, 8, 3, groups=2)


def test_depthwise_conv():
    layer = DepthwiseConv(8, kernel_size=3)
    output = layer(torch.randn(2, 8, 16, 16))

    assert output.shape == (2, 8, 16, 16)
    assert layer.block[0].groups == 8
    assert layer.block[0].in_channels == layer.block[0].out_channels == 8
