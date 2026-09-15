import pytest
import torch
import torch.nn as nn

from src.module.conv import DepthwiseConv, PointwiseConv, DWConv


def test_depthwise_conv():
    layer = DepthwiseConv(8, kernel_size=3)
    output = layer(torch.randn(2, 8, 16, 16))

    assert output.shape == (2, 8, 16, 16)
    assert layer.block[0].groups == 8
    assert layer.block[0].in_channels == layer.block[0].out_channels == 8


def test_pointwise_conv():
    layer = PointwiseConv(8, 16)
    output = layer(torch.randn(2, 8, 16, 16))

    assert output.shape == (2, 16, 16, 16)
    assert layer.block[0].in_channels == 8
    assert layer.block[0].out_channels == 16
    assert layer.block[0].kernel_size == (1, 1)


def test_dwconv():
    layer = DWConv(8, 16)
    output = layer(torch.randn(2, 8, 16, 16))

    assert output.shape == (2, 16, 16, 16)
    assert layer.depthwise_conv.block[0].groups == 8
    assert layer.depthwise_conv.block[0].in_channels == layer.depthwise_conv.block[0].out_channels == 8
    assert layer.pointwise_conv.block[0].in_channels == 8
    assert layer.pointwise_conv.block[0].out_channels == 16
    assert layer.pointwise_conv.block[0].kernel_size == (1, 1)