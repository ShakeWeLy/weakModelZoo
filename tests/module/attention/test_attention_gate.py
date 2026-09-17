import pytest
import torch

from src.module.attention.AttentionGate.attentionGate import AttentionGate


def test_attention_gate_output_shape_matches_x():
    layer = AttentionGate(g_in_channels=64, x_in_channels=64, hidden_channels=32)
    g = torch.randn(2, 64, 32, 32)
    x = torch.randn(2, 64, 32, 32)

    output = layer(g, x)

    assert output.shape == x.shape
    assert torch.isfinite(output).all()


def test_attention_gate_attention_is_bounded_by_x():
    layer = AttentionGate(g_in_channels=8, x_in_channels=8, hidden_channels=4)
    g = torch.randn(1, 8, 16, 16)
    x = torch.randn(1, 8, 16, 16)

    output = layer(g, x)

    assert output.abs().le(x.abs() + 1e-6).all()


def test_attention_gate_supports_different_channel_sizes():
    layer = AttentionGate(g_in_channels=128, x_in_channels=64, hidden_channels=32)
    g = torch.randn(2, 128, 20, 20)
    x = torch.randn(2, 64, 20, 20)

    output = layer(g, x)

    assert output.shape == (2, 64, 20, 20)


def test_attention_gate_requires_matching_spatial_size():
    layer = AttentionGate(g_in_channels=16, x_in_channels=16, hidden_channels=8)
    g = torch.randn(1, 16, 32, 32)
    x = torch.randn(1, 16, 31, 32)

    with pytest.raises(ValueError, match="空间尺寸必须一致"):
        layer(g, x)


def test_attention_gate_preserves_gradients():
    layer = AttentionGate(g_in_channels=16, x_in_channels=16, hidden_channels=8)
    g = torch.randn(1, 16, 8, 8, requires_grad=True)
    x = torch.randn(1, 16, 8, 8, requires_grad=True)

    output = layer(g, x).sum()
    output.backward()

    assert g.grad is not None
    assert x.grad is not None
