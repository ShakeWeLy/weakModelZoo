import torch
import torch.nn as nn

from src.models.segment.AttentionGateUnet.AttentionGateUnet import AttentionGateUnet, AttentionGateUp


def test_attention_gate_up_forward_shape():
    layer = AttentionGateUp(in_channels=16, out_channels=8)
    x_input = torch.randn(2, 16, 16, 16)
    x_skip = torch.randn(2, 8, 32, 32)

    output = layer(x_input, x_skip)

    assert output.shape == (2, 8, 32, 32)
    assert torch.isfinite(output).all()


def test_attention_gate_up_aligns_odd_spatial_dimensions():
    layer = AttentionGateUp(in_channels=16, out_channels=8)
    x_input = torch.randn(1, 16, 8, 9)
    x_skip = torch.randn(1, 8, 17, 19)

    output = layer(x_input, x_skip)

    assert output.shape == (1, 8, 17, 19)


def test_attention_gate_up_gates_skip_connection():
    layer = AttentionGateUp(in_channels=8, out_channels=4, gate_hidden_channels=2)
    x_input = torch.zeros(1, 8, 4, 4)
    x_skip = torch.ones(1, 4, 8, 8)

    output = layer(x_input, x_skip)

    assert output.shape == (1, 4, 8, 8)


def test_attention_gate_unet_forward_preserves_spatial_size():
    model = AttentionGateUnet(in_channels=1, out_channels=3, hidden_channels=4)
    x = torch.randn(2, 1, 65, 67)

    output = model(x)

    assert output.shape == (2, 3, 65, 67)
    assert torch.isfinite(output).all()


def test_attention_gate_unet_uses_attention_up_layers():
    model = AttentionGateUnet(in_channels=1, out_channels=2, hidden_channels=4)

    assert isinstance(model.up1, AttentionGateUp)
    assert isinstance(model.up2, AttentionGateUp)
    assert isinstance(model.up3, AttentionGateUp)
    assert isinstance(model.up4, AttentionGateUp)
    assert isinstance(model.up1.up, nn.ConvTranspose2d)
