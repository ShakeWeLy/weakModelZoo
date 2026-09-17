import pytest
import torch

from src.module.attention.SelfAttention.multiHeadAttention import MultiHeadAttention
from src.module.transformer.transformerBlock import TransformerDecoderBlock


def test_multi_head_attention_output_shape():
    layer = MultiHeadAttention(embed_dim=64, num_heads=8)
    x = torch.randn(2, 16, 64)

    output = layer(x)

    assert output.shape == x.shape
    assert torch.isfinite(output).all()


def test_multi_head_attention_rejects_invalid_head_count():
    with pytest.raises(ValueError, match="divisible"):
        MultiHeadAttention(embed_dim=64, num_heads=5)


def test_transformer_decoder_block_output_shape():
    block = TransformerDecoderBlock(embed_dim=64, num_heads=8, ffn_hidden_channels=128)
    x = torch.randn(2, 16, 64)

    output = block(x)

    assert output.shape == x.shape
    assert torch.isfinite(output).all()
