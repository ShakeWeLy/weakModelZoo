import pytest
import torch

from src.module.transformer.swinTransformer.swinTransformer4D import (
    PatchExpanding4D,
    PatchMerging4D,
    ShiftedWindowAttention4D,
    SwinTransformerBlock4D,
    WindowAttentionLayer4D,
    build_shifted_window_attn_mask_4D,
    window_partition_4D,
    window_reverse_4D,
)


def test_window_partition_and_reverse_4d():
    x = torch.randn(2, 4, 4, 4, 4, 32)
    windows = window_partition_4D(x, window_size=2)
    restored = window_reverse_4D(
        windows, window_size=2, time=4, height=4, width=4, depth=4, batch_size=2
    )

    # 2 * (2^4) windows, 2^4 tokens
    assert windows.shape == (32, 16, 32)
    assert restored.shape == x.shape
    assert torch.allclose(restored, x)


def test_window_attention_preserves_shape_4d():
    layer = WindowAttentionLayer4D(embed_dim=32, window_size=2, num_heads=4)
    x = torch.randn(1, 4, 4, 4, 4, 32)

    output = layer(x)

    assert output.shape == x.shape
    assert torch.isfinite(output).all()


def test_window_attention_rejects_invalid_size_4d():
    layer = WindowAttentionLayer4D(embed_dim=32, window_size=2, num_heads=4)
    x = torch.randn(1, 3, 4, 4, 4, 32)

    with pytest.raises(ValueError, match="整除"):
        layer(x)


def test_shifted_window_attn_mask_shape_4d():
    mask = build_shifted_window_attn_mask_4D(
        4, 4, 4, 4, window_size=2, shift_size=1, device="cpu", dtype=torch.float32
    )

    assert mask.shape == (16, 16, 16)  # 2^4 windows, 2^4 tokens


def test_shifted_window_attention_preserves_shape_4d():
    layer = ShiftedWindowAttention4D(
        embed_dim=32, window_size=2, num_heads=4, shift_size=1
    )
    x = torch.randn(1, 4, 4, 4, 4, 32)

    output = layer(x)

    assert output.shape == x.shape
    assert torch.isfinite(output).all()


def test_patch_merging_downsamples_4d():
    layer = PatchMerging4D(embed_dim=32)
    x = torch.randn(1, 4, 4, 4, 4, 32)

    output = layer(x)

    assert output.shape == (1, 2, 2, 2, 2, 64)
    assert torch.isfinite(output).all()


def test_patch_expanding_upsamples_4d():
    layer = PatchExpanding4D(embed_dim=64)
    x = torch.randn(1, 2, 2, 2, 2, 64)

    output = layer(x)

    assert output.shape == (1, 4, 4, 4, 4, 32)
    assert torch.isfinite(output).all()


def test_swin_transformer_block_preserves_shape_4d():
    block = SwinTransformerBlock4D(embed_dim=32, num_heads=4, window_size=2, shift_size=0)
    x = torch.randn(1, 4, 4, 4, 4, 32)

    output = block(x)

    assert output.shape == x.shape
    assert torch.isfinite(output).all()


def test_swin_transformer_block_with_shift_4d():
    block = SwinTransformerBlock4D(embed_dim=32, num_heads=4, window_size=2, shift_size=1)
    x = torch.randn(1, 4, 4, 4, 4, 32)

    output = block(x)

    assert output.shape == x.shape
    assert torch.isfinite(output).all()
