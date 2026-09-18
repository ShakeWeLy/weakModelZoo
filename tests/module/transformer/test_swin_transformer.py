import pytest
import torch

from src.module.transformer.vit.swinTransformer import (
    PatchMerging,
    ShiftedWindowAttention,
    SwinTransformerBlock,
    WindowAttention,
    build_shifted_window_attn_mask,
    window_partition,
    window_reverse,
)


def test_window_partition_and_reverse():
    x = torch.randn(2, 8, 8, 64)
    windows = window_partition(x, window_size=4)
    restored = window_reverse(windows, window_size=4, height=8, width=8, batch_size=2)

    assert windows.shape == (8, 16, 64)
    assert restored.shape == x.shape
    assert torch.allclose(restored, x)


def test_window_attention_preserves_shape():
    layer = WindowAttention(embed_dim=64, window_size=4, num_heads=4)
    x = torch.randn(2, 8, 8, 64)

    output = layer(x)

    assert output.shape == x.shape
    assert torch.isfinite(output).all()


def test_window_attention_rejects_invalid_size():
    layer = WindowAttention(embed_dim=64, window_size=4, num_heads=4)
    x = torch.randn(1, 7, 8, 64)

    with pytest.raises(ValueError, match="整除"):
        layer(x)


def test_shifted_window_attn_mask_shape():
    mask = build_shifted_window_attn_mask(8, 8, window_size=4, shift_size=2, device="cpu", dtype=torch.float32)

    assert mask.shape == (4, 16, 16)


def test_shifted_window_attention_preserves_shape():
    layer = ShiftedWindowAttention(embed_dim=64, window_size=4, num_heads=4, shift_size=2)
    x = torch.randn(2, 8, 8, 64)

    output = layer(x)

    assert output.shape == x.shape
    assert torch.isfinite(output).all()


def test_patch_merging_downsamples_and_doubles_channels():
    layer = PatchMerging(embed_dim=64)
    x = torch.randn(2, 8, 8, 64)

    output = layer(x)

    assert output.shape == (2, 4, 4, 128)
    assert torch.isfinite(output).all()


def test_swin_transformer_block_preserves_shape():
    block = SwinTransformerBlock(embed_dim=64, num_heads=4, window_size=4, shift_size=0)
    x = torch.randn(2, 8, 8, 64)

    output = block(x)

    assert output.shape == x.shape
    assert torch.isfinite(output).all()


def test_swin_transformer_block_with_shift():
    block = SwinTransformerBlock(embed_dim=64, num_heads=4, window_size=4, shift_size=2)
    x = torch.randn(2, 8, 8, 64)

    output = block(x)

    assert output.shape == x.shape
    assert torch.isfinite(output).all()
