import pytest
import torch

from src.module.transformer.swinTransformer.swinTransformer3D import (
    PatchExpanding3D,
    PatchMerging3D,
    ShiftedWindowAttention3D,
    SwinTransformerBlock3D,
    WindowAttentionLayer3D,
    build_shifted_window_attn_mask_3D,
    window_partition_3D,
    window_reverse_3D,
)


def test_window_partition_and_reverse_3d():
    x = torch.randn(2, 8, 8, 8, 64)
    windows = window_partition_3D(x, window_size=4)
    restored = window_reverse_3D(
        windows, window_size=4, height=8, width=8, depth=8, batch_size=2
    )

    assert windows.shape == (16, 64, 64)  # 2 * 2*2*2 windows, 4^3 tokens
    assert restored.shape == x.shape
    assert torch.allclose(restored, x)


def test_window_attention_preserves_shape_3d():
    layer = WindowAttentionLayer3D(embed_dim=64, window_size=4, num_heads=4)
    x = torch.randn(2, 8, 8, 8, 64)

    output = layer(x)

    assert output.shape == x.shape
    assert torch.isfinite(output).all()


def test_window_attention_rejects_invalid_size_3d():
    layer = WindowAttentionLayer3D(embed_dim=64, window_size=4, num_heads=4)
    x = torch.randn(1, 7, 8, 8, 64)

    with pytest.raises(ValueError, match="整除"):
        layer(x)


def test_shifted_window_attn_mask_shape_3d():
    mask = build_shifted_window_attn_mask_3D(
        8, 8, 8, window_size=4, shift_size=2, device="cpu", dtype=torch.float32
    )

    assert mask.shape == (8, 64, 64)  # 2*2*2 windows, 4^3 tokens


def test_shifted_window_attention_preserves_shape_3d():
    layer = ShiftedWindowAttention3D(
        embed_dim=64, window_size=4, num_heads=4, shift_size=2
    )
    x = torch.randn(2, 8, 8, 8, 64)

    output = layer(x)

    assert output.shape == x.shape
    assert torch.isfinite(output).all()


def test_patch_merging_downsamples_3d():
    layer = PatchMerging3D(embed_dim=64)
    x = torch.randn(2, 8, 8, 8, 64)

    output = layer(x)

    assert output.shape == (2, 4, 4, 4, 128)
    assert torch.isfinite(output).all()


def test_patch_expanding_upsamples_3d():
    layer = PatchExpanding3D(embed_dim=128)
    x = torch.randn(2, 4, 4, 4, 128)

    output = layer(x)

    assert output.shape == (2, 8, 8, 8, 64)
    assert torch.isfinite(output).all()


def test_swin_transformer_block_preserves_shape_3d():
    block = SwinTransformerBlock3D(embed_dim=64, num_heads=4, window_size=4, shift_size=0)
    x = torch.randn(2, 8, 8, 8, 64)

    output = block(x)

    assert output.shape == x.shape
    assert torch.isfinite(output).all()


def test_swin_transformer_block_with_shift_3d():
    block = SwinTransformerBlock3D(embed_dim=64, num_heads=4, window_size=4, shift_size=2)
    x = torch.randn(2, 8, 8, 8, 64)

    output = block(x)

    assert output.shape == x.shape
    assert torch.isfinite(output).all()
