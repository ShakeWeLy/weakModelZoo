import pytest
import torch

from src.module.transformer.vit.vit import PatchEmbedding, ViT


def test_patch_embedding_output_shape_with_cls_token():
    layer = PatchEmbedding(in_channels=3, image_size=32, patch_size=8, embed_dim=64)
    x = torch.randn(2, 3, 32, 32)

    output = layer(x)

    assert output.shape == (2, 17, 64)
    assert torch.isfinite(output).all()


def test_patch_embedding_output_shape_without_cls_token():
    layer = PatchEmbedding(
        in_channels=1,
        image_size=32,
        patch_size=8,
        embed_dim=64,
        use_cls_token=False,
    )
    x = torch.randn(2, 1, 32, 32)

    output = layer(x)

    assert output.shape == (2, 16, 64)


def test_patch_embedding_rejects_invalid_image_size():
    with pytest.raises(ValueError, match="整除"):
        PatchEmbedding(in_channels=3, image_size=30, patch_size=8, embed_dim=64)


def test_vit_forward_shape():
    model = ViT(
        in_channels=3,
        image_size=32,
        patch_size=8,
        embed_dim=64,
        depth=2,
        num_heads=4,
        ffn_hidden_channels=128,
        num_classes=10,
    )
    x = torch.randn(2, 3, 32, 32)

    output = model(x)

    assert output.shape == (2, 10)
    assert torch.isfinite(output).all()
