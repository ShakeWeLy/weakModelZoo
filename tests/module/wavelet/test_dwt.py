import torch

from src.module.wavelet import haar_dwt, haar_idwt


def test_haar_dwt_idwt_roundtrip():
    x = torch.randn(2, 8, 32, 32)
    ll, lh, hl, hh = haar_dwt(x)
    assert ll.shape == (2, 8, 16, 16)
    reconstructed = haar_idwt(ll, lh, hl, hh)
    assert reconstructed.shape == (2, 8, 32, 32)
    assert torch.allclose(reconstructed, x, atol=1e-5)


def test_haar_dwt_supports_odd_size():
    x = torch.randn(1, 4, 33, 31)
    ll, lh, hl, hh = haar_dwt(x)
    assert ll.shape[-2:] == (17, 16)
