import torch
import torch.nn as nn
from torch.nn import functional as F


def window_partition_3D(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """[B, H, W, D, C] -> [B * num_windows, window_size * window_size * window_size, C]"""
    batch_size, height, width, depth, channels = x.shape
    x = x.view(
        batch_size,
        height // window_size,
        window_size,
        width // window_size,
        window_size,
        depth // window_size,
        window_size,
        channels,
    )
    windows = x.permute(0, 1, 3, 5, 2, 4, 6).contiguous()
    return windows.view(-1, window_size * window_size * window_size, channels)
