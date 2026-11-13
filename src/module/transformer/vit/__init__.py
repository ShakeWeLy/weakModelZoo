"""Vision Transformer 变体。"""

from .deit import DeiT
from .pvt import PVT
from .cvt import CvT
from .t2tVit import T2TViT
from .crossVit import CrossViT
from .maxVit import MaxViT

__all__ = [
    "DeiT",
    "PVT",
    "CvT",
    "T2TViT",
    "CrossViT",
    "MaxViT",
]
