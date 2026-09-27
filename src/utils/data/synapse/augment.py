"""Synapse 2D 切片训练数据增强（图像与标签同步变换）。"""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch


@dataclass
class AugmentConfig:
    enabled: bool = False
    flip_horizontal: bool = True
    flip_vertical: bool = True
    rotate: bool = True
    intensity_jitter: bool = True
    intensity_scale: tuple[float, float] = (0.9, 1.1)
    intensity_shift: tuple[float, float] = (-0.1, 0.1)
    gaussian_noise_std: float = 0.02


def build_augment_config(dataset_cfg: dict) -> AugmentConfig:
    scale = dataset_cfg.get("intensity_scale", [0.9, 1.1])
    shift = dataset_cfg.get("intensity_shift", [-0.1, 0.1])
    return AugmentConfig(
        enabled=bool(dataset_cfg.get("augment", False)),
        flip_horizontal=bool(dataset_cfg.get("flip_horizontal", True)),
        flip_vertical=bool(dataset_cfg.get("flip_vertical", True)),
        rotate=bool(dataset_cfg.get("rotate", True)),
        intensity_jitter=bool(dataset_cfg.get("intensity_jitter", True)),
        intensity_scale=(float(scale[0]), float(scale[1])),
        intensity_shift=(float(shift[0]), float(shift[1])),
        gaussian_noise_std=float(dataset_cfg.get("gaussian_noise_std", 0.02)),
    )


def apply_augmentation(
    image: torch.Tensor,
    label: torch.Tensor,
    config: AugmentConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """对 (C,H,W) 图像与 (H,W) 标签应用同步增强。"""
    if not config.enabled:
        return image, label

    if config.flip_horizontal and random.random() < 0.5:
        image = torch.flip(image, dims=[-1])
        label = torch.flip(label, dims=[-1])

    if config.flip_vertical and random.random() < 0.5:
        image = torch.flip(image, dims=[-2])
        label = torch.flip(label, dims=[-2])

    if config.rotate:
        rotations = random.randint(0, 3)
        if rotations:
            image = torch.rot90(image, rotations, dims=[-2, -1])
            label = torch.rot90(label, rotations, dims=[-2, -1])

    if config.intensity_jitter:
        scale = random.uniform(*config.intensity_scale)
        shift = random.uniform(*config.intensity_shift)
        image = image * scale + shift
        if config.gaussian_noise_std > 0 and random.random() < 0.5:
            image = image + torch.randn_like(image) * config.gaussian_noise_std
        image = image.clamp(0.0, 1.0)

    return image, label.contiguous()
