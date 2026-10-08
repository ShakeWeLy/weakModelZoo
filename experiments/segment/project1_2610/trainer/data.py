"""weakUnet_Synapse 2D 数据集与 DataLoader。

- 训练 / 验证：``data_dir/train_npz/*.npz``，按病例从 ``list_dir/train.txt`` 留出验证集
- 测试：``data_dir/test_vol_h5/*.npy.h5`` 按 ``list_dir/test_vol.txt`` 展开为 2D 切片

``__getitem__`` 统一返回 ``(image[C,H,W], label[H,W], name)``，标签 0..8。
"""

from __future__ import annotations

import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np
import torch
from scipy import ndimage
from torch.utils.data import DataLoader, Dataset

from .config import ROOT, resolve_path

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.data.synapse.class_guaranteed_sampler import ClassGuaranteedEpochSampler  # noqa: E402


class RandomGenerator:
    """与 Swin-Unet ``RandomGenerator`` 一致：50% rot90+flip，否则 50% 小角度旋转；再 zoom 到 image_size。"""

    def __init__(self, image_size: int, *, augment: bool):
        self.image_size = image_size
        self.augment = augment

    def __call__(self, image: np.ndarray, label: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        if self.augment:
            if random.random() > 0.5:
                k = np.random.randint(0, 4)
                axis = np.random.randint(0, 2)
                image = np.flip(np.rot90(image, k), axis=axis)
                label = np.flip(np.rot90(label, k), axis=axis)
            elif random.random() > 0.5:
                angle = np.random.randint(-20, 20)
                image = ndimage.rotate(image, angle, order=0, reshape=False)
                label = ndimage.rotate(label, angle, order=0, reshape=False)
        h, w = image.shape
        if (h, w) != (self.image_size, self.image_size):
            scale = (self.image_size / h, self.image_size / w)
            image = ndimage.zoom(image, scale, order=3)
            label = ndimage.zoom(label, scale, order=0)
        image_t = torch.from_numpy(np.ascontiguousarray(image, dtype=np.float32)).unsqueeze(0)
        label_t = torch.from_numpy(np.ascontiguousarray(label, dtype=np.float32)).long()
        return image_t, label_t


class SliceDataset(Dataset):
    """``train_npz`` 中按名称列出的 2D 切片（训练 / 验证）。"""

    def __init__(self, data_dir: Path, names: list[str], image_size: int, *,
                 augment: bool = False, repeat_gray_to_rgb: bool = False):
        self.npz_dir = Path(data_dir) / "train_npz"
        if not self.npz_dir.is_dir():
            raise FileNotFoundError(f"未找到 train_npz: {self.npz_dir}")
        if not names:
            raise FileNotFoundError("切片列表为空")
        self.names = list(names)
        self.transform = RandomGenerator(image_size, augment=augment)
        self.repeat_gray_to_rgb = repeat_gray_to_rgb

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, index: int):
        name = self.names[index]
        data = np.load(self.npz_dir / f"{name}.npz")
        image, label = self.transform(data["image"], data["label"])
        if self.repeat_gray_to_rgb:
            image = image.repeat(3, 1, 1)
        return image, label, name

    def load_label(self, index: int) -> np.ndarray:
        """原始分辨率、未增强的标签，用于构建保底采样池。"""
        return np.load(self.npz_dir / f"{self.names[index]}.npz")["label"]


class VolumeSliceDataset(Dataset):
    """``test_vol_h5`` 体数据展开为 2D 切片（仅评估用）。"""

    def __init__(self, data_dir: Path, list_dir: Path, image_size: int, *,
                 repeat_gray_to_rgb: bool = False, target_slices: set[str] | None = None):
        self.vol_dir = Path(data_dir) / "test_vol_h5"
        self.transform = RandomGenerator(image_size, augment=False)
        self.repeat_gray_to_rgb = repeat_gray_to_rgb
        self.items: list[tuple[str, int]] = []
        for vol in _read_list(Path(list_dir) / "test_vol.txt"):
            path = self.vol_dir / f"{vol}.npy.h5"
            if not path.exists():
                continue
            with h5py.File(path, "r") as f:
                depth = f["image"].shape[0]
            self.items.extend(
                (vol, z) for z in range(depth)
                if target_slices is None or f"{vol}_slice{z:04d}" in target_slices
            )
        if not self.items:
            raise FileNotFoundError(f"test_vol_h5 无可用切片: {self.vol_dir}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        vol, z = self.items[index]
        with h5py.File(self.vol_dir / f"{vol}.npy.h5", "r") as f:
            image = np.asarray(f["image"][z])
            label = np.asarray(f["label"][z])
        image_t, label_t = self.transform(image, label)
        if self.repeat_gray_to_rgb:
            image_t = image_t.repeat(3, 1, 1)
        return image_t, label_t, f"{vol}_slice{z:04d}"


def _read_list(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def split_train_val(list_dir: Path, *, val_cases: int, seed: int) -> tuple[list[str], list[str]]:
    """从 ``train.txt`` 按病例随机留出 ``val_cases`` 个病例作为验证集。"""
    by_case: dict[str, list[str]] = defaultdict(list)
    for name in _read_list(Path(list_dir) / "train.txt"):
        by_case[name.split("_slice", 1)[0]].append(name)
    cases = sorted(by_case)
    n_val = min(max(1, val_cases), max(1, len(cases) - 1))
    val_set = set(random.Random(seed).sample(cases, n_val))
    train_names = [n for c in cases if c not in val_set for n in by_case[c]]
    val_names = [n for c in cases if c in val_set for n in by_case[c]]
    return train_names, val_names


def build_dataset(cfg: Mapping[str, Any], split: str, *, train: bool = False,
                  target_slices: list[str] | None = None) -> Dataset:
    """``train=True`` 时启用数据增强（仅用于训练集）。"""
    dataset_cfg = cfg["dataset"]
    data_dir = resolve_path(dataset_cfg.get("data_dir"))
    list_dir = resolve_path(dataset_cfg.get("list_dir"))
    if data_dir is None or list_dir is None:
        raise ValueError("需要配置 dataset.data_dir 与 dataset.list_dir")
    image_size = int(dataset_cfg["image_size"])
    rgb = bool(cfg["model"].get("repeat_gray_to_rgb", False))
    wanted = {Path(s).stem for s in target_slices} if target_slices else None

    if split == "test":
        return VolumeSliceDataset(data_dir, list_dir, image_size, repeat_gray_to_rgb=rgb, target_slices=wanted)
    if split not in {"train", "val"}:
        raise ValueError(f"未知 split: {split}")
    train_names, val_names = split_train_val(
        list_dir,
        val_cases=int(dataset_cfg.get("val_holdout_cases", 2)),
        seed=int(dataset_cfg.get("split_seed", 42)),
    )
    names = train_names if split == "train" else val_names
    if wanted is not None:
        names = [n for n in names if n in wanted]
    augment = train and bool(dataset_cfg.get("augment", False))
    return SliceDataset(data_dir, names, image_size, augment=augment, repeat_gray_to_rgb=rgb)


def _seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def build_class_sampler(dataset: SliceDataset, train_cfg: Mapping[str, Any], batch_size: int,
                        class_name) -> ClassGuaranteedEpochSampler:
    groups = [[int(v) for v in group] for group in train_cfg.get("guaranteed_class_groups", [])]
    if not groups:
        raise ValueError("guaranteed_sampling=true 需要配置 train.guaranteed_class_groups")
    pools: list[list[int]] = [[] for _ in groups]
    for index in range(len(dataset)):
        present = set(np.unique(dataset.load_label(index)).tolist())
        for pool, group in zip(pools, groups):
            if present.intersection(group):
                pool.append(index)
    return ClassGuaranteedEpochSampler(
        dataset_size=len(dataset),
        pools=pools,
        min_samples_per_group=int(train_cfg.get("min_samples_per_group_per_epoch", batch_size)),
        seed=int(train_cfg.get("seed", 42)),
        group_names=["+".join(class_name(c) for c in group) for group in groups],
    )


def build_loader(dataset: Dataset, cfg: Mapping[str, Any], *, shuffle: bool = False,
                 sampler=None, batch_size: int | None = None, pin_memory: bool = False) -> DataLoader:
    dataset_cfg = cfg["dataset"]
    num_workers = int(dataset_cfg.get("num_workers", 0))
    return DataLoader(
        dataset,
        batch_size=int(batch_size or dataset_cfg["batch_size"]),
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=_seed_worker,
        persistent_workers=num_workers > 0,
    )
