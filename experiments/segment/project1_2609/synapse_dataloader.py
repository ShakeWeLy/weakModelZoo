"""Synapse 2D 训练/推理数据管线。

默认与 weakUnet / TransUNet 官方一致：
  - 数据: ``F:/myproject/weakUnet/data/Synapse``（``train_npz`` + ``test_vol_h5``）
  - 列表: ``F:/myproject/weakUnet/Swin-Unet/lists/lists_Synapse``
  - 在线增强: RandomGenerator（旋转/翻转 + zoom 到 image_size）

可选 ``format = "modelzoo"`` 使用 ``weakModelZoo/data/synapse_processed`` 原管线。
"""

from __future__ import annotations

import random
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from scipy.ndimage import zoom
from torch.utils.data import DataLoader, Dataset

PROJECT_SEGMENT_DIR = Path(__file__).resolve().parent
ROOT = PROJECT_SEGMENT_DIR.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.data.synapse.augment import AugmentConfig, apply_augmentation, build_augment_config
from src.utils.data.synapse.class_guaranteed_sampler import (
    ClassGuaranteedEpochSampler,
    build_sample_pools,
    format_class_groups,
)
from src.utils.data.synapse.labels import apply_label_map

# weakUnet / TransUNet 预处理数据（本项目训练默认）
DEFAULT_TRANSUNET_SYNAPSE_ROOT = Path(r"F:/myproject/weakUnet/data/Synapse")
DEFAULT_TRANSUNET_LIST_DIR = Path(r"F:/myproject/weakUnet/Swin-Unet/lists/lists_Synapse")

# weakModelZoo 自有预处理
DEFAULT_MODELZOO_DATA_DIR = ROOT / "data" / "synapse_processed"

DEFAULT_DATA_DIR = DEFAULT_TRANSUNET_SYNAPSE_ROOT


def resolve_path(value: str | Path | None, default: Path, *, base: Path | None = None) -> Path:
    if value is None or (isinstance(value, str) and not value.strip()):
        return default.resolve()
    path = Path(value)
    if not path.is_absolute():
        path = (base or ROOT) / path
    return path.resolve()


def resolve_data_dir(data_dir: str | Path | None = None, *, root: Path | None = None) -> Path:
    return resolve_path(data_dir, DEFAULT_TRANSUNET_SYNAPSE_ROOT, base=root)


def resolve_list_dir(list_dir: str | Path | None = None) -> Path:
    return resolve_path(list_dir, DEFAULT_TRANSUNET_LIST_DIR)


def get_dataset_format(dataset_cfg: dict, train_cfg: dict | None = None) -> str:
    fmt = dataset_cfg.get("format")
    if fmt:
        return str(fmt).lower()
    if train_cfg and train_cfg.get("format"):
        return str(train_cfg["format"]).lower()
    return "transunet"


def validate_image_size(image_size: int, patch_size: int = 4, window_size: int = 7) -> None:
    if image_size % 32 != 0:
        raise ValueError(f"image_size 需能被 32 整除，当前为 {image_size}")
    if (image_size // patch_size) % window_size != 0:
        raise ValueError(
            f"image_size/{patch_size} 需能被 window_size={window_size} 整除，"
            f"当前 image_size={image_size}"
        )


# --- TransUNet / Swin-Unet 同款 RandomGenerator ---


def _random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def _random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class TransUNetRandomGenerator:
    """与 Swin-Unet ``datasets/dataset_synapse.RandomGenerator`` 一致。"""

    def __init__(self, output_size: list[int], *, augment: bool = True):
        self.output_size = output_size
        self.augment = augment

    def __call__(self, sample: dict) -> dict:
        image, label = sample["image"], sample["label"]
        if self.augment:
            if random.random() > 0.5:
                image, label = _random_rot_flip(image, label)
            elif random.random() > 0.5:
                image, label = _random_rotate(image, label)
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(
                image,
                (self.output_size[0] / x, self.output_size[1] / y),
                order=3,
            )
            label = zoom(
                label,
                (self.output_size[0] / x, self.output_size[1] / y),
                order=0,
            )
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32)).long()
        return {"image": image, "label": label}


def split_transunet_train_val(
    list_dir: Path,
    *,
    val_cases: int = 2,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    """从 ``train.txt`` 按 case 留出验证集（TransUNet 无官方 val 列表）。"""
    train_file = list_dir / "train.txt"
    lines = [ln.strip() for ln in train_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
    by_case: dict[str, list[str]] = defaultdict(list)
    for name in lines:
        case = name.split("_slice", 1)[0]
        by_case[case].append(name)
    cases = sorted(by_case.keys())
    rng = random.Random(seed)
    n_val = min(max(1, val_cases), max(1, len(cases) - 1))
    val_case_set = set(rng.sample(cases, n_val))
    train_names: list[str] = []
    val_names: list[str] = []
    for case in cases:
        if case in val_case_set:
            val_names.extend(by_case[case])
        else:
            train_names.extend(by_case[case])
    return train_names, val_names


class TransUNetSliceDataset(Dataset):
    """TransUNet ``train_npz`` + 切片名列表。"""

    def __init__(
        self,
        synapse_root: Path,
        sample_names: list[str],
        image_size: int,
        num_classes: int,
        *,
        augment: bool = False,
        repeat_gray_to_rgb: bool = False,
    ):
        self.synapse_root = Path(synapse_root)
        self.npz_dir = self.synapse_root / "train_npz"
        self.sample_names = list(sample_names)
        self.image_size = image_size
        self.num_classes = num_classes
        self.repeat_gray_to_rgb = repeat_gray_to_rgb
        self.transform = TransUNetRandomGenerator(
            [image_size, image_size], augment=augment
        )
        if not self.sample_names:
            raise FileNotFoundError("TransUNet 切片列表为空")
        if not self.npz_dir.is_dir():
            raise FileNotFoundError(f"未找到 train_npz: {self.npz_dir}")

    def __len__(self) -> int:
        return len(self.sample_names)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        name = self.sample_names[index]
        data = np.load(self.npz_dir / f"{name}.npz")
        image, label = data["image"], data["label"]
        out = self.transform({"image": image, "label": label})
        image, label = out["image"], out["label"]
        if self.repeat_gray_to_rgb:
            image = image.repeat(3, 1, 1)
        return image, label


def build_sample_pools_transunet(
    sample_names: list[str],
    npz_dir: Path,
    class_groups: list[list[int]],
) -> list[list[int]]:
    pools: list[list[int]] = [[] for _ in class_groups]
    for index, name in enumerate(sample_names):
        label = np.load(npz_dir / f"{name}.npz")["label"]
        present = set(int(v) for v in np.unique(label))
        for group_idx, class_ids in enumerate(class_groups):
            if present.intersection(class_ids):
                pools[group_idx].append(index)
    return pools


# --- weakModelZoo synapse_processed ---


def resize_slice_pair(
    image: torch.Tensor,
    label: torch.Tensor,
    image_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    image = F.interpolate(
        image.unsqueeze(0),
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)
    label = F.interpolate(
        label.unsqueeze(0).unsqueeze(0).float(),
        size=(image_size, image_size),
        mode="nearest",
    ).squeeze(0).squeeze(0).long()
    return image, label


class SynapseSliceDataset(Dataset):
    """modelzoo: synapse_processed 767→224 + eval8 + AugmentConfig。"""

    def __init__(
        self,
        data_dir: Path,
        split: str,
        image_size: int,
        num_classes: int,
        repeat_gray_to_rgb: bool = False,
        label_map: str | None = None,
        augment_config: AugmentConfig | None = None,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.image_dir = self.data_dir / split / "images"
        self.label_dir = self.data_dir / split / "labels"
        self.image_size = image_size
        self.num_classes = num_classes
        self.repeat_gray_to_rgb = repeat_gray_to_rgb
        self.label_map = label_map
        self.augment_config = augment_config
        self.samples = sorted(self.image_dir.glob("*.npy"))
        if not self.samples:
            raise FileNotFoundError(f"{self.image_dir} 下未找到 .npy 切片")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path = self.samples[index]
        image = np.load(image_path).astype("float32")
        label = apply_label_map(np.load(self.label_dir / image_path.name).astype("int64"), self.label_map)
        image = torch.from_numpy(image).unsqueeze(0)
        label = torch.from_numpy(label)
        image, label = resize_slice_pair(image, label, self.image_size)
        if self.augment_config is not None:
            image, label = apply_augmentation(image, label, self.augment_config)
        if self.repeat_gray_to_rgb:
            image = image.repeat(3, 1, 1)
        return image, label


def normalize_slice_filenames(slice_names: list[str]) -> list[str]:
    return [name if name.endswith(".npy") else f"{name}.npy" for name in slice_names]


class TransUNetVolumeSliceDataset(Dataset):
    """从 ``test_vol_h5`` 展开为 2D 切片（评估用）。"""

    def __init__(
        self,
        synapse_root: Path,
        list_dir: Path,
        image_size: int,
        repeat_gray_to_rgb: bool = False,
    ):
        self.synapse_root = Path(synapse_root)
        self.vol_dir = self.synapse_root / "test_vol_h5"
        self.image_size = image_size
        self.repeat_gray_to_rgb = repeat_gray_to_rgb
        self.transform = TransUNetRandomGenerator([image_size, image_size], augment=False)
        vol_names = [
            ln.strip()
            for ln in (list_dir / "test_vol.txt").read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
        self.items: list[tuple[str, int]] = []
        for vol in vol_names:
            path = self.vol_dir / f"{vol}.npy.h5"
            if not path.exists():
                continue
            with h5py.File(path, "r") as f:
                depth = f["image"].shape[0]
            for z in range(depth):
                self.items.append((vol, z))
        if not self.items:
            raise FileNotFoundError(f"test_vol_h5 无有效体数据: {self.vol_dir}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        vol, z = self.items[index]
        with h5py.File(self.vol_dir / f"{vol}.npy.h5", "r") as f:
            image = f["image"][z]
            label = f["label"][z]
        out = self.transform({"image": image, "label": label})
        image, label = out["image"], out["label"]
        if self.repeat_gray_to_rgb:
            image = image.repeat(3, 1, 1)
        return image, label, f"{vol}_slice{z:04d}"


class TransUNetInferenceDataset(Dataset):
    """TransUNet 推理：train/val 用 npz 列表，test 用 test_vol 展开。"""

    def __init__(
        self,
        synapse_root: Path,
        list_dir: Path,
        split: str,
        image_size: int,
        sample_names: list[str] | None = None,
        repeat_gray_to_rgb: bool = False,
        target_slices: list[str] | None = None,
    ):
        self.synapse_root = Path(synapse_root)
        self.list_dir = Path(list_dir)
        self.split = split
        self.image_size = image_size
        self.repeat_gray_to_rgb = repeat_gray_to_rgb
        self.transform = TransUNetRandomGenerator([image_size, image_size], augment=False)
        self.npz_dir = self.synapse_root / "train_npz"

        if split == "test":
            self._mode = "volume"
            self._volume_ds = TransUNetVolumeSliceDataset(
                synapse_root, list_dir, image_size, repeat_gray_to_rgb
            )
            return

        self._mode = "npz"
        if sample_names is not None:
            names = sample_names
        elif split == "train":
            names = [
                ln.strip()
                for ln in (list_dir / "train.txt").read_text(encoding="utf-8").splitlines()
                if ln.strip()
            ]
        else:
            _, val_names = split_transunet_train_val(list_dir)
            names = val_names

        if target_slices:
            wanted = {s.replace(".npz", "") for s in target_slices}
            names = [n for n in names if n in wanted]
        self.sample_names = names
        if not self.sample_names:
            raise FileNotFoundError(f"TransUNet split={split} 无可用切片")

    def __len__(self) -> int:
        if self._mode == "volume":
            return len(self._volume_ds)
        return len(self.sample_names)

    def __getitem__(self, index: int):
        if self._mode == "volume":
            return self._volume_ds[index]
        name = self.sample_names[index]
        data = np.load(self.npz_dir / f"{name}.npz")
        out = self.transform({"image": data["image"], "label": data["label"]})
        image, label = out["image"], out["label"]
        if self.repeat_gray_to_rgb:
            image = image.repeat(3, 1, 1)
        return image, label, name


SynapseInferenceDataset = TransUNetInferenceDataset


class ModelZooInferenceDataset(Dataset):
    def __init__(
        self,
        data_dir: Path,
        split: str,
        image_size: int,
        repeat_gray_to_rgb: bool = False,
        label_map: str | None = None,
        target_slices: list[str] | None = None,
    ):
        self.image_dir = Path(data_dir) / split / "images"
        self.label_dir = Path(data_dir) / split / "labels"
        self.image_size = image_size
        self.repeat_gray_to_rgb = repeat_gray_to_rgb
        self.label_map = label_map
        all_samples = sorted(self.image_dir.glob("*.npy"))
        if target_slices:
            wanted = set(normalize_slice_filenames(target_slices))
            self.samples = [p for p in all_samples if p.name in wanted]
        else:
            self.samples = all_samples
        if not self.samples:
            raise FileNotFoundError(f"{self.image_dir} 下未找到 .npy 切片")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path = self.samples[index]
        image = np.load(image_path).astype("float32")
        image = torch.from_numpy(image).unsqueeze(0)
        image = F.interpolate(
            image.unsqueeze(0),
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        if self.repeat_gray_to_rgb:
            image = image.repeat(3, 1, 1)
        label_path = self.label_dir / image_path.name
        if label_path.exists():
            label = apply_label_map(np.load(label_path).astype("int64"), self.label_map)
            label = torch.from_numpy(label).unsqueeze(0).unsqueeze(0).float()
            label = F.interpolate(
                label,
                size=(self.image_size, self.image_size),
                mode="nearest",
            ).squeeze(0).squeeze(0).long()
        else:
            label = torch.full((self.image_size, self.image_size), -1, dtype=torch.long)
        return image, label, image_path.name


DEFAULT_GUARANTEED_CLASS_GROUPS = [[4], [5], [6]]


def build_train_dataloader(
    train_set: Dataset,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    train_cfg: dict,
) -> tuple[DataLoader, ClassGuaranteedEpochSampler | None]:
    train_sampler: ClassGuaranteedEpochSampler | None = None
    if train_cfg.get("guaranteed_sampling", False):
        class_groups = [
            [int(v) for v in group]
            for group in train_cfg.get("guaranteed_class_groups", DEFAULT_GUARANTEED_CLASS_GROUPS)
        ]
        min_samples_per_group = int(
            train_cfg.get("min_samples_per_group_per_epoch", batch_size)
        )
        if isinstance(train_set, TransUNetSliceDataset):
            pools = build_sample_pools_transunet(
                train_set.sample_names, train_set.npz_dir, class_groups
            )
        elif isinstance(train_set, SynapseSliceDataset):
            pools = build_sample_pools(train_set.samples, train_set.label_dir, class_groups)
        else:
            pools = []
        train_sampler = ClassGuaranteedEpochSampler(
            dataset_size=len(train_set),
            pools=pools,
            min_samples_per_group=min_samples_per_group,
            seed=int(train_cfg.get("sampler_seed", 42)),
            group_names=format_class_groups(class_groups),
        )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, train_sampler


def build_train_val_datasets(
    dataset_cfg: dict,
    model_cfg: dict,
    train_cfg: dict,
) -> tuple[Dataset, Dataset]:
    fmt = get_dataset_format(dataset_cfg, train_cfg)
    image_size = int(dataset_cfg["image_size"])
    num_classes = int(model_cfg["class_nums"])
    repeat_gray_to_rgb = bool(model_cfg.get("repeat_gray_to_rgb", False))

    if fmt == "transunet":
        synapse_root = resolve_data_dir(train_cfg.get("data_dir") or dataset_cfg.get("data_dir"))
        list_dir = resolve_list_dir(dataset_cfg.get("list_dir") or train_cfg.get("list_dir"))
        val_cases = int(train_cfg.get("val_holdout_cases", 2))
        seed = int(train_cfg.get("sampler_seed", 42))
        train_names, val_names = split_transunet_train_val(
            list_dir, val_cases=val_cases, seed=seed
        )
        use_augment = bool(dataset_cfg.get("augment", True))
        train_set = TransUNetSliceDataset(
            synapse_root,
            train_names,
            image_size,
            num_classes,
            augment=use_augment,
            repeat_gray_to_rgb=repeat_gray_to_rgb,
        )
        val_set = TransUNetSliceDataset(
            synapse_root,
            val_names,
            image_size,
            num_classes,
            augment=False,
            repeat_gray_to_rgb=repeat_gray_to_rgb,
        )
        train_set.augment_config = None
        return train_set, val_set

    data_dir = resolve_path(
        train_cfg.get("data_dir") or dataset_cfg.get("data_dir"),
        DEFAULT_MODELZOO_DATA_DIR,
    )
    label_map = model_cfg.get("label_map") or None
    if label_map == "":
        label_map = None
    augment_config = build_augment_config(dataset_cfg) if dataset_cfg.get("augment") else None
    train_set = SynapseSliceDataset(
        data_dir,
        "train",
        image_size,
        num_classes,
        repeat_gray_to_rgb=repeat_gray_to_rgb,
        label_map=label_map,
        augment_config=augment_config,
    )
    val_set = SynapseSliceDataset(
        data_dir,
        "val",
        image_size,
        num_classes,
        repeat_gray_to_rgb=repeat_gray_to_rgb,
        label_map=label_map,
    )
    train_set.augment_config = augment_config
    return train_set, val_set


def build_inference_dataset(
    split: str,
    dataset_cfg: dict,
    model_cfg: dict,
    train_cfg: dict,
    test_cfg: dict,
    *,
    target_slices: list[str] | None = None,
) -> Dataset:
    fmt = get_dataset_format(dataset_cfg, train_cfg)
    image_size = int(dataset_cfg["image_size"])
    repeat_gray_to_rgb = bool(model_cfg.get("repeat_gray_to_rgb", False))
    label_map = model_cfg.get("label_map") or None
    if label_map == "":
        label_map = None

    if fmt == "transunet":
        synapse_root = resolve_data_dir(
            test_cfg.get("data_dir") or train_cfg.get("data_dir") or dataset_cfg.get("data_dir")
        )
        list_dir = resolve_list_dir(
            dataset_cfg.get("list_dir") or train_cfg.get("list_dir") or test_cfg.get("list_dir")
        )
        return TransUNetInferenceDataset(
            synapse_root,
            list_dir,
            split,
            image_size,
            repeat_gray_to_rgb=repeat_gray_to_rgb,
            target_slices=target_slices,
        )

    data_dir = resolve_path(
        test_cfg.get("data_dir") or train_cfg.get("data_dir"),
        DEFAULT_MODELZOO_DATA_DIR,
    )
    return ModelZooInferenceDataset(
        data_dir,
        split,
        image_size,
        repeat_gray_to_rgb=repeat_gray_to_rgb,
        label_map=label_map,
        target_slices=target_slices,
    )
