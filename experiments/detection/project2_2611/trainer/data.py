"""YOLO 格式检测数据集。

- 训练：按 ``mosaic_prob`` 使用 Mosaic（复用 ``src/utils/data/yolo/mosaic.py``），否则 letterbox；
  再以 ``fliplr`` 概率水平翻转。
- 验证 / 测试：仅 letterbox，保证与推理时的几何变换一致。

返回：图像 uint8 (3, H, W) RGB，标签 float (n, 5) -> (cls, xc, yc, w, h)，均相对于 image_size 的正方形画布。
``collate_fn`` 会把标签拼接为 (nt, 6)，第一列为 batch 内图像索引。
"""

from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch
import yaml
from torch.utils.data import Dataset

from .config import ROOT

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.data.yolo.mosaic import IMG_EXTS, mosaic, read_yolo_labels  # noqa: E402


def load_dataset_meta(data_dir: Path) -> tuple[int, list[str]]:
    """读取 data_dir/data.yaml 中的类别信息，返回 (nc, names)。"""
    with (Path(data_dir) / "data.yaml").open(encoding="utf-8") as f:
        meta = yaml.safe_load(f)
    names = meta["names"]
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    names = [str(n) for n in names]
    nc = int(meta.get("nc", len(names)))
    if nc != len(names):
        raise ValueError(f"data.yaml 中 nc={nc} 与 names 数量 {len(names)} 不一致")
    return nc, names


def letterbox(img: np.ndarray, labels: np.ndarray, size: int, fill: int = 114) -> tuple[np.ndarray, np.ndarray]:
    """等比缩放并居中填充到 size x size；标签同步变换并归一化到画布尺寸。"""
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nw, nh = max(1, round(w * r)), max(1, round(h * r))
    padw, padh = (size - nw) // 2, (size - nh) // 2
    canvas = np.full((size, size, 3), fill, dtype=np.uint8)
    interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
    canvas[padh:padh + nh, padw:padw + nw] = cv2.resize(img, (nw, nh), interpolation=interp)

    labels = labels.astype(np.float32, copy=True)
    if len(labels):
        labels[:, 1] = (labels[:, 1] * nw + padw) / size
        labels[:, 2] = (labels[:, 2] * nh + padh) / size
        labels[:, 3] = labels[:, 3] * nw / size
        labels[:, 4] = labels[:, 4] * nh / size
    return canvas, labels


class YoloDetDataset(Dataset):
    def __init__(
        self,
        img_dir: Path,
        label_dir: Path,
        image_size: int,
        *,
        train: bool,
        mosaic_prob: float = 0.0,
        fliplr: float = 0.0,
    ) -> None:
        self.img_dir = Path(img_dir)
        self.label_dir = Path(label_dir)
        if not self.img_dir.is_dir():
            raise FileNotFoundError(f"图片目录不存在: {self.img_dir}")
        self.paths = sorted(p for p in self.img_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
        if not self.paths:
            raise FileNotFoundError(f"{self.img_dir} 中没有图片")
        self.image_size = image_size
        self.train = train
        self.mosaic_prob = mosaic_prob if train else 0.0
        self.fliplr = fliplr if train else 0.0

    def __len__(self) -> int:
        return len(self.paths)

    def _load(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        path = self.paths[index]
        img = cv2.imread(str(path))
        if img is None:
            raise IOError(f"无法读取图片: {path}")
        return img, read_yolo_labels(self.label_dir / f"{path.stem}.txt")

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        size = self.image_size
        if self.mosaic_prob > 0 and random.random() < self.mosaic_prob:
            picks = [index] + [random.randrange(len(self.paths)) for _ in range(3)]
            loaded = [self._load(i) for i in picks]
            img, labels = mosaic([x[0] for x in loaded], [x[1] for x in loaded], img_size=size)
        else:
            img, raw_labels = self._load(index)
            img, labels = letterbox(img, raw_labels, size)

        if self.fliplr > 0 and random.random() < self.fliplr:
            img = img[:, ::-1].copy()
            if len(labels):
                labels[:, 1] = 1.0 - labels[:, 1]

        rgb = np.ascontiguousarray(img[..., ::-1])
        return torch.from_numpy(rgb).permute(2, 0, 1), torch.from_numpy(labels.astype(np.float32))


def collate_fn(batch: Sequence[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
    imgs, labels = zip(*batch)
    targets = [
        torch.cat((torch.full((len(lab), 1), float(i)), lab), dim=1) for i, lab in enumerate(labels)
    ]
    return torch.stack(imgs, 0), torch.cat(targets, 0)
