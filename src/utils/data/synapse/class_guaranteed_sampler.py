"""保证每个 epoch 至少包含指定类别样本的 Sampler。

用于缓解胆囊、胰腺、肾上腺等稀有类在随机 shuffle 下长期缺席的问题。
类别 ID 使用原始 Synapse 14 类标签空间（映射前 .npy 中的值）。
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
from torch.utils.data import Sampler


def build_sample_pools(
    samples: list[Path],
    label_dir: Path,
    class_groups: list[list[int]],
) -> list[list[int]]:
    """按原始标签构建每个类别组的样本索引池。

    Args:
        samples: 图像路径列表，与 Dataset 索引一一对应。
        label_dir: 标签目录。
        class_groups: 例如 [[4], [11], [12, 13]] 表示胆囊、胰腺、肾上腺（左右合并）。
    """
    pools: list[list[int]] = [[] for _ in class_groups]
    for index, image_path in enumerate(samples):
        label = np.load(label_dir / image_path.name)
        present = set(int(v) for v in np.unique(label))
        for group_idx, class_ids in enumerate(class_groups):
            if present.intersection(class_ids):
                pools[group_idx].append(index)
    return pools


def format_class_groups(class_groups: list[list[int]]) -> list[str]:
    names = {
        4: "gallbladder",
        11: "pancreas",
        12: "right_adrenal",
        13: "left_adrenal",
    }
    formatted: list[str] = []
    for group in class_groups:
        parts = [names.get(class_id, f"class_{class_id}") for class_id in group]
        formatted.append("+".join(parts))
    return formatted


class ClassGuaranteedEpochSampler(Sampler[int]):
    """每个 epoch 先从各类别池抽取固定数量样本，再补齐并打乱。"""

    def __init__(
        self,
        dataset_size: int,
        pools: list[list[int]],
        min_samples_per_group: int,
        seed: int = 42,
        group_names: list[str] | None = None,
    ):
        if dataset_size <= 0:
            raise ValueError(f"dataset_size 必须 > 0，当前为 {dataset_size}")
        if min_samples_per_group <= 0:
            raise ValueError(
                f"min_samples_per_group 必须 > 0，当前为 {min_samples_per_group}"
            )
        for group_idx, pool in enumerate(pools):
            if not pool:
                raise ValueError(f"guaranteed class group #{group_idx} 在训练集中无样本")

        self.dataset_size = dataset_size
        self.pools = pools
        self.min_samples_per_group = min_samples_per_group
        self.seed = seed
        self.epoch = 0
        self.group_names = group_names or [f"group_{idx}" for idx in range(len(pools))]
        self.pool_sizes = [len(pool) for pool in pools]

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        indices: list[int] = []

        for pool in self.pools:
            indices.extend(rng.choices(pool, k=self.min_samples_per_group))

        all_indices = list(range(self.dataset_size))
        rng.shuffle(all_indices)
        indices.extend(all_indices)

        if len(indices) > self.dataset_size:
            indices = indices[: self.dataset_size]
        while len(indices) < self.dataset_size:
            indices.append(rng.choice(all_indices))

        rng.shuffle(indices)
        yield from indices

    def __len__(self) -> int:
        return self.dataset_size
