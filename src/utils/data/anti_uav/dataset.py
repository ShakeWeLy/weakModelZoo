"""Anti-UAV 孪生跟踪 Dataset：同序列采帧对 → 裁剪 → 锚框标签。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from src.utils.data.anti_uav.crops import crop_search, crop_template
from src.utils.data.anti_uav.labels import AnchorTarget
from src.utils.data.anti_uav.parse import SequenceMeta, discover_sequences, read_frame

Modality = Literal["ir", "rgb", "auto"]


def _to_tensor_image(img: np.ndarray) -> torch.Tensor:
    """``HWC uint8 RGB -> CHW float32 [0,1]``。"""
    x = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float() / 255.0
    return x


class AntiUAVPairDataset(Dataset):
    """从 Anti-UAV 采样 ``(template, search, labels)``。

    每个 ``__getitem__`` 随机（或按索引循环）取一条序列内的有效帧对，裁出
    127×127 模板与 255×255 搜索图，并生成 RPN 标签。

    Args:
        root: Anti-UAV 根目录。
        modality: ``ir`` / ``rgb`` / ``auto``。
        splits: 如 ``("train",)``；``None`` 自动发现。
        pairs_per_epoch: 一个 epoch 的样本数（按索引取模采样）。
        max_frame_gap: 模板与搜索最大帧间隔。
        search_shift: 搜索中心最大相对偏移（比例），对应 spatial-aware sampling。
    """

    def __init__(
        self,
        root: str | Path,
        *,
        modality: Modality = "ir",
        splits: Sequence[str] | None = ("train",),
        pairs_per_epoch: int = 60000,
        max_frame_gap: int = 100,
        exemplar_size: int = 127,
        search_size: int = 255,
        context_amount: float = 0.5,
        search_shift: float = 0.2,
        score_size: int = 25,
        stride: int = 8,
        pos_thr: float = 0.6,
        neg_thr: float = 0.3,
    ) -> None:
        super().__init__()
        self.sequences = discover_sequences(root, modality=modality, splits=splits)
        if not self.sequences:
            raise RuntimeError(f"在 {root} 未发现可用 Anti-UAV 序列（modality={modality}, splits={splits}）")

        # 仅保留至少 2 个有效帧的序列
        self.sequences = [s for s in self.sequences if len(s.valid_indices) >= 2]
        if not self.sequences:
            raise RuntimeError("没有包含 ≥2 个有效目标帧的序列")

        self.pairs_per_epoch = int(pairs_per_epoch)
        self.max_frame_gap = int(max_frame_gap)
        self.exemplar_size = exemplar_size
        self.search_size = search_size
        self.context_amount = context_amount
        self.search_shift = search_shift
        self.anchor_target = AnchorTarget(
            stride=stride,
            score_size=score_size,
            search_size=search_size,
            pos_thr=pos_thr,
            neg_thr=neg_thr,
        )
        # 预计算每条序列的候选帧对数量上界，用于加权采样
        self._seq_weights = np.array(
            [max(1, len(s.valid_indices)) for s in self.sequences], dtype=np.float64
        )
        self._seq_weights /= self._seq_weights.sum()

    def __len__(self) -> int:
        return self.pairs_per_epoch

    def _sample_pair(self, seq: SequenceMeta) -> tuple[int, int]:
        valid = seq.valid_indices
        for _ in range(50):
            ti = int(np.random.choice(valid))
            candidates = [j for j in valid if abs(j - ti) <= self.max_frame_gap]
            if len(candidates) >= 1:
                sj = int(np.random.choice(candidates))
                return ti, sj
        # 回退：相邻有效帧
        i = int(np.random.randint(0, len(valid) - 1))
        return valid[i], valid[i + 1]

    def __getitem__(self, index: int) -> dict[str, Any]:
        # index 仅用于驱动 DataLoader 长度；实际随机采样
        seq = self.sequences[int(np.random.choice(len(self.sequences), p=self._seq_weights))]
        ti, sj = self._sample_pair(seq)

        img_z = read_frame(seq, ti)
        img_x = read_frame(seq, sj)
        box_z = seq.annos[ti].bbox_xywh
        box_x = seq.annos[sj].bbox_xywh

        template, _ = crop_template(
            img_z, box_z, self.exemplar_size, self.context_amount
        )
        shift = (
            float(np.random.uniform(-self.search_shift, self.search_shift)),
            float(np.random.uniform(-self.search_shift, self.search_shift)),
        )
        search, gt_xywh, _ = crop_search(
            img_x,
            box_x,
            exemplar_size=self.exemplar_size,
            search_size=self.search_size,
            context_amount=self.context_amount,
            center_shift=shift,
        )
        label_cls, label_loc, label_loc_weight = self.anchor_target(gt_xywh)

        return {
            "template": _to_tensor_image(template),
            "search": _to_tensor_image(search),
            "label_cls": label_cls,
            "label_loc": label_loc,
            "label_loc_weight": label_loc_weight,
            "seq": seq.name,
            "frame_z": ti,
            "frame_x": sj,
        }


def collate_anti_uav(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """堆叠张量字段，保留元信息列表。"""
    return {
        "template": torch.stack([b["template"] for b in batch], dim=0),
        "search": torch.stack([b["search"] for b in batch], dim=0),
        "label_cls": torch.stack([b["label_cls"] for b in batch], dim=0),
        "label_loc": torch.stack([b["label_loc"] for b in batch], dim=0),
        "label_loc_weight": torch.stack([b["label_loc_weight"] for b in batch], dim=0),
        "seq": [b["seq"] for b in batch],
        "frame_z": [b["frame_z"] for b in batch],
        "frame_x": [b["frame_x"] for b in batch],
    }
