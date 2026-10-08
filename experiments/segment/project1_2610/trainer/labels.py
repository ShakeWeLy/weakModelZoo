"""weakUnet_Synapse 标签：0=背景，1..8 = spleen / right_kidney / left_kidney / gallbladder / liver / stomach / aorta / pancreas。"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .config import ROOT

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.data.synapse.labels import (  # noqa: E402
    EVAL_MODEL_TO_ORIGINAL,
    LABEL_NAMES,
    LABEL_NAMES_CN,
    remap_label_from_eval_model,
)

NUM_CLASSES = 9


@dataclass(frozen=True)
class LabelSpace:
    metric_class_ids: tuple[int, ...]
    num_classes: int = NUM_CLASSES

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> LabelSpace:
        ids = tuple(int(v) for v in (cfg.get("train", {}).get("metric_class_ids") or range(1, NUM_CLASSES)))
        invalid = [v for v in ids if not 1 <= v < NUM_CLASSES]
        if invalid:
            raise ValueError(f"metric_class_ids 超出 1..{NUM_CLASSES - 1}: {invalid}")
        return cls(ids)

    @staticmethod
    def to_original(array: np.ndarray) -> np.ndarray:
        """转回 Synapse 原始 14 类 ID，用于沿用 ORGAN_COLORS 着色。"""
        return remap_label_from_eval_model(array)

    @staticmethod
    def class_name(class_id: int) -> str:
        return LABEL_NAMES.get(EVAL_MODEL_TO_ORIGINAL.get(class_id, 0 if class_id == 0 else -1), f"class_{class_id}")

    @staticmethod
    def class_name_cn(class_id: int) -> str:
        original = EVAL_MODEL_TO_ORIGINAL.get(class_id, 0 if class_id == 0 else -1)
        return LABEL_NAMES_CN.get(original, f"class_{class_id}")

    def organ_map(self) -> dict[int, str]:
        return {class_id: self.class_name(class_id) for class_id in self.metric_class_ids}
