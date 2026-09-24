"""Synapse 多器官分割标签定义。"""

from __future__ import annotations

import numpy as np

LABEL_NAMES = {
    0: "background",
    1: "spleen",
    2: "right_kidney",
    3: "left_kidney",
    4: "gallbladder",
    5: "esophagus",
    6: "liver",
    7: "stomach",
    8: "aorta",
    9: "ivc",
    10: "portal_vein_and_splenic_vein",
    11: "pancreas",
    12: "right_adrenal_gland",
    13: "left_adrenal_gland",
}

LABEL_NAMES_CN = {
    0: "背景",
    1: "脾脏",
    2: "右肾",
    3: "左肾",
    4: "胆囊",
    5: "食管",
    6: "肝脏",
    7: "胃",
    8: "主动脉",
    9: "下腔静脉",
    10: "门静脉与脾静脉",
    11: "胰腺",
    12: "右肾上腺",
    13: "左肾上腺",
}

ORGAN_COLORS = {
    0: "#000000",
    1: "#E74C3C",
    2: "#2ECC71",
    3: "#3498DB",
    4: "#F1C40F",
    5: "#9B59B6",
    6: "#E67E22",
    7: "#1ABC9C",
    8: "#EC407A",
    9: "#16A085",
    10: "#D35400",
    11: "#8E44AD",
    12: "#7F8C8D",
    13: "#2C3E50",
}

FALLBACK_COLORS = ["#95A5A6", "#34495E", "#F39C12", "#C0392B", "#27AE60"]

FOREGROUND_CLASS_IDS = tuple(class_id for class_id in LABEL_NAMES if class_id != 0)

# 与论文评估保持一致的主要器官类别（原始 14 类标签空间）
EVAL_CLASS_IDS = (1, 2, 3, 4, 6, 7, 8, 11)

# 8 器官模型空间：0=背景，1..8 对应 EVAL_CLASS_IDS 顺序
EVAL_MODEL_CLASS_NUM = 9
EVAL_MODEL_CLASS_IDS = tuple(range(1, len(EVAL_CLASS_IDS) + 1))
EVAL_MODEL_TO_ORIGINAL = {
    model_id: original_id for model_id, original_id in enumerate(EVAL_CLASS_IDS, start=1)
}
ORIGINAL_TO_EVAL_MODEL = {0: 0, **{original_id: model_id for model_id, original_id in EVAL_MODEL_TO_ORIGINAL.items()}}
_ORIGINAL_TO_EVAL_MODEL_LUT = np.zeros(14, dtype=np.int64)
for original_id, model_id in ORIGINAL_TO_EVAL_MODEL.items():
    _ORIGINAL_TO_EVAL_MODEL_LUT[original_id] = model_id

# 全量前景类别（训练监控 / 分析指标 / 可视化）
ALL_METRIC_CLASS_IDS = FOREGROUND_CLASS_IDS


def metric_class_name(class_id: int) -> str:
    return LABEL_NAMES.get(class_id, f"class_{class_id}")


def metric_class_name_cn(class_id: int) -> str:
    return LABEL_NAMES_CN.get(class_id, metric_class_name(class_id))


def build_metric_class_map() -> dict[int, str]:
    return {class_id: metric_class_name(class_id) for class_id in ALL_METRIC_CLASS_IDS}


def build_eval_model_metric_class_map() -> dict[int, str]:
    return {
        model_id: metric_class_name(EVAL_MODEL_TO_ORIGINAL[model_id])
        for model_id in EVAL_MODEL_CLASS_IDS
    }


def build_organ_metrics(label_map: str | None = None) -> dict[int, str]:
    if label_map == "eval8":
        return build_eval_model_metric_class_map()
    return build_metric_class_map()


def eval_model_class_name(model_id: int) -> str:
    original_id = EVAL_MODEL_TO_ORIGINAL.get(model_id)
    if original_id is None:
        return f"class_{model_id}"
    return metric_class_name(original_id)


def eval_model_class_name_cn(model_id: int) -> str:
    original_id = EVAL_MODEL_TO_ORIGINAL.get(model_id)
    if original_id is None:
        return f"class_{model_id}"
    return metric_class_name_cn(original_id)


def remap_label_to_eval_model(label: np.ndarray) -> np.ndarray:
    """将原始 14 类标签映射为 9 类（背景 + 8 器官），非评估类置为背景。"""
    clipped = np.clip(label, 0, _ORIGINAL_TO_EVAL_MODEL_LUT.size - 1)
    return _ORIGINAL_TO_EVAL_MODEL_LUT[clipped]


def remap_label_from_eval_model(label: np.ndarray) -> np.ndarray:
    """将 9 类模型标签还原为原始器官 ID，便于沿用 ORGAN_COLORS 可视化。"""
    output = np.zeros_like(label, dtype=label.dtype)
    for model_id, original_id in EVAL_MODEL_TO_ORIGINAL.items():
        output[label == model_id] = original_id
    return output


def apply_label_map(label: np.ndarray, label_map: str | None) -> np.ndarray:
    if label_map == "eval8":
        return remap_label_to_eval_model(label)
    return label


def class_color(class_id: int) -> str:
    if class_id in ORGAN_COLORS:
        return ORGAN_COLORS[class_id]
    return FALLBACK_COLORS[class_id % len(FALLBACK_COLORS)]


def make_overlay(image: np.ndarray, label: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """将 label 中所有前景类别叠加到灰度图像上。"""
    import matplotlib.pyplot as plt

    rgb = np.stack([image, image, image], axis=-1)
    for class_id in np.unique(label):
        class_id = int(class_id)
        if class_id == 0:
            continue
        mask = label == class_id
        if not np.any(mask):
            continue
        color_rgb = np.array(plt.matplotlib.colors.to_rgb(class_color(class_id)))
        rgb[mask] = (1 - alpha) * rgb[mask] + alpha * color_rgb
    return np.clip(rgb, 0, 1)
