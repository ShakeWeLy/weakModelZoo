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

# 与论文评估/analysis 脚本保持一致的主要器官类别
EVAL_CLASS_IDS = (1, 2, 3, 4, 6, 7, 8, 11)


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
