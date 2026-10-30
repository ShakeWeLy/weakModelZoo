"""检测结果可视化：真值（绿）与预测（红）画在 letterbox 后的输入图上。"""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np


def render_detections(
    image_bgr: np.ndarray,
    gt_xyxy: np.ndarray,
    pred: np.ndarray,
    names: Sequence[str],
    vis_conf: float = 0.25,
) -> np.ndarray:
    """Args:
        image_bgr: (H, W, 3) uint8。
        gt_xyxy: (M, 4) 真值框（像素）。
        pred: (N, 6) -> x1, y1, x2, y2, conf, cls（像素）。
        vis_conf: 只画置信度 >= vis_conf 的预测。
    """
    out = image_bgr.copy()
    for x1, y1, x2, y2 in gt_xyxy:
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    for x1, y1, x2, y2, conf, cls in pred:
        if conf < vis_conf:
            continue
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        name = names[int(cls)] if int(cls) < len(names) else str(int(cls))
        cv2.putText(out, f"{name} {conf:.2f}", (int(x1), max(int(y1) - 4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    return out
