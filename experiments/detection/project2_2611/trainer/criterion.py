"""按 ``model.arch`` 构建损失：yolov5 → YOLOv5Loss；yolov8 / sgc 等同理。"""

from __future__ import annotations

from typing import Any, Mapping

from torch import nn

from src.models.detection.YOLO.loss import YOLOv5Loss
from src.models.detection.YOLO.yolov8_loss import YOLOv8Loss


def build_criterion(cfg: Mapping[str, Any], model: nn.Module, img_size: int) -> nn.Module:
    arch = str(cfg["model"]["arch"]).lower()
    hyp = cfg["loss"]
    if arch in {"yolov5", "sgc_yolov5"}:
        return YOLOv5Loss(model.detect, hyp, img_size)
    if arch == "yolov8":
        return YOLOv8Loss(model.detect, hyp, img_size)
    raise ValueError(f"未知 arch 的损失: {arch}")
