"""YOLO 检测模型、损失与后处理（YOLOv5 / YOLOv8）。"""

from src.models.detection.YOLO.box import bbox_iou, xywh2xyxy, xyxy2xywh
from src.models.detection.YOLO.loss import YOLOv5Loss
from src.models.detection.YOLO.postprocess import non_max_suppression
from src.models.detection.YOLO.yolov5 import VARIANTS as YOLOV5_VARIANTS
from src.models.detection.YOLO.yolov5 import YOLOv5
from src.models.detection.YOLO.yolov8 import VARIANTS as YOLOV8_VARIANTS
from src.models.detection.YOLO.yolov8 import YOLOv8
from src.models.detection.YOLO.yolov8_loss import YOLOv8Loss

# 兼容旧导入名
VARIANTS = YOLOV5_VARIANTS

__all__ = [
    "VARIANTS",
    "YOLOV5_VARIANTS",
    "YOLOV8_VARIANTS",
    "YOLOv5",
    "YOLOv8",
    "YOLOv5Loss",
    "YOLOv8Loss",
    "bbox_iou",
    "non_max_suppression",
    "xywh2xyxy",
    "xyxy2xywh",
]
