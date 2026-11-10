"""模型注册表：``configs/models/<x>.toml`` 中的 ``[model].arch`` 选择这里的条目，``[model.params]`` 原样传给构造函数。

``nc`` 由框架从 ``data.yaml`` 注入。新增模型：在 ``MODEL_ARCHS`` 中加一项，并新建对应 toml 即可。
"""

from __future__ import annotations

import sys
from typing import Any, Callable, Mapping

from torch import nn

from .config import ROOT

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.detection.YOLO.yolov5 import YOLOv5  # noqa: E402
from src.models.detection.YOLO.yolov8 import YOLOv8  # noqa: E402
from src.models.detection.YOLO.SGC_YOLOv5.sgc_yolov5 import SGC_YOLOv5  # noqa: E402
from src.models.detection.YOLO.SGC_YOLOv5.sgc_yolov5_ablation import (  # noqa: E402
    SGC_YOLOv5_C3Neck,
    SGC_YOLOv5_NoCBAM,
    SGC_YOLOv5_StdBackbone,
    SGC_YOLOv5_StdEarly,
)

MODEL_ARCHS: dict[str, Callable[..., nn.Module]] = {
    "yolov5": YOLOv5,
    "yolov8": YOLOv8,
    "sgc_yolov5": SGC_YOLOv5,
    "sgc_yolov5_nocbam": SGC_YOLOv5_NoCBAM,
    "sgc_yolov5_c3neck": SGC_YOLOv5_C3Neck,
    "sgc_yolov5_stdearly": SGC_YOLOv5_StdEarly,
    "sgc_yolov5_stdbackbone": SGC_YOLOv5_StdBackbone,
}


def build_model(cfg: Mapping[str, Any], nc: int) -> nn.Module:
    model_cfg = cfg["model"]
    arch = str(model_cfg["arch"]).lower()
    if arch not in MODEL_ARCHS:
        raise ValueError(f"未知 arch: {arch}（可选 {', '.join(MODEL_ARCHS)}）")
    params = dict(model_cfg.get("params", {}))
    return MODEL_ARCHS[arch](nc=nc, **params)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
