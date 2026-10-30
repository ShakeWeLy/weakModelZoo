"""推理后处理：置信度过滤 + 类别偏移的 NMS（单标签）。"""

from __future__ import annotations

import torch
import torchvision

from src.models.detection.YOLO.box import xywh2xyxy


def non_max_suppression(
    prediction: torch.Tensor,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    max_det: int = 300,
    max_nms: int = 30000,
    max_wh: float = 7680.0,
) -> list[torch.Tensor]:
    """Args:
        prediction: (bs, N, 5 + nc)，即 YOLOv5 eval 模式下的 pred，坐标为 (cx, cy, w, h)。
    Returns:
        长度为 bs 的列表，每个元素形状 (n, 6)，列为 (x1, y1, x2, y2, conf, cls)。
    """
    out: list[torch.Tensor] = []
    for x in prediction:
        x = x[x[:, 4] > conf_thres]
        if x.shape[0] == 0:
            out.append(prediction.new_zeros((0, 6)))
            continue

        x[:, 5:] = x[:, 5:] * x[:, 4:5]          # conf = obj * cls
        boxes = xywh2xyxy(x[:, :4])
        conf, cls = x[:, 5:].max(dim=1, keepdim=True)
        dets = torch.cat((boxes, conf, cls.float()), dim=1)[conf.view(-1) > conf_thres]
        if dets.shape[0] == 0:
            out.append(prediction.new_zeros((0, 6)))
            continue
        if dets.shape[0] > max_nms:
            dets = dets[dets[:, 4].argsort(descending=True)[:max_nms]]

        # 不同类别之间互不抑制：给每个类别的框加上不同的偏移量
        offset = dets[:, 5:6] * max_wh
        keep = torchvision.ops.nms(dets[:, :4] + offset, dets[:, 4], iou_thres)[:max_det]
        out.append(dets[keep])
    return out
