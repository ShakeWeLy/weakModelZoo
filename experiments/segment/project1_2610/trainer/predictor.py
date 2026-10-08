"""推理 + 评估：逐切片 / 逐器官指标、预测掩码、可视化，并汇总到 run 目录。

输出（与 project1_2609 analysis 一致）::

    predictions/<split>/
    ├── predictions/*.npy          预测掩码 0..8 (int16)
    ├── visualizations/*.png
    ├── slice_metrics.csv          每切片一行（含各器官指标与切片均值）
    ├── organ_slice_metrics.csv    每 (切片, 器官) 一行
    ├── organ_metrics.csv          按器官聚合
    └── summary.json
    paper_metrics.csv / summary.json["evaluation"] / summary.md
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from .data import build_dataset, build_loader
from .labels import LabelSpace
from .metrics import PERCENT_KEYS, SLICE_METRIC_KEYS, slice_organ_metrics, summarize_organs
from .models import build_model
from .recorder import RunRecorder, read_json, write_csv, write_json
from .report import ORGAN_METRICS_FIELDS, build_paper_metrics_rows


def load_trained_model(cfg: Mapping[str, Any], checkpoint_path: Path, device: torch.device):
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"未找到 checkpoint: {checkpoint_path}（请先运行 train.py）")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    space = LabelSpace.from_config(cfg)
    model = build_model(cfg, space.num_classes).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


class Predictor:
    def __init__(self, cfg: Mapping[str, Any], recorder: RunRecorder, model: torch.nn.Module,
                 device: torch.device, log=print):
        self.cfg = cfg
        self.recorder = recorder
        self.model = model
        self.device = device
        self.space = LabelSpace.from_config(cfg)
        self.organ_map = self.space.organ_map()
        self.test_cfg = cfg.get("test", {})
        self.log = log

    def _split_flag(self, split: str, key: str) -> bool:
        if not self.test_cfg.get(key, True):
            return False
        splits = self.test_cfg.get(f"{key}_splits")
        return splits is None or split in splits

    @torch.no_grad()
    def run_split(self, split: str, *, target_slices: list[str] | None = None,
                  save_predictions: bool | None = None, save_visualizations: bool | None = None) -> dict[str, Any]:
        save_pred = self._split_flag(split, "save_predictions") if save_predictions is None else save_predictions
        save_vis = self._split_flag(split, "save_visualizations") if save_visualizations is None else save_visualizations
        visualize_num = len(target_slices) if target_slices else int(self.test_cfg.get("visualize_num", 5))

        dataset = build_dataset(self.cfg, split, target_slices=target_slices)
        loader = build_loader(dataset, self.cfg, batch_size=self.test_cfg.get("batch_size"),
                              pin_memory=self.device.type == "cuda")
        split_dir = self.recorder.predictions_dir / split
        pred_dir, vis_dir = split_dir / "predictions", split_dir / "visualizations"
        if save_pred:
            pred_dir.mkdir(parents=True, exist_ok=True)
        if save_vis:
            from .visualize import save_overlay

        slice_rows: list[dict[str, Any]] = []
        organ_rows: list[dict[str, Any]] = []
        per_class: dict[int, dict[str, list]] = {}
        visualized = 0

        for images, labels, names in loader:
            preds = self.model(images.to(self.device)).argmax(dim=1).cpu().numpy()
            images_np = images[:, 0].numpy()
            for pred, image, label, name in zip(preds, images_np, labels.numpy(), names):
                has_label = bool(label.min() >= 0)
                label = label if has_label else None
                row: dict[str, Any] = {"slice_name": name, "split": split, "has_label": has_label,
                                       "total_pixels": int(pred.size)}
                if save_pred:
                    np.save(pred_dir / f"{name}.npy", pred.astype(np.int16))

                organ_metrics: dict[int, dict[str, Any]] = {}
                if has_label:
                    organ_metrics = slice_organ_metrics(pred, label, self.organ_map)
                    slice_mean: dict[str, list[float]] = {k: [] for k in PERCENT_KEYS}
                    for class_id, metrics in organ_metrics.items():
                        organ = self.organ_map[class_id]
                        bucket = per_class.setdefault(class_id, {k: [] for k in SLICE_METRIC_KEYS})
                        for key in SLICE_METRIC_KEYS:
                            bucket[key].append(metrics.get(key))
                            row[f"{key}_{organ}"] = metrics.get(key)
                        for key in PERCENT_KEYS:
                            if metrics.get(key) is not None:
                                slice_mean[key].append(metrics[key])
                        organ_rows.append({
                            "slice_name": name, "split": split, "organ": organ,
                            "organ_cn": self.space.class_name_cn(class_id),
                            **{key: metrics.get(key) for key in SLICE_METRIC_KEYS},
                        })
                    for key, values in slice_mean.items():
                        row[f"mean_{key}"] = float(np.mean(values)) if values else None

                if save_vis and visualized < visualize_num:
                    save_overlay(image, label, pred, organ_metrics if has_label else None,
                                 vis_dir / f"{name}.png", self.space)
                    visualized += 1
                slice_rows.append(row)

        summary = summarize_organs(per_class, self.space)
        summary.update({
            "split": split,
            "num_slices": len(slice_rows),
            "num_labeled_slices": sum(1 for r in slice_rows if r["has_label"]),
            "num_classes": self.space.num_classes,
        })
        split_dir.mkdir(parents=True, exist_ok=True)
        if slice_rows:
            write_csv(split_dir / "slice_metrics.csv", slice_rows)
        if organ_rows:
            write_csv(split_dir / "organ_slice_metrics.csv", organ_rows)
        write_csv(split_dir / "organ_metrics.csv", [
            {
                "organ": organ, "organ_cn": m["name_cn"], "num_slices": m["num_slices"],
                **{f"{k}_mean": m[k] for k in (*PERCENT_KEYS, "hd95")},
                **{f"{k}_percent": m[f"{k}_percent"] for k in PERCENT_KEYS},
                **{f"{k}_mean": m[f"{k}_mean"] for k in ("gt_area", "pred_area", "pred_gt_ratio",
                                                       "gt_percent", "pred_percent")},
            }
            for organ, m in summary["per_organ"].items()
        ], ORGAN_METRICS_FIELDS)
        write_json(split_dir / "summary.json", summary)
        return summary

    def finalize(self, summaries: list[dict[str, Any]], checkpoint_path: Path) -> Path | None:
        """合并本次与已有 split 的结果，写 run 级 summary.json / paper_metrics.csv / summary.md。"""
        existing = read_json(self.recorder.summary_path).get("evaluation", {})
        splits = {item["split"]: item for item in existing.get("splits", [])}
        splits.update({item["split"]: item for item in summaries})
        self.recorder.update_summary(evaluation={
            "checkpoint": str(checkpoint_path),
            "splits": [splits[key] for key in sorted(splits)],
        })
        paper_rows = build_paper_metrics_rows(self.recorder.predictions_dir)
        if paper_rows:
            write_csv(self.recorder.paper_metrics_path, paper_rows, encoding="utf-8-sig")
        self.recorder.write_summary_md()
        return self.recorder.paper_metrics_path if paper_rows else None


def _num(value: float | None, width: int = 6) -> str:
    return f"{value:{width}.2f}" if value is not None else "N/A".rjust(width)


def format_split_summary(summary: Mapping[str, Any]) -> str:
    lines = [
        f"===== {summary['split'].upper()} =====",
        f"切片总数: {summary['num_slices']} | 带标签: {summary['num_labeled_slices']}",
    ]
    if summary.get("mean_dice") is not None:
        lines.append(
            f"平均 Dice {_num(summary['mean_dice_percent'], 0)}%  IoU {_num(summary['mean_iou_percent'], 0)}%  "
            f"P {_num(summary['mean_precision_percent'], 0)}%  R {_num(summary['mean_recall_percent'], 0)}%  "
            f"HD95 {_num(summary['mean_hd95'], 0)}px"
        )
    for organ, m in summary["per_organ"].items():
        if not m["num_slices"]:
            continue
        lines.append(
            f"  {organ:<13} {m['name_cn']}\tDice={_num(m['dice_percent'])}%  "
            f"IoU={_num(m['iou_percent'])}%  HD95={_num(m['hd95'])}  slices={m['num_slices']}"
        )
    return "\n".join(lines)
