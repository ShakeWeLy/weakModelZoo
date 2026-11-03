"""评估入口：加载 run 的 checkpoint，在指定 split 上计算 mAP / P / R，并保存可视化。

用法（在项目根目录执行）：
    python experiments/detection/project2_2611/analyze.py -m yolov5s -n 2026-10-30_002_YOLOv5s_V1
    python experiments/detection/project2_2611/analyze.py -m yolov5s -n <run> --split val --checkpoint last --no-vis
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from trainer import (  # noqa: E402
    RunRecorder,
    available_models,
    build_config,
    build_model,
    evaluate,
    model_output_dir,
    resolve_device,
)
from trainer.config import ROOT  # noqa: E402
from trainer.data import YoloDetDataset, collate_fn, load_dataset_meta  # noqa: E402
from trainer.visualize import plot_metric_curves, plot_results_from_tensorboard, render_detections  # noqa: E402

from src.models.detection.YOLO.box import xywh2xyxy  # noqa: E402
from src.models.detection.YOLO.postprocess import non_max_suppression  # noqa: E402
from trainer.criterion import build_criterion  # noqa: E402


def _data_dir(cfg: dict) -> Path:
    path = Path(cfg["dataset"]["data_dir"])
    return path if path.is_absolute() else (ROOT / path).resolve()


def run_analysis(
    run_dir: Path,
    *,
    device: torch.device,
    splits: list[str] | None = None,
    checkpoint: str | None = None,
    visualize: bool | None = None,
    visualize_num: int | None = None,
) -> dict:
    rec = RunRecorder.open(run_dir)
    cfg = rec.load_config()
    test_cfg = cfg["test"]
    splits = splits or list(test_cfg.get("splits", ["val"]))
    kind = checkpoint or test_cfg.get("checkpoint", "best")
    do_vis = test_cfg.get("save_visualizations", True) if visualize is None else visualize
    vis_num = int(test_cfg.get("visualize_num", 20) if visualize_num is None else visualize_num)
    log = rec.logger("analyze.log")

    ckpt_path = rec.checkpoints_dir / f"{kind}.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"未找到 checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    nc, names, img_size = int(ckpt["nc"]), list(ckpt["names"]), int(ckpt["img_size"])
    model = build_model(cfg, nc).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    criterion = build_criterion(cfg, model, img_size).to(device)

    data_dir = _data_dir(cfg)
    load_dataset_meta(data_dir)  # 校验 data.yaml
    log.info(f"分析 {rec.name} | checkpoint={kind} (epoch {ckpt.get('epoch')}) | splits={splits}")

    results: dict[str, dict] = {}
    for split in splits:
        ds = YoloDetDataset(data_dir / "images" / split, data_dir / "labels" / split, img_size, train=False)
        loader = DataLoader(ds, batch_size=int(cfg["dataset"]["batch_size"]), shuffle=False, collate_fn=collate_fn)
        m = evaluate(
            model, loader, device, img_size=img_size, names=names, criterion=criterion,
            conf_thres=float(test_cfg.get("conf_thres", 0.001)),
            iou_thres=float(test_cfg.get("iou_thres", 0.6)),
            max_det=int(test_cfg.get("max_det", 300)),
            with_curves=True,
        )
        results[split] = m
        out_dir = rec.predictions_dir / split
        out_dir.mkdir(parents=True, exist_ok=True)
        # metrics.json 只写标量，曲线数组另存图
        curves = m.pop("curves", None)
        confusion = m.pop("confusion", None)
        (out_dir / "metrics.json").write_text(
            json.dumps({"split": split, "checkpoint": kind, "epoch": ckpt.get("epoch"), **m},
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        log.info(
            f"[{split}] images={m['num_images']} | mAP50={m['map50']:.4f} mAP50-95={m['map']:.4f} "
            f"P={m['precision']:.4f} R={m['recall']:.4f}"
            + (f" | loss={m.get('val_loss', 0.0):.4f}" if "val_loss" in m else "")
        )
        for name, ap in m["per_class_ap50"].items():
            log.info(f"  AP50 {name}: {ap:.4f}")

        if do_vis:
            vis_dir = out_dir / "visualizations"
            vis_dir.mkdir(exist_ok=True)
            if curves is not None:
                saved = plot_metric_curves(curves, confusion, vis_dir)
                log.info(f"  指标图 {len(saved)} 张 -> {vis_dir}")
            if vis_num > 0:
                vis_conf = float(test_cfg.get("vis_conf_thres", 0.25))
                for i in range(min(vis_num, len(ds))):
                    img_t, lab = ds[i]
                    with torch.no_grad():
                        pred, _ = model(img_t[None].to(device).float() / 255.0)
                    det = non_max_suppression(pred, vis_conf, float(test_cfg.get("iou_thres", 0.6)),
                                              int(test_cfg.get("max_det", 300)))[0].cpu().numpy()
                    gt = xywh2xyxy(lab[:, 1:5]).numpy() * img_size
                    rgb = img_t.permute(1, 2, 0).numpy()
                    bgr = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR)
                    canvas = render_detections(bgr, gt, det, names, vis_conf)
                    cv2.imwrite(str(vis_dir / f"{ds.paths[i].stem}.png"), canvas)
                log.info(f"  检测可视化 {min(vis_num, len(ds))} 张 -> {vis_dir}")

    # 训练曲线总览（类似官方 results.png）
    if do_vis:
        results_png = plot_results_from_tensorboard(rec.tensorboard_dir, rec.run_dir / "results.png")
        if results_png is not None:
            log.info(f"训练结果图 -> {results_png}")

    rec.close_logger()
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-m", "--model", required=True,
                        help=f"configs/models/<model>.toml；可选: {', '.join(available_models())}")
    parser.add_argument("-n", "--name", required=True, help="run 名称（runs 目录下的文件夹名）")
    parser.add_argument("--split", action="append", default=None, help="评估的 split，可多次指定（默认 test.splits）")
    parser.add_argument("--checkpoint", choices=["best", "last"], default=None)
    parser.add_argument("--no-vis", action="store_true", help="不保存可视化")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = build_config(args.model)
    run_dir = model_output_dir(cfg) / args.name
    device = resolve_device(args.device or cfg["test"].get("device"))
    run_analysis(run_dir, device=device, splits=args.split, checkpoint=args.checkpoint,
                 visualize=False if args.no_vis else None)


if __name__ == "__main__":
    main()
