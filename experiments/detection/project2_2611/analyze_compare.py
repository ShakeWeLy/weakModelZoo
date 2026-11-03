"""对比两个检测 run：按目标尺寸分层 Recall，并导出 YOLO-only 漏检可视化。

用法（在项目根目录执行）：
    python experiments/detection/project2_2611/analyze_compare.py \\
        --a-model yolov5s --a-name 2026-10-30_002_YOLOv5s_V3 \\
        --b-model sgc_yolov5s --b-name 2026-10-30_003_SGC_YOLOv5s_V4 \\
        --split test --conf 0.25 --vis-num 30
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torchvision.ops import box_iou  # noqa: E402

from trainer import RunRecorder, available_models, build_config, build_model, model_output_dir, resolve_device  # noqa: E402
from trainer.config import ROOT  # noqa: E402
from trainer.data import YoloDetDataset, load_dataset_meta  # noqa: E402

from src.models.detection.YOLO.box import xywh2xyxy  # noqa: E402
from src.models.detection.YOLO.postprocess import non_max_suppression  # noqa: E402

SIZE_BINS = (
    ("small", 0.0, 32.0**2),
    ("medium", 32.0**2, 96.0**2),
    ("large", 96.0**2, float("inf")),
)


def _data_dir(cfg: dict) -> Path:
    path = Path(cfg["dataset"]["data_dir"])
    return path if path.is_absolute() else (ROOT / path).resolve()


def _load_run(model_key: str, name: str, device: torch.device, checkpoint: str = "best"):
    cfg = build_config(model_key)
    run_dir = model_output_dir(cfg) / name
    rec = RunRecorder.open(run_dir)
    run_cfg = rec.load_config()
    ckpt_path = rec.checkpoints_dir / f"{checkpoint}.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"未找到 checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    nc, names, img_size = int(ckpt["nc"]), list(ckpt["names"]), int(ckpt["img_size"])
    model = build_model(run_cfg, nc).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return {
        "rec": rec,
        "cfg": run_cfg,
        "model": model,
        "nc": nc,
        "names": names,
        "img_size": img_size,
        "epoch": ckpt.get("epoch"),
        "tag": f"{run_cfg['model'].get('display_name', model_key)}/{name}",
    }


def _size_name(area: float) -> str:
    for name, lo, hi in SIZE_BINS:
        if lo <= area < hi:
            return name
    return "large"


def _match_gt(pred_xyxy: np.ndarray, gt_xyxy: np.ndarray, iou_thres: float = 0.5) -> np.ndarray:
    """返回每个 GT 是否被命中（贪心一对一，按 IoU 从高到低）。"""
    n_gt = len(gt_xyxy)
    hit = np.zeros(n_gt, dtype=bool)
    if n_gt == 0 or len(pred_xyxy) == 0:
        return hit
    iou = box_iou(torch.from_numpy(gt_xyxy).float(), torch.from_numpy(pred_xyxy).float()).numpy()
    pairs = [(float(iou[i, j]), i, j) for i in range(n_gt) for j in range(pred_xyxy.shape[0]) if iou[i, j] >= iou_thres]
    pairs.sort(reverse=True)
    used_gt, used_pred = set(), set()
    for _, gi, pj in pairs:
        if gi in used_gt or pj in used_pred:
            continue
        used_gt.add(gi)
        used_pred.add(pj)
        hit[gi] = True
    return hit


@torch.no_grad()
def _predict(model, img_t: torch.Tensor, device: torch.device, conf: float, iou: float, max_det: int) -> np.ndarray:
    pred, _ = model(img_t[None].to(device).float() / 255.0)
    det = non_max_suppression(pred.float(), conf, iou, max_det)[0]
    return det.cpu().numpy() if det.numel() else np.zeros((0, 6), dtype=np.float32)


def _draw_compare(
    image_bgr: np.ndarray,
    gt_xyxy: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    names: list[str],
    conf: float,
) -> np.ndarray:
    """GT 绿 / A(YOLO) 蓝 / B(SGC) 红。"""
    out = image_bgr.copy()
    for x1, y1, x2, y2 in gt_xyxy:
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(out, "GT", (int(x1), max(int(y1) - 4, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
    for x1, y1, x2, y2, score, cls in pred_a:
        if score < conf:
            continue
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), (255, 128, 0), 2)
        name = names[int(cls)] if int(cls) < len(names) else str(int(cls))
        cv2.putText(out, f"A {name} {score:.2f}", (int(x1), max(int(y1) - 4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 128, 0), 1)
    for x1, y1, x2, y2, score, cls in pred_b:
        if score < conf:
            continue
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        name = names[int(cls)] if int(cls) < len(names) else str(int(cls))
        cv2.putText(out, f"B {name} {score:.2f}", (int(x1), min(int(y2) + 14, out.shape[0] - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
    return out


def run_compare(
    *,
    a_model: str,
    a_name: str,
    b_model: str,
    b_name: str,
    split: str,
    checkpoint: str,
    conf: float,
    iou_nms: float,
    iou_match: float,
    max_det: int,
    vis_num: int,
    device: torch.device,
    out_dir: Path | None,
) -> dict:
    run_a = _load_run(a_model, a_name, device, checkpoint)
    run_b = _load_run(b_model, b_name, device, checkpoint)
    if run_a["img_size"] != run_b["img_size"]:
        raise ValueError(f"两边 img_size 不一致: {run_a['img_size']} vs {run_b['img_size']}")
    img_size = run_a["img_size"]
    names = run_a["names"]

    data_dir = _data_dir(run_a["cfg"])
    load_dataset_meta(data_dir)
    ds = YoloDetDataset(data_dir / "images" / split, data_dir / "labels" / split, img_size, train=False)

    out_dir = Path(out_dir) if out_dir else (run_a["rec"].run_dir / "compare" / f"vs_{run_b['rec'].name}_{split}")
    out_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = out_dir / "yolo_only_visualizations"
    vis_dir.mkdir(exist_ok=True)

    # 统计容器
    size_stats = {k: {"gt": 0, "hit_a": 0, "hit_b": 0, "both": 0, "a_only": 0, "b_only": 0, "miss": 0} for k, _, _ in SIZE_BINS}
    overall = {"gt": 0, "hit_a": 0, "hit_b": 0, "both": 0, "a_only": 0, "b_only": 0, "miss": 0}
    yolo_only_rows: list[dict] = []
    # 缓存含 YOLO-only 的图，供后续可视化，避免二次推理
    cache: dict[str, dict] = {}

    for i in range(len(ds)):
        img_t, lab = ds[i]
        stem = ds.paths[i].stem
        gt_xyxy = (xywh2xyxy(lab[:, 1:5]).numpy() * img_size).astype(np.float32) if len(lab) else np.zeros((0, 4), np.float32)
        areas = (gt_xyxy[:, 2] - gt_xyxy[:, 0]).clip(min=0) * (gt_xyxy[:, 3] - gt_xyxy[:, 1]).clip(min=0)

        pred_a = _predict(run_a["model"], img_t, device, conf, iou_nms, max_det)
        pred_b = _predict(run_b["model"], img_t, device, conf, iou_nms, max_det)
        hit_a = _match_gt(pred_a[:, :4] if len(pred_a) else pred_a.reshape(0, 6)[:, :4], gt_xyxy, iou_match)
        hit_b = _match_gt(pred_b[:, :4] if len(pred_b) else pred_b.reshape(0, 6)[:, :4], gt_xyxy, iou_match)

        has_a_only = False
        for gi in range(len(gt_xyxy)):
            size = _size_name(float(areas[gi]))
            a_ok, b_ok = bool(hit_a[gi]), bool(hit_b[gi])
            for bucket in (overall, size_stats[size]):
                bucket["gt"] += 1
                bucket["hit_a"] += int(a_ok)
                bucket["hit_b"] += int(b_ok)
                if a_ok and b_ok:
                    bucket["both"] += 1
                elif a_ok and not b_ok:
                    bucket["a_only"] += 1
                elif b_ok and not a_ok:
                    bucket["b_only"] += 1
                else:
                    bucket["miss"] += 1

            if a_ok and not b_ok:
                has_a_only = True
                x1, y1, x2, y2 = gt_xyxy[gi].tolist()
                yolo_only_rows.append({
                    "image": stem,
                    "gt_index": gi,
                    "size": size,
                    "area": float(areas[gi]),
                    "sqrt_area": float(areas[gi] ** 0.5),
                    "gt_xyxy": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                    "n_pred_a": int(len(pred_a)),
                    "n_pred_b": int(len(pred_b)),
                    "max_conf_a": float(pred_a[:, 4].max()) if len(pred_a) else 0.0,
                    "max_conf_b": float(pred_b[:, 4].max()) if len(pred_b) else 0.0,
                })

        if has_a_only:
            cache[stem] = {
                "img_t": img_t,
                "gt_xyxy": gt_xyxy,
                "pred_a": pred_a,
                "pred_b": pred_b,
                "min_sqrt_area": float(min(
                    (areas[gi] ** 0.5) for gi in range(len(gt_xyxy)) if hit_a[gi] and not hit_b[gi]
                )),
            }

    # 优先可视化含更小 YOLO-only 目标的图
    yolo_only_rows.sort(key=lambda r: r["sqrt_area"])
    ranked_stems = sorted(cache.keys(), key=lambda s: cache[s]["min_sqrt_area"])[:vis_num]
    for rank, stem in enumerate(ranked_stems):
        item = cache[stem]
        rgb = item["img_t"].permute(1, 2, 0).numpy()
        bgr = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR)
        canvas = _draw_compare(bgr, item["gt_xyxy"], item["pred_a"], item["pred_b"], names, conf)
        bar = np.full((28, canvas.shape[1], 3), 40, dtype=np.uint8)
        cv2.putText(bar, f"{stem} | green=GT blue=A({a_name}) red=B({b_name}) | conf>={conf}",
                    (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        canvas = np.vstack([bar, canvas])
        cv2.imwrite(str(vis_dir / f"{rank:03d}_{stem}.png"), canvas)
    seen_imgs = ranked_stems

    def _pack(bucket: dict) -> dict:
        gt = max(bucket["gt"], 1)
        return {
            **bucket,
            "recall_a": bucket["hit_a"] / gt if bucket["gt"] else 0.0,
            "recall_b": bucket["hit_b"] / gt if bucket["gt"] else 0.0,
            "recall_gap_a_minus_b": (bucket["hit_a"] - bucket["hit_b"]) / gt if bucket["gt"] else 0.0,
        }

    summary = {
        "split": split,
        "conf": conf,
        "iou_nms": iou_nms,
        "iou_match": iou_match,
        "num_images": len(ds),
        "a": {"model": a_model, "name": a_name, "tag": run_a["tag"], "epoch": run_a["epoch"]},
        "b": {"model": b_model, "name": b_name, "tag": run_b["tag"], "epoch": run_b["epoch"]},
        "overall": _pack(overall),
        "by_size": {k: _pack(v) for k, v in size_stats.items()},
        "yolo_only_count": len(yolo_only_rows),
        "yolo_only_images": len({r["image"] for r in yolo_only_rows}),
        "visualized_images": seen_imgs,
        "output_dir": str(out_dir),
    }

    (out_dir / "compare_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out_dir / "yolo_only.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["image", "gt_index", "size", "area", "sqrt_area", "gt_xyxy",
                        "n_pred_a", "n_pred_b", "max_conf_a", "max_conf_b"],
        )
        writer.writeheader()
        for row in yolo_only_rows:
            writer.writerow(row)

    # 简洁 markdown 表
    lines = [
        f"# Compare A={a_name} vs B={b_name} ({split})",
        "",
        f"- conf={conf}, match IoU>={iou_match}, images={len(ds)}",
        f"- A hit-only (YOLO-only GT): **{len(yolo_only_rows)}** boxes / **{len({r['image'] for r in yolo_only_rows})}** images",
        "",
        "| subset | GT | Recall A | Recall B | gap(A-B) | A-only | B-only | both | miss |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key in ["overall", "small", "medium", "large"]:
        b = summary["overall"] if key == "overall" else summary["by_size"][key]
        lines.append(
            f"| {key} | {b['gt']} | {b['recall_a']:.4f} | {b['recall_b']:.4f} | {b['recall_gap_a_minus_b']:.4f} "
            f"| {b['a_only']} | {b['b_only']} | {b['both']} | {b['miss']} |"
        )
    lines += ["", f"Visualizations: `{vis_dir}`", ""]
    (out_dir / "compare_summary.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--a-model", default="yolov5s", help=f"A 模型 toml；可选: {', '.join(available_models())}")
    p.add_argument("--a-name", default="2026-10-30_002_YOLOv5s_V3")
    p.add_argument("--b-model", default="sgc_yolov5s")
    p.add_argument("--b-name", default="2026-10-30_003_SGC_YOLOv5s_V4")
    p.add_argument("--split", default="test")
    p.add_argument("--checkpoint", choices=["best", "last"], default="best")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou-nms", type=float, default=0.6)
    p.add_argument("--iou-match", type=float, default=0.5)
    p.add_argument("--max-det", type=int, default=300)
    p.add_argument("--vis-num", type=int, default=30)
    p.add_argument("--device", default=None)
    p.add_argument("--out-dir", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    summary = run_compare(
        a_model=args.a_model,
        a_name=args.a_name,
        b_model=args.b_model,
        b_name=args.b_name,
        split=args.split,
        checkpoint=args.checkpoint,
        conf=args.conf,
        iou_nms=args.iou_nms,
        iou_match=args.iou_match,
        max_det=args.max_det,
        vis_num=args.vis_num,
        device=device,
        out_dir=Path(args.out_dir) if args.out_dir else None,
    )
    print(Path(summary["output_dir"], "compare_summary.md").read_text(encoding="utf-8"))
    print(f"详细 JSON: {summary['output_dir']}/compare_summary.json")
    print(f"YOLO-only CSV: {summary['output_dir']}/yolo_only.csv")
    print(f"可视化: {summary['output_dir']}/yolo_only_visualizations")


if __name__ == "__main__":
    main()
