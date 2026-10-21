"""对已训练 run 做分割 Grad-CAM 可视化。

核心逻辑在 ``src/utils/gradcam.py``；本脚本负责加载 checkpoint / 数据并落盘。

用法（项目根目录）::

    python experiments/segment/project1_2610/gradcam.py \\
        --run-dir experiments/segment/project1_2610/UNet/runs/2026-10-08_001_UNet_V1 \\
        --layer down4.conv --class-id 5 --num 8

    python experiments/segment/project1_2610/gradcam.py -m unet -n 2026-10-08_001_UNet_V1 \\
        --layer up1.conv --class-id 1 --split val --use-pred-mask
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from trainer import (  # noqa: E402
    ROOT,
    RunRecorder,
    available_models,
    build_config,
    load_trained_model,
    model_output_dir,
    resolve_device,
)
from trainer.data import build_dataset, build_loader  # noqa: E402
from trainer.labels import LabelSpace  # noqa: E402

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.gradcam import (  # noqa: E402
    GradCAM,
    SegmentationCategoryTarget,
    overlay_cam_on_image,
    resolve_module,
    save_gradcam_figure,
    tensor_to_rgb_float,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-m", "--model", default=None, help=f"可选: {', '.join(available_models())}")
    p.add_argument("-n", "--name", default=None, help="run 名称")
    p.add_argument("--run-dir", type=Path, default=None, help="直接指定 run 目录")
    p.add_argument("--checkpoint", default="best", choices=["best", "last"])
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--layer", default="down4.conv", help="目标层，如 down4.conv / up1.conv")
    p.add_argument("--class-id", type=int, default=5, help="要解释的类别（Synapse eval: 1..8，5=liver）")
    p.add_argument("--num", type=int, default=8, help="可视化切片数")
    p.add_argument("--device", default=None)
    p.add_argument(
        "--use-pred-mask",
        action="store_true",
        help="仅在预测为该类的区域上反传（更聚焦器官）",
    )
    p.add_argument(
        "--use-gt-mask",
        action="store_true",
        help="仅在 GT 为该类的区域上反传",
    )
    p.add_argument("--target-slices", nargs="+", default=None)
    return p.parse_args()


def find_run_dir(args: argparse.Namespace) -> Path:
    if args.run_dir is not None:
        run_dir = args.run_dir if args.run_dir.is_absolute() else ROOT / args.run_dir
        return run_dir.resolve()
    if args.model is None:
        raise SystemExit("需要 -m/--model 或 --run-dir")
    cfg = build_config(args.model)
    runs_root = model_output_dir(cfg)
    name = args.name or cfg["experiments"].get("name")
    if name and (runs_root / name).exists():
        return (runs_root / name).resolve()
    if args.name:
        raise FileNotFoundError(f"未找到 run: {runs_root / args.name}")
    candidates = sorted(
        (p for p in runs_root.glob("*") if (p / "checkpoints").is_dir()),
        key=lambda p: p.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError(f"{runs_root} 下没有可用 run")
    print(f"[gradcam] 使用最新 run: {candidates[-1].name}")
    return candidates[-1].resolve()


def main() -> None:
    args = parse_args()
    run_dir = find_run_dir(args)
    recorder = RunRecorder.open(run_dir)
    cfg = recorder.load_config()
    device = resolve_device(args.device or cfg.get("test", {}).get("device") or cfg.get("train", {}).get("device"))
    ckpt_path = recorder.checkpoint_path(args.checkpoint)
    model, ckpt = load_trained_model(cfg, ckpt_path, device)
    space = LabelSpace.from_config(cfg)

    try:
        layer = resolve_module(model, args.layer)
    except Exception as exc:
        raise SystemExit(f"无法解析层 {args.layer!r}: {exc}") from exc

    dataset = build_dataset(cfg, args.split, target_slices=args.target_slices)
    loader = build_loader(dataset, cfg, batch_size=1, pin_memory=device.type == "cuda")

    out_dir = run_dir / "gradcam" / f"class{args.class_id}_{args.layer.replace('.', '-')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    class_name = space.class_name(args.class_id)
    print(f"[gradcam] run={run_dir.name} ckpt=epoch{ckpt.get('epoch')} layer={args.layer} "
          f"class={args.class_id}({class_name}) → {out_dir}")

    saved = 0
    with GradCAM(model, [layer]) as cam:
        for images, labels, names in loader:
            if saved >= args.num:
                break
            images = images.to(device)
            labels = labels.to(device)
            name = names[0] if isinstance(names, (list, tuple)) else str(names[0])

            mask = None
            if args.use_pred_mask or args.use_gt_mask:
                with torch.no_grad():
                    pred = model(images).argmax(dim=1)
                if args.use_gt_mask:
                    mask = (labels == args.class_id).float()
                else:
                    mask = (pred == args.class_id).float()
                if mask.sum() < 1:
                    print(f"[skip] {name}: 无类别 {args.class_id} 区域")
                    continue

            grayscale = cam(
                images,
                targets=[SegmentationCategoryTarget(args.class_id, mask=mask)],
            )[0]

            rgb = tensor_to_rgb_float(images[0])
            overlay = overlay_cam_on_image(rgb, grayscale)
            title = f"{name} | {args.layer} | class {args.class_id} ({class_name})"
            path = save_gradcam_figure(
                out_dir / f"{name}.png",
                rgb=rgb,
                cam=grayscale,
                overlay=overlay,
                title=title,
            )
            print(f"  saved {path.name}")
            saved += 1

    print(f"[gradcam] 完成 {saved} 张 → {out_dir}")


if __name__ == "__main__":
    main()
