"""预测分析入口：加载 run 的 checkpoint，在各 split 上推理并输出指标 / 预测 / 可视化。

dataset / model 配置取自 run 目录下训练时保存的 config.yaml（保证与训练一致）；
[test] 段取当前 configs（global + model），因此修改可视化数量等无需重新训练。

用法（在项目根目录执行）：
    python experiments/segment/project1_2610/analyze.py -m swin_unet                     # experiments.name 或最新 run
    python experiments/segment/project1_2610/analyze.py -m swin_unet -n 2026-10-08_001_Swin-UNet_V1
    python experiments/segment/project1_2610/analyze.py -m swin_unet --split val --split test
    python experiments/segment/project1_2610/analyze.py --run-dir experiments/segment/project1_2610/Swin-UNet/runs/xxx
    python experiments/segment/project1_2610/analyze.py -m unet --checkpoint last --no-vis
    python experiments/segment/project1_2610/analyze.py -m unet --target-slices case0005_slice060 case0005_slice070
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent))

from trainer import (  # noqa: E402
    ROOT,
    Predictor,
    RunRecorder,
    available_models,
    build_config,
    format_split_summary,
    load_trained_model,
    model_output_dir,
    resolve_device,
)
from trainer.config import apply_overrides, deep_merge  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-m", "--model", default=None, help=f"可选: {', '.join(available_models())}")
    parser.add_argument("-n", "--name", default=None, help="run 名称；缺省取 experiments.name，不存在则取最新 run")
    parser.add_argument("--run-dir", type=Path, default=None, help="直接指定 run 目录（无需 -m / -n）")
    parser.add_argument("--split", action="append", choices=["train", "val", "test"], default=None,
                        help="可重复；缺省取 test.splits")
    parser.add_argument("--checkpoint", default=None, choices=["best", "last"], help="缺省取 test.checkpoint")
    parser.add_argument("--device", default=None)
    parser.add_argument("--target-slices", nargs="+", default=None, help="只分析指定切片（全部可视化）")
    parser.add_argument("--no-vis", action="store_true", help="不保存可视化")
    parser.add_argument("--no-save-pred", action="store_true", help="不保存预测 .npy")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                        help="覆盖分析配置，如 --set test.visualize_num=20")
    return parser.parse_args()


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
        return runs_root / name
    if args.name:
        raise FileNotFoundError(f"未找到 run: {runs_root / args.name}")
    candidates = sorted((p for p in runs_root.glob("*") if (p / "checkpoints").is_dir()),
                        key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"{runs_root} 下没有可用 run")
    print(f"[analyze] 使用最新 run: {candidates[-1].name}")
    return candidates[-1]


def analysis_config(recorder: RunRecorder, overrides: Iterable[str] = ()) -> dict[str, Any]:
    cfg = recorder.load_config()
    meta = cfg.get("_meta", {})
    model_config = meta.get("model_config")
    if model_config and Path(model_config).exists():
        current = build_config(model_config, quick=bool(meta.get("quick", False)))
        cfg["test"] = deep_merge(cfg.get("test", {}), current.get("test", {}))
    return apply_overrides(cfg, overrides)


def run_analysis(run_dir: Path, *, splits: list[str] | None = None, checkpoint: str | None = None,
                 device=None, target_slices: list[str] | None = None, save_visualizations: bool | None = None,
                 save_predictions: bool | None = None, overrides: Iterable[str] = ()) -> None:
    recorder = RunRecorder.open(run_dir)
    cfg = analysis_config(recorder, overrides)
    test_cfg = cfg.get("test", {})
    if device is None:
        device = resolve_device(test_cfg.get("device") or cfg.get("train", {}).get("device"))
    checkpoint_path = recorder.checkpoint_path(checkpoint or test_cfg.get("checkpoint", "best"))
    model, ckpt = load_trained_model(cfg, checkpoint_path, device)
    target_slices = target_slices or list(test_cfg.get("target_slices") or []) or None

    log = recorder.logger(filename="analyze.log").info
    log(f"Experiment: {recorder.name}")
    log(f"Model: {cfg['model']['display_name']} | Device: {device}")
    log(f"Checkpoint: {checkpoint_path} (epoch {ckpt.get('epoch')}, val_dice={ckpt.get('val_dice', 0):.4f})")
    predictor = Predictor(cfg, recorder, model, device, log=log)
    summaries = []
    try:
        for split in splits or list(test_cfg.get("splits", ["val", "test"])):
            try:
                summary = predictor.run_split(split, target_slices=target_slices,
                                              save_predictions=save_predictions,
                                              save_visualizations=save_visualizations)
            except FileNotFoundError as exc:
                if not target_slices:
                    raise
                log(f"[skip] {split}: {exc}")
                continue
            summaries.append(summary)
            log(format_split_summary(summary))
        paper_path = predictor.finalize(summaries, checkpoint_path)
        if paper_path:
            log(f"论文指标表: {paper_path}")
        log(f"摘要: {recorder.run_dir / 'summary.md'}")
    finally:
        recorder.close_logger()


def main() -> None:
    args = parse_args()
    run_analysis(
        find_run_dir(args),
        splits=args.split,
        checkpoint=args.checkpoint,
        device=resolve_device(args.device) if args.device else None,
        target_slices=args.target_slices,
        save_visualizations=False if args.no_vis else None,
        save_predictions=False if args.no_save_pred else None,
        overrides=args.overrides,
    )


if __name__ == "__main__":
    main()
