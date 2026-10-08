"""训练入口：全局配置 + 模型配置合并后交给通用 Trainer。

用法（在项目根目录执行）：
    python experiments/segment/project1_2610/train.py -m swin_unet
    python experiments/segment/project1_2610/train.py -m unet --epochs 100
    python experiments/segment/project1_2610/train.py -m swin_unet --quick            # 冒烟测试
    python experiments/segment/project1_2610/train.py -m swin_unet -n 2026-10-08_001_Swin-UNet_V1
    python experiments/segment/project1_2610/train.py -m swin_unet --set hyperparameters.learning_rate=0.005
    python experiments/segment/project1_2610/train.py -m swin_unet --analyze          # 训练后直接评估
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from trainer import (  # noqa: E402
    RunRecorder,
    Trainer,
    available_models,
    build_config,
    model_output_dir,
    resolve_device,
    unique_run_name,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-m", "--model", required=True,
                        help=f"configs/models/<model>.toml 或 toml 路径；可选: {', '.join(available_models())}")
    parser.add_argument("-n", "--name", default=None,
                        help="run 名称（覆盖 experiments.name）；已存在时默认清空重建")
    parser.add_argument("--no-overwrite", action="store_true", help="与 -n 联用：run 已存在时报错")
    parser.add_argument("--epochs", type=int, default=None, help="覆盖 hyperparameters.num_epochs")
    parser.add_argument("--quick", action="store_true", help="合并配置中的 [quick] 段，用于快速试跑")
    parser.add_argument("--device", default=None, help="覆盖 train.device")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                        help="任意覆盖，如 --set dataset.batch_size=4 --set model.params.embed_dim=48")
    parser.add_argument("--analyze", action="store_true", help="训练结束后用 best checkpoint 运行 analyze")
    return parser.parse_args()


def resolve_run_name(args: argparse.Namespace, cfg: dict, runs_root: Path) -> tuple[str, bool]:
    exp = cfg["experiments"]
    if args.name is not None:
        name = args.name.strip()
        if not name:
            raise ValueError("-n / --name 不能为空")
        if args.no_overwrite and (runs_root / name).exists():
            raise FileExistsError(f"run 已存在（--no-overwrite）: {runs_root / name}")
        return name, True
    base = exp.get("name") or f"{exp['date']}_{cfg['model']['display_name']}_V1"
    name = unique_run_name(runs_root, base)
    if name != base:
        print(f"[experiments] {base} 已存在，自动使用 {name}")
    return name, False


def main() -> None:
    args = parse_args()
    cfg = build_config(args.model, quick=args.quick, epochs=args.epochs, overrides=args.overrides)
    if args.device:
        cfg["train"]["device"] = args.device

    runs_root = model_output_dir(cfg)
    name, overwrite = resolve_run_name(args, cfg, runs_root)
    cfg["experiments"]["name"] = name
    if overwrite and (runs_root / name).exists():
        print(f"[experiments] 覆盖已有 run: {runs_root / name}")

    device = resolve_device(cfg["train"].get("device"))
    recorder = RunRecorder.create(runs_root, name, cfg, overwrite=overwrite)
    try:
        Trainer(cfg, recorder, device).fit()
    finally:
        recorder.close_logger()

    if args.analyze:
        from analyze import run_analysis

        run_analysis(recorder.run_dir, device=device)


if __name__ == "__main__":
    main()
