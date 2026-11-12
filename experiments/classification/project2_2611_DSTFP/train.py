"""训练入口（占位）：配置合并 → 解析 run 目录 → 创建结果骨架。

用法（在仓库根目录执行）::

    python experiments/classification/project2_2611_DSTFP/train.py -m dstfp
    python experiments/classification/project2_2611_DSTFP/train.py -m dstfp --quick
    python experiments/classification/project2_2611_DSTFP/train.py -m dstfp -n 001_DSTFP_V1
    python experiments/classification/project2_2611_DSTFP/train.py -m swin4d_fmri
    python experiments/classification/project2_2611_DSTFP/train.py -m swin3d_fa

结果路径（新约定）::

    experiments/classification/project2_2611_DSTFP/runs/<display_name>/<name>/
    例: runs/DSTFP/001_DSTFP_V1/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from trainer import (  # noqa: E402
    PlaceholderTrainer,
    RunRecorder,
    available_models,
    build_config,
    model_output_dir,
    resolve_device,
    unique_run_name,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "-m",
        "--model",
        required=True,
        help=f"configs/models/<model>.toml；可选: {', '.join(available_models())}",
    )
    parser.add_argument("-n", "--name", default=None, help="run 名称（覆盖默认）；已存在时默认清空重建")
    parser.add_argument("--no-overwrite", action="store_true", help="与 -n 联用：run 已存在时报错")
    parser.add_argument("--epochs", type=int, default=None, help="覆盖 hyperparameters.num_epochs")
    parser.add_argument("--quick", action="store_true", help="合并配置中的 [quick] 段")
    parser.add_argument("--device", default=None, help="覆盖 train.device")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="任意覆盖，如 --set dataset.batch_size=2",
    )
    parser.add_argument(
        "--init-only",
        action="store_true",
        help="只创建 run 目录与 summary.json，不进入训练（默认行为，因训练尚未接通）",
    )
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
    base = exp.get("name") or f"001_{cfg['model']['display_name']}_V1"
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
    print(f"[experiments] run_dir = {recorder.run_dir}")
    print(f"[experiments] description = {cfg['experiments'].get('description')}")

    if args.init_only:
        print("[experiments] --init-only：仅初始化，跳过训练")
        return

    # 训练尚未接通：创建目录后给出明确提示，避免 silent fail
    try:
        PlaceholderTrainer(cfg, recorder, device).fit()
    except NotImplementedError as exc:
        print(f"[experiments] 占位中止: {exc}")
        print(f"[experiments] 已初始化: {recorder.run_dir}")


if __name__ == "__main__":
    main()
