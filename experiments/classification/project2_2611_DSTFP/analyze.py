"""预测 / 评估入口（占位）。

用法（计划）::

    python experiments/classification/project2_2611_DSTFP/analyze.py -m dstfp -n 001_DSTFP_V1
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-m", "--model", required=True, help="模型配置名，如 dstfp")
    parser.add_argument("-n", "--name", required=True, help="run 名称，如 001_DSTFP_V1")
    parser.add_argument("--split", default="test", help="评估 split")
    parser.add_argument("--checkpoint", default="best", choices=["best", "last"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project = Path(__file__).resolve().parent
    # 新布局：runs/<display_name>/<name>/ ；display_name 暂用 model 名大写规则由 configs 决定
    # 正式实现时应读 config.yaml 中的 model.display_name
    raise NotImplementedError(
        f"analyze 尚未接通。预期读取: {project / 'runs'}/<display_name>/{args.name}/checkpoints/{args.checkpoint}.pth"
    )


if __name__ == "__main__":
    main()
