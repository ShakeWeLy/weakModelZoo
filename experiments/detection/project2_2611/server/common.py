"""服务器脚本共用的工具：路径、数据集数量检查、JSON 读取。

只在需要时导入 trainer，避免无关依赖；所有路径都相对仓库根目录，不写死本机绝对路径。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent     # .../experiments/detection/project2_2611/server
PROJECT = HERE.parent                      # .../experiments/detection/project2_2611
REPO = PROJECT.parents[2]                  # 仓库根目录

if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

TRAIN_PY = PROJECT / "train.py"
ANALYZE_PY = PROJECT / "analyze.py"
LOG_DIR = HERE / "logs"
RESULT_DIR = HERE / "results"

# 论文口径的数据划分。与之不符时队列默认拒绝启动，防止误用不完整的数据集
# （例如本机 data/Det_Fly 只有 14 张训练图）。
EXPECTED_COUNTS = {"train": 7000, "val": 1500, "test": 1501}
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")
# train.py 写入 summary.json 的 status：completed / early_stopped 表示训练已正常结束
DONE_STATUS = {"completed", "early_stopped"}


def read_json(path: Path) -> dict:
    path = Path(path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def data_dir() -> Path:
    """读取 global.toml 中的 dataset.data_dir（相对仓库根目录解析）。"""
    from trainer import ROOT, build_config

    value = build_config("yolov5s")["dataset"]["data_dir"]
    path = Path(value)
    return path if path.is_absolute() else (ROOT / path).resolve()


def count_files(folder: Path, exts: tuple[str, ...]) -> int:
    if not folder.is_dir():
        return 0
    return sum(1 for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts)


def dataset_report() -> dict:
    """检查数据目录：data.yaml、各 split 的图片与标签数量。返回 counts / mismatch 列表。"""
    root = data_dir()
    report: dict = {"data_dir": str(root), "exists": root.is_dir(), "counts": {}, "mismatch": []}
    if not root.is_dir():
        report["mismatch"].append(f"数据目录不存在: {root}")
        return report
    if not (root / "data.yaml").exists():
        report["mismatch"].append(f"缺少 data.yaml: {root / 'data.yaml'}")
    for split, expected in EXPECTED_COUNTS.items():
        n_img = count_files(root / "images" / split, IMG_EXTS)
        n_lbl = count_files(root / "labels" / split, (".txt",))
        report["counts"][split] = {"images": n_img, "labels": n_lbl}
        if n_img != expected:
            report["mismatch"].append(f"images/{split}: {n_img} 张，期望 {expected} 张")
        if n_lbl != n_img:
            report["mismatch"].append(f"labels/{split}: {n_lbl} 个标签文件，与图片数 {n_img} 不一致")
    return report
