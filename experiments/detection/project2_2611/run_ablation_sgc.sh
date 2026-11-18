#!/usr/bin/env bash
# 训练队列（2026-11-17）：Ghost 本征占比消融。每个条目对应 configs/models/<name>.toml，按顺序训练。
#
#   上一轮 015–017 基于残缺数据（train=14 / val=3）训练，结果无效，已删除。
#
#   seed42_rerun           基线 seed 42 重训（保留旧 2026-11-13_029_..._V1，新 run 名含 seed42_rerun）
#   nodfc_cbam_none        seed 42
#   nodfc_cbam_none_seed43 seed 43
#   nodfc_cbam_none_seed44 seed 44
#
#   本脚本只训练；test 评估需要训练结束后单独运行 analyze.py。
#
# 用法（在仓库根目录）：
#   bash experiments/detection/project2_2611/run_ablation_sgc.sh
#
# 服务器必须先 git pull（含 7ba8b5d+ 的 CAM 导入修复，否则 CBAM 构建报 module is not callable）。

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

DATA_DIR="${DET_FLY_DIR:-$ROOT/data/Det_Fly}"
MIN_TRAIN="${MIN_TRAIN:-5000}"
MIN_VAL="${MIN_VAL:-1000}"

EXTRA_ARGS=()
case "$(uname -s)" in
  MINGW*|MSYS*|CYGWIN*)
    # Windows：global.toml 默认 num_workers=24 会 spawn 多进程并反复加载 torch，易 WinError 1455
    EXTRA_ARGS+=(--set "dataset.num_workers=0")
    ;;
esac

python - "$DATA_DIR" "$MIN_TRAIN" "$MIN_VAL" <<'PY'
import sys
from pathlib import Path

data = Path(sys.argv[1])
min_train, min_val = int(sys.argv[2]), int(sys.argv[3])
if not (data / "data.yaml").is_file():
    sys.exit(f"错误: 未找到 {data / 'data.yaml'}，请挂载完整 Det_Fly（train≈7000 val≈1500）")

exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def count(split: str) -> int:
    im = data / "images" / split
    if not im.is_dir():
        return 0
    return sum(1 for p in im.iterdir() if p.suffix.lower() in exts)

nt, nv = count("train"), count("val")
print(f"[preflight] Det_Fly train={nt} val={nv} ({data})")
if nt < min_train or nv < min_val:
    raise SystemExit(
        f"错误: 数据量不足（需要 train>={min_train} val>={min_val}）。"
        " 本地残缺集勿训；请恢复完整 data/Det_Fly 或 export DET_FLY_DIR=..."
    )
PY

MODELS=(
  sgc_yolov5s_c3_sgdown_gbot_dw5_seed42_rerun
)

for m in "${MODELS[@]}"; do
  echo "========================================"
  echo "[$(date '+%F %T')] start: $m"
  echo "========================================"
  python experiments/detection/project2_2611/train.py -m "$m" "${EXTRA_ARGS[@]}"
  echo "[$(date '+%F %T')] done:  $m"
  echo
done

echo "全部 nodfc + nocbam 训练完成。"