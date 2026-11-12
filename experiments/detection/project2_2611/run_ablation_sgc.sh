#!/usr/bin/env bash
# c3_ghostbot 上仅改 3×3 段 Ghost 廉价 DW（失败则停止）
# dw3 : mid_dw_kernel_size=3
# dw5 : mid_dw_kernel_size=5
# 用法（在仓库根目录）：
#   bash experiments/detection/project2_2611/run_ablation_sgc.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

MODELS=(
  sgc_yolov5s_c3_ghostbot_dw3
  sgc_yolov5s_c3_ghostbot_dw5
)

for m in "${MODELS[@]}"; do
  echo "========================================"
  echo "[$(date '+%F %T')] start: $m"
  echo "========================================"
  python experiments/detection/project2_2611/train.py -m "$m"
  echo "[$(date '+%F %T')] done:  $m"
  echo
done

echo "全部 mid-DW 消融训练完成。"
