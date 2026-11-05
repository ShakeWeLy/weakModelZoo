#!/usr/bin/env bash
# 依次训练 SGC Ghost 参数消融（失败则停止）
# A: dw_kernel_size=3
# B: kernel_size=3 + dw_kernel_size=3
# C: B + groups_ratio=4
# 用法（在仓库根目录）：
#   bash experiments/detection/project2_2611/run_ablation_sgc.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

MODELS=(
  sgc_yolov5s_dw3
  sgc_yolov5s_k3dw3
  sgc_yolov5s_k3dw3_g4
)

for m in "${MODELS[@]}"; do
  echo "========================================"
  echo "[$(date '+%F %T')] start: $m"
  echo "========================================"
  python experiments/detection/project2_2611/train.py -m "$m"
  echo "[$(date '+%F %T')] done:  $m"
  echo
done

echo "全部参数消融训练完成。"
