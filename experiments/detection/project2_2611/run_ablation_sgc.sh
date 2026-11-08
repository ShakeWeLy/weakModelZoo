#!/usr/bin/env bash
# 依次训练 SGC Ghost 参数消融（失败则停止）
# G: kernel_size=3, dw_kernel_size=1, groups_ratio=4
# H: kernel_size=3, dw_kernel_size=1, groups_ratio=8
# 用法（在仓库根目录）：
#   bash experiments/detection/project2_2611/run_ablation_sgc.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

MODELS=(
  # sgc_yolov5s_k3_g4
  # sgc_yolov5s_k3_g8
  sgc_yolov5s
  yolov5s
  sgc_yolov5m
  yolov5m
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
