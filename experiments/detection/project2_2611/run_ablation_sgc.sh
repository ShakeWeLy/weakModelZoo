#!/usr/bin/env bash
# 依次训练 SGC Ghost 参数消融（失败则停止）
# D: kernel_size=3, dw_kernel_size=1
# E: D + groups_ratio=3
# F: C 基础上 groups_ratio=8（k=3 dw=3 g=8）
# 用法（在仓库根目录）：
#   bash experiments/detection/project2_2611/run_ablation_sgc.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

MODELS=(
  sgc_yolov5s_k3
  sgc_yolov5s_k3_g3
  sgc_yolov5s_k3dw3_g8
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
