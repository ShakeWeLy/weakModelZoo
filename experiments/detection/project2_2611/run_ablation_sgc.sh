#!/usr/bin/env bash
# 依次训练 configs/models 下全部已有模型（失败则停止）
# 用法（在仓库根目录）：
#   bash experiments/detection/project2_2611/run_ablation_sgc.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

MODELS=(
  yolov5s
  yolov5m
  yolov8s
  sgc_yolov5s
  sgc_yolov5m
  sgc_yolov5s_nocbam
  sgc_yolov5s_c3neck
  sgc_yolov5s_stdearly
)

for m in "${MODELS[@]}"; do
  echo "========================================"
  echo "[$(date '+%F %T')] start: $m"
  echo "========================================"
  python experiments/detection/project2_2611/train.py -m "$m"
  echo "[$(date '+%F %T')] done:  $m"
  echo
done

echo "全部模型训练完成。"
