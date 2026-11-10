#!/usr/bin/env bash
# 依次训练 VOVGCSPYolo 相关消融（失败则停止）
# K: G + VOVGCSPYolo
# L: J + VOVGCSPYolo
# M: 全标准 Backbone + VOVGCSPYolo Neck
# 用法（在仓库根目录）：
#   bash experiments/detection/project2_2611/run_ablation_sgc.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

MODELS=(
  sgc_yolov5s_g_vovyolo
  sgc_yolov5s_j_vovyolo
  sgc_yolov5s_stdbb_vovyolo
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
