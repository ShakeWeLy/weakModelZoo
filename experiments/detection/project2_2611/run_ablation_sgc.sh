#!/usr/bin/env bash
# M→L 逐步退化消融（失败则停止）
# M  : 标准 Conv+C3 Backbone + VOVGCSPYolo（已有 stdbb_vovyolo）
# 1  : Conv下采样 + SDGhost(e=0.5, 1x1投影=Conv, 稠密Bottleneck)
# 2  : 投影改为 Ghost GBS
# 3  : Bottleneck 改为 Ghost
# L  : 下采样再改为 SGhost（已有 j_vovyolo）
# 用法（在仓库根目录）：
#   bash experiments/detection/project2_2611/run_ablation_sgc.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

MODELS=(
  sgc_yolov5s_m2l_1
  sgc_yolov5s_m2l_2
  sgc_yolov5s_m2l_3
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
