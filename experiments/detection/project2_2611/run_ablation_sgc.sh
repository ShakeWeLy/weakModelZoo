#!/usr/bin/env bash
# 从 c3topo_1 逐步加回 Ghost（每次只改一项）
# A  : 仅 downsample=sghost
# B  : 仅 bottleneck=ghost
# 用法（在仓库根目录）：
#   bash experiments/detection/project2_2611/run_ablation_sgc.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

MODELS=(
  sgc_yolov5s_c3_sgdown
  sgc_yolov5s_c3_ghostbot
)

for m in "${MODELS[@]}"; do
  echo "========================================"
  echo "[$(date '+%F %T')] start: $m"
  echo "========================================"
  python experiments/detection/project2_2611/train.py -m "$m"
  echo "[$(date '+%F %T')] done:  $m"
  echo
done

echo "全部 Ghost-back 消融训练完成。"
