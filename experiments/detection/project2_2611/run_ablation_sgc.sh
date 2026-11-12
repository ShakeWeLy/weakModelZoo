#!/usr/bin/env bash
# C3 拓扑消融（失败则停止）
# 相对 m2l residual 拓扑，验证 SDGhost 改成 YOLO C3 双路 concat 后能否回升
# 1  : Conv↓ + SDGhost(csp=yolo, proj=conv, dense-bot)   ≈ 应对齐 M/C3
# 2  : 投影改为 Ghost GBS
# 3  : Bottleneck 改为 Ghost
# 用法（在仓库根目录）：
#   bash experiments/detection/project2_2611/run_ablation_sgc.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

MODELS=(
  sgc_yolov5s_c3topo_1
  sgc_yolov5s_c3topo_2
  sgc_yolov5s_c3topo_3
)

for m in "${MODELS[@]}"; do
  echo "========================================"
  echo "[$(date '+%F %T')] start: $m"
  echo "========================================"
  python experiments/detection/project2_2611/train.py -m "$m"
  echo "[$(date '+%F %T')] done:  $m"
  echo
done

echo "全部 C3 拓扑消融训练完成。"
