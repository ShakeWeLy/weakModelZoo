#!/usr/bin/env bash
# 训练队列（2026-11-17）：Ghost 本征占比消融。每个条目对应 configs/models/<name>.toml，按顺序训练。
#
#   上一轮 015–017 基于残缺数据（train=14 / val=3）训练，结果无效，已删除。
#
#   nodfc_cbam_none        seed 42
#   nodfc_cbam_none_seed43 seed 43
#   nodfc_cbam_none_seed44 seed 44
#
#   每个 run 训练结束后自动在 test 上评估（--analyze，口径同其它消融）。
#
# 用法（在仓库根目录）：
#   bash experiments/detection/project2_2611/run_ablation_sgc.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

MODELS=(
  sgc_yolov5s_c3_sgdown_gbot_dw5_nodfc_cbam_none
  sgc_yolov5s_c3_sgdown_gbot_dw5_nodfc_cbam_none_seed43
  sgc_yolov5s_c3_sgdown_gbot_dw5_nodfc_cbam_none_seed44
)

for m in "${MODELS[@]}"; do
  echo "========================================"
  echo "[$(date '+%F %T')] start: $m"
  echo "========================================"
  python experiments/detection/project2_2611/train.py -m "$m"
  echo "[$(date '+%F %T')] done:  $m"
  echo
done

echo "全部 nodfc + nocbam 训练与测试完成。"
