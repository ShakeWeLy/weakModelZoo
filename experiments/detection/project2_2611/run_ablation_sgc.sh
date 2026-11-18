#!/usr/bin/env bash
# 训练队列（2026-11-18）：去 DFC + 去 CBAM 组合，重新训练 seed 42 / 43 / 44。
#
#   上一轮 015–017 基于残缺数据（train=14 / val=3）训练，结果无效，已删除。
#   本脚本开始前会检查数据量，不是完整 Det_Fly（train 7000 / val 1500 / test 1501）则直接退出。
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

DATA_DIR="data/Det_Fly"
count() { find "$DATA_DIR/images/$1" -type f 2>/dev/null | wc -l | tr -d ' '; }
n_train=$(count train); n_val=$(count val); n_test=$(count test)
if [ "$n_train" != "7000" ] || [ "$n_val" != "1500" ] || [ "$n_test" != "1501" ]; then
  echo "数据不完整：train=$n_train val=$n_val test=$n_test（期望 7000 / 1500 / 1501），停止训练。" >&2
  exit 1
fi
echo "数据检查通过：train=$n_train val=$n_val test=$n_test"

MODELS=(
  sgc_yolov5s_c3_sgdown_gbot_dw5_nodfc_cbam_none
  sgc_yolov5s_c3_sgdown_gbot_dw5_nodfc_cbam_none_seed43
  sgc_yolov5s_c3_sgdown_gbot_dw5_nodfc_cbam_none_seed44
)

for m in "${MODELS[@]}"; do
  echo "========================================"
  echo "[$(date '+%F %T')] start: $m"
  echo "========================================"
  python experiments/detection/project2_2611/train.py -m "$m" --analyze
  echo "[$(date '+%F %T')] done:  $m"
  echo
done

echo "全部 nodfc + nocbam 训练与测试完成。"
