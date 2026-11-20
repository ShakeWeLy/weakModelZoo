#!/usr/bin/env bash
# 训练队列（2026-11-20）：
#   seed 45：dw5 基线 / dw7 / 无 DFC+无 CBAM（压方差、替换异常 seed）
#   基线 + GhostNetV2 DFC（dfc_type=ghostv2，仅改门控）
#
# 用法（仓库根目录）：
#   bash experiments/detection/project2_2611/run_ablation_sgc.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

EXTRA_ARGS=()
case "$(uname -s)" in
  MINGW*|MSYS*|CYGWIN*)
    EXTRA_ARGS+=(--set "dataset.num_workers=0")
    ;;
esac

MODELS=(
  sgc_yolov5s_c3_sgdown_gbot_dw5_seed45
  sgc_yolov5s_c3_sgdown_gbot_dw7_seed45
  sgc_yolov5s_c3_sgdown_gbot_dw5_nodfc_cbam_none_seed45
  sgc_yolov5s_c3_sgdown_gbot_dw5_dfc_ghostv2
)

run_train() {
  if ((${#EXTRA_ARGS[@]} > 0)); then
    python experiments/detection/project2_2611/train.py -m "$1" "${EXTRA_ARGS[@]}"
  else
    python experiments/detection/project2_2611/train.py -m "$1"
  fi
}

for m in "${MODELS[@]}"; do
  echo "========================================"
  echo "[$(date '+%F %T')] start: $m"
  echo "========================================"
  run_train "$m"
  echo "[$(date '+%F %T')] done:  $m"
  echo
done

echo "seed45 + DFC GhostV2 训练队列完成。"