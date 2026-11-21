#!/usr/bin/env bash
# 训练队列：无 DFC + 无 CBAM，补 seed 46/47/48（已有 42/43/44/45）。
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
  sgc_yolov5s_c3_sgdown_gbot_dw5_nodfc_cbam_none_seed46
  sgc_yolov5s_c3_sgdown_gbot_dw5_nodfc_cbam_none_seed47
  sgc_yolov5s_c3_sgdown_gbot_dw5_nodfc_cbam_none_seed48
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

echo "无 DFC+无 CBAM seed 46/47/48 训练队列完成。"