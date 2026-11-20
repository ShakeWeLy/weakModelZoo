#!/usr/bin/env bash
# 训练队列：
#   c3_sgdown_gbot_dw3     三 seed（42/43/44）
#   c3_sgdown_gbot_dw7     seed 43/44（42 已有 `…_030_…_V2`）
#   c3_ghostbot_dw3/dw5/dw7 Conv↓  各 seed 43/44（42 已有 `…_026/027/028_…_V1`）
#
#   本脚本只训练；test 用 analyze.py。
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
  sgc_yolov5s_c3_ghostbot_dw3_seed43
  sgc_yolov5s_c3_ghostbot_dw3_seed44
  sgc_yolov5s_c3_ghostbot_dw5_seed43
  sgc_yolov5s_c3_ghostbot_dw5_seed44
  sgc_yolov5s_c3_ghostbot_dw7_seed43
  sgc_yolov5s_c3_ghostbot_dw7_seed44
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

echo "训练队列完成。"