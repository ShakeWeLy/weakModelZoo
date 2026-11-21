#!/usr/bin/env bash
# 训练队列：无 DFC + 无 CBAM + 训练期去掉 ChannelShuffle（3 seed，对照 015/016/017）。
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
  sgc_yolov5s_c3_sgdown_gbot_dw5_nodfc_cbam_none_no_shuffle
  sgc_yolov5s_c3_sgdown_gbot_dw5_nodfc_cbam_none_no_shuffle_seed43
  sgc_yolov5s_c3_sgdown_gbot_dw5_nodfc_cbam_none_no_shuffle_seed44
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

echo "no ChannelShuffle（3 seed）训练队列完成。"