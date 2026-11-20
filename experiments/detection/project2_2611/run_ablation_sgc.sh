#!/usr/bin/env bash
# 训练队列：c3_sgdown / c3topo_1 去 CBAM（cbam_mode=none），各 3 seed（42/43/44）。
# 原带 CBAM 单 seed 仍为 `…_024_…` / `…_021_…`，与本队列 run 分开。
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
  sgc_yolov5s_c3_sgdown_cbam_none
  sgc_yolov5s_c3_sgdown_cbam_none_seed43
  sgc_yolov5s_c3_sgdown_cbam_none_seed44
  sgc_yolov5s_c3topo_1_cbam_none
  sgc_yolov5s_c3topo_1_cbam_none_seed43
  sgc_yolov5s_c3topo_1_cbam_none_seed44
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