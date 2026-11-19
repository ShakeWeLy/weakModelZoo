#!/usr/bin/env bash
# 训练队列：c3_sgdown_gbot_dw5 三项结构消融补齐 seed 43/44（seed 42 已有单 run）。
#
#   无 DFC          → nodfc_seed43 / nodfc_seed44
#   无 CBAM         → cbam_none_seed43 / cbam_none_seed44
#   恒等残差        → resid_identity_seed43 / resid_identity_seed44
#
#   本脚本只训练；test 评估训练结束后单独运行 analyze.py。
#
# 用法（在仓库根目录）：
#   bash experiments/detection/project2_2611/run_ablation_sgc.sh
#
# 服务器必须先 git pull（含 CAM 导入修复，否则 CBAM 构建报 module is not callable）。

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

EXTRA_ARGS=()
case "$(uname -s)" in
  MINGW*|MSYS*|CYGWIN*)
    # Windows：global.toml 默认 num_workers=24 会 spawn 多进程并反复加载 torch，易 WinError 1455
    EXTRA_ARGS+=(--set "dataset.num_workers=0")
    ;;
esac

MODELS=(
  sgc_yolov5s_c3_sgdown_gbot_dw5_nodfc_seed43
  sgc_yolov5s_c3_sgdown_gbot_dw5_nodfc_seed44
  sgc_yolov5s_c3_sgdown_gbot_dw5_cbam_none_seed43
  sgc_yolov5s_c3_sgdown_gbot_dw5_cbam_none_seed44
  sgc_yolov5s_c3_sgdown_gbot_dw5_resid_identity_seed43
  sgc_yolov5s_c3_sgdown_gbot_dw5_resid_identity_seed44
)

for m in "${MODELS[@]}"; do
  echo "========================================"
  echo "[$(date '+%F %T')] start: $m"
  echo "========================================"
  python experiments/detection/project2_2611/train.py -m "$m" "${EXTRA_ARGS[@]}"
  echo "[$(date '+%F %T')] done:  $m"
  echo
done

echo "结构消融 3×2 seed（43/44）训练队列完成。"