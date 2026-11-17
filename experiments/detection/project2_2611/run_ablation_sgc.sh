#!/usr/bin/env bash
# 训练队列（2026-11-17）。每个条目对应 configs/models/<name>.toml，按顺序训练。
#
#   A. 噪声基线（40 ep）：本模型与 YOLOv5s 各补 seed 43 / 44
#      （seed 42 已有：本模型 V1 test mAP50 0.788；YOLOv5s V3 0.805）
#   B. 结构消融（40 ep，seed 42，与本模型 V1 对照）：
#      cbam_none      去掉输出前全部 CBAM（P3/P4/P5）
#      cbam_cam_p3    P3 去掉 SAM，只保留 CAM；P4/P5 保持完整 CBAM
#      nodfc          Ghost 瓶颈去掉 DFC 门控
#      resid_identity Ghost 瓶颈残差支路由 3×3 DW 改为恒等捷径
#
# 注意：同配置重复训练的验证 mAP50-95 相差 0.25–1.8 点，单 seed 的差异不可靠。
# B 组中出现明显差异的配置，再补 seed 43 / 44（需要时另加 TOML）。
#
# 用法（在仓库根目录）：
#   bash experiments/detection/project2_2611/run_ablation_sgc.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

MODELS=(
  # ---- A. 噪声基线 ----
  sgc_yolov5s_c3_sgdown_gbot_dw5_seed43
  sgc_yolov5s_c3_sgdown_gbot_dw5_seed44
  yolov5s_seed43
  yolov5s_seed44
  # ---- B. 结构消融（seed 42）----
  sgc_yolov5s_c3_sgdown_gbot_dw5_cbam_none
  sgc_yolov5s_c3_sgdown_gbot_dw5_cbam_cam_p3
  sgc_yolov5s_c3_sgdown_gbot_dw5_nodfc
  sgc_yolov5s_c3_sgdown_gbot_dw5_resid_identity
)

for m in "${MODELS[@]}"; do
  echo "========================================"
  echo "[$(date '+%F %T')] start: $m"
  echo "========================================"
  python experiments/detection/project2_2611/train.py -m "$m"
  echo "[$(date '+%F %T')] done:  $m"
  echo
done

echo "全部消融训练完成。"
