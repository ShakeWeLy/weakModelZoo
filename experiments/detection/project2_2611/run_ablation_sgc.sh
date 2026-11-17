#!/usr/bin/env bash
# 训练队列（2026-11-17）：Ghost 本征占比消融。每个条目对应 configs/models/<name>.toml，按顺序训练。
#
#   本征占比 r = 本征通道 / 输出通道（原 ratio=2 即 r=0.5）。
#   对比 r = 0.5 / 0.625 / 0.75，每组 3 个 seed（42 / 43 / 44）。
#
#   r=0.5 不在队列中：sgc_yolov5s_c3_sgdown_gbot_dw5 的 seed 42/43/44 已完成
#   （结构与权重已验证与 intrinsic_ratio=0.5 完全一致），直接复用。
#
#   ir0625  r=0.625（5/8）  seed 42 / 43 / 44
#   ir075   r=0.75（3/4）   seed 42 / 43 / 44
#   nodfc_cbam_none  去 DFC + 去 CBAM 组合（约 4.69M）  seed 42 / 43 / 44
#
# 注意：同配置重复训练的验证 mAP50-95 相差 0.25–1.8 点，单 seed 的差异不可靠，需看三个 seed 的均值。
# 队列按 seed 成对排列（每个 seed 先跑 0.625 再跑 0.75），中途中断时两组对比仍然对齐。
#
# 用法（在仓库根目录）：
#   bash experiments/detection/project2_2611/run_ablation_sgc.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

MODELS=(
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

echo "全部 Ghost 本征占比消融训练完成。"
