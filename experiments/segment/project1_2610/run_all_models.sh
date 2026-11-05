#!/usr/bin/env bash
# 依次训练 configs/models 下全部已有模型（失败则停止）
# 用法（在仓库根目录）：
#   bash experiments/segment/project1_2610/run_all_models.sh
# 可选：附加参数会透传给每个 train.py，例如
#   bash experiments/segment/project1_2610/run_all_models.sh --analyze
#   bash experiments/segment/project1_2610/run_all_models.sh --quick -n _smoke

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

MODELS=(
  unet
  attn_unet
  psc_unet
  vm_unet
  transunet
  transunet_official
  swin_unet
  swin_unet_official
  slr_unet
  mt_unet_official
)

for m in "${MODELS[@]}"; do
  echo "========================================"
  echo "[$(date '+%F %T')] start: $m"
  echo "========================================"
  python experiments/segment/project1_2610/train.py -m "$m" "$@"
  echo "[$(date '+%F %T')] done:  $m"
  echo
done

echo "全部模型训练完成。"
