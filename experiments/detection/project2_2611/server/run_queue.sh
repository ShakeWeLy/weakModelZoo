#!/usr/bin/env bash
# 训练队列的包装脚本：激活 conda 环境后调用 run_queue.py，参数原样传递。
#
# 用法（仓库根目录）：
#   mkdir -p experiments/detection/project2_2611/server/logs
#   nohup bash experiments/detection/project2_2611/server/run_queue.sh --gpus 0 \
#     > experiments/detection/project2_2611/server/logs/queue.out 2>&1 &
#   tail -f experiments/detection/project2_2611/server/logs/queue.out
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${ENV_NAME:-py312}"
mkdir -p "$HERE/logs"

if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  set +u
  conda activate "$ENV_NAME"
  set -u
fi

exec python "$HERE/run_queue.py" "$@"
