#!/usr/bin/env bash
# 服务器环境安装：conda 环境（默认 py312，Python 3.12.12）+ torch 2.6.0（PyTorch 官方 CUDA 轮子）+ 固定版本依赖。
#
# 用法（仓库根目录）：
#   bash experiments/detection/project2_2611/server/setup_env.sh
#
# 可选环境变量：
#   ENV_NAME=py312              conda 环境名
#   PY_VER=3.12.12              Python 版本（仓库约定：默认 3.12.12）
#   TORCH_CUDA=cu124            PyTorch 轮子的 CUDA 版本：cu118 / cu124 / cu126，需不高于驱动支持的 CUDA 版本
#   DET_FLY_DIR=/path/Det_Fly   若设置，则把 data/Det_Fly 软链到该目录（数据集不在仓库内）
set -euo pipefail

ENV_NAME="${ENV_NAME:-py312}"
PY_VER="${PY_VER:-3.12.12}"
TORCH_CUDA="${TORCH_CUDA:-cu124}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
REQ="$REPO/experiments/detection/project2_2611/server/requirements-server.txt"

if ! command -v conda >/dev/null 2>&1; then
  echo "未找到 conda，请先安装 Miniconda / Anaconda，并确保 conda 在 PATH 中。" >&2
  exit 1
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

envs="$(conda env list | awk 'NR>2 {print $1}')"
if ! grep -qx "$ENV_NAME" <<< "$envs"; then
  echo "创建 conda 环境 $ENV_NAME（python=$PY_VER）"
  conda create -y -n "$ENV_NAME" "python=$PY_VER"
fi
conda activate "$ENV_NAME"

python -m pip install --upgrade pip
python -m pip install "torch==2.6.0" "torchvision==0.21.0" --index-url "https://download.pytorch.org/whl/$TORCH_CUDA"
python -m pip install -r "$REQ"

if [[ -n "${DET_FLY_DIR:-}" ]]; then
  mkdir -p "$REPO/data"
  if [[ -e "$REPO/data/Det_Fly" || -L "$REPO/data/Det_Fly" ]]; then
    echo "已存在 data/Det_Fly，跳过软链。"
  else
    ln -s "$DET_FLY_DIR" "$REPO/data/Det_Fly"
    echo "已链接 data/Det_Fly -> $DET_FLY_DIR"
  fi
fi

python -c "import torch; print('torch', torch.__version__, '| cuda', torch.version.cuda, '| available', torch.cuda.is_available())"
echo "环境就绪。之后请执行: conda activate $ENV_NAME"
