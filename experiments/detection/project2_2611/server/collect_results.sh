#!/usr/bin/env bash
# 打包训练结果，便于拷回本机。默认不含权重（*.pth）；加 --with-ckpt 连同权重一起打包。
#
# 用法（仓库根目录）：
#   bash experiments/detection/project2_2611/server/collect_results.sh 2026-11-17
#   bash experiments/detection/project2_2611/server/collect_results.sh 2026-11-17 --with-ckpt
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
PROJECT="$REPO/experiments/detection/project2_2611"
RESULT_DIR="$PROJECT/server/results"

PREFIX="${1:-}"
if [[ -z "$PREFIX" ]]; then
  echo "用法: bash collect_results.sh <run 名称前缀> [--with-ckpt]" >&2
  exit 2
fi
WITH_CKPT=0
if [[ "${2:-}" == "--with-ckpt" ]]; then
  WITH_CKPT=1
fi

# 运行目录布局：<PROJECT>/<display_name>/runs/<run>
mapfile -t RUN_DIRS < <(find "$PROJECT" -mindepth 3 -maxdepth 3 -type d -path "*/runs/${PREFIX}*" | sort)
if [[ ${#RUN_DIRS[@]} -eq 0 ]]; then
  echo "没有匹配前缀 '${PREFIX}' 的 run。" >&2
  exit 1
fi

REL=()
for d in "${RUN_DIRS[@]}"; do
  REL+=("${d#"$REPO"/}")
done

mkdir -p "$RESULT_DIR"
OUT="$RESULT_DIR/results_${PREFIX}_$(date +%Y%m%d_%H%M%S).tar.gz"
EXCLUDE=()
if [[ $WITH_CKPT -eq 0 ]]; then
  EXCLUDE=(--exclude='*.pth')
fi

tar -czf "$OUT" ${EXCLUDE[@]+"${EXCLUDE[@]}"} -C "$REPO" "${REL[@]}"
echo "已打包 ${#REL[@]} 个 run -> $OUT"
for r in "${REL[@]}"; do
  echo "  $r"
done
