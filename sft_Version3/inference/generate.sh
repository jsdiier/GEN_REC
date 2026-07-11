#!/bin/bash
# 批量推理：窗口内用户全历史作前缀，trie 约束 beam search 出 top-K 推荐（无 label 不算指标）。
# 前置：ckpt（train_sft）与 outputs/vocab.json（tokenizer_sid）已生成，
#       common.conf [inference] 已配置。
# 用法: bash generate.sh [common.conf]
#   不传参时默认用项目根目录的 common.conf
set -e
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONF_FILE="${1:-${SCRIPT_DIR}/../common.conf}"
echo ">>> conf_file=${CONF_FILE}"
echo ">>> python=$(which python)"

cd "${SCRIPT_DIR}"
python generate.py "${CONF_FILE}"
