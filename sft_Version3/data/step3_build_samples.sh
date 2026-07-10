#!/bin/bash
# session 粒度留一法切分 + 构造 (input, label) 样本（first cut，仅打印核对）。
# 用法: bash step3_build_samples.sh [common.conf]
#   不传参时默认用项目根目录的 common.conf
set -e
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONF_FILE="${1:-${SCRIPT_DIR}/../common.conf}"
echo ">>> conf_file=${CONF_FILE}"
echo ">>> python=$(which python)"

cd "${SCRIPT_DIR}"
python step3_build_samples.py "${CONF_FILE}"
