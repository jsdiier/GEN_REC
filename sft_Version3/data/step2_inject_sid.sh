#!/bin/bash
# 把用户行为序列里的 item_id 映射成 geo_sid，打印映射样本并统计缺失率分布。
# 用法: bash step2_inject_sid.sh [common.conf]
#   不传参时默认用项目根目录的 common.conf
set -e
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONF_FILE="${1:-${SCRIPT_DIR}/../common.conf}"
echo ">>> conf_file=${CONF_FILE}"
echo ">>> python=$(which python)"

cd "${SCRIPT_DIR}"
python step2_inject_sid.py "${CONF_FILE}"
