#!/bin/bash
# 从 HDFS 流式读取 user_action_sample_d_whole_v2 的用户行为样本并打印原始数据。
# 用法: bash step1_get_user_action.sh [common.conf]
#   不传参时默认用项目根目录的 common.conf
set -e
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONF_FILE="${1:-${SCRIPT_DIR}/../common.conf}"
echo ">>> conf_file=${CONF_FILE}"
echo ">>> python=$(which python)"

cd "${SCRIPT_DIR}"
python step1_get_user_action.py "${CONF_FILE}"
