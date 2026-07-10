#!/bin/bash
# GAMER 从零训练：读 common.conf [train]，产出 outputs/ckpt/{best,last}.pt。
# 前置：outputs/samples/{train,val}.jsonl（step3）与 outputs/vocab.json（tokenizer_sid）已生成。
# 用法: bash train_sft.sh [common.conf]
#   不传参时默认用项目根目录的 common.conf
set -e
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONF_FILE="${1:-${SCRIPT_DIR}/../common.conf}"
echo ">>> conf_file=${CONF_FILE}"
echo ">>> python=$(which python)"

cd "${SCRIPT_DIR}"
python train_sft.py "${CONF_FILE}"
