#!/bin/bash
# 独立 test 窗口评测：约束 beam search 推理 + HR/Recall/NDCG 报表（clk/pay 分开）。
# 前置：outputs/ckpt/best.pt（train_sft）与 outputs/vocab.json（tokenizer_sid）已生成，
#       common.conf [data] test_start/test_end/test_sample_rate 与 [eval] 已配置。
# 用法: bash run_eval.sh [common.conf]
#   不传参时默认用项目根目录的 common.conf
set -e
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONF_FILE="${1:-${SCRIPT_DIR}/../common.conf}"
echo ">>> conf_file=${CONF_FILE}"
echo ">>> python=$(which python)"

cd "${SCRIPT_DIR}"
python run_eval.py "${CONF_FILE}"
