#!/bin/bash
# 训练统一入口：按 common.conf [train] launch_mode 分发。
#   local    -> 当前机器 GPU 直接跑 train_sft.sh
#   platform -> submit_train.py 提交 Luban 平台并轮询到结束
# 用法: bash run_train.sh [common.conf]
#   （平台任务本身执行的是 train_platform.sh -> train_sft.sh，不会再读
#     launch_mode，不存在递归提交）
set -e
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONF_FILE="${1:-${SCRIPT_DIR}/../common.conf}"

MODE=$(python3 - "${CONF_FILE}" <<'PY'
import configparser, sys
cp = configparser.ConfigParser()
cp.read(sys.argv[1], encoding="utf-8")
print(cp.get("train", "launch_mode", fallback="local").strip() or "local")
PY
)
echo ">>> conf_file=${CONF_FILE}"
echo ">>> launch_mode=${MODE}"

case "${MODE}" in
  local)
    exec bash "${SCRIPT_DIR}/train_sft.sh" "${CONF_FILE}"
    ;;
  platform)
    exec python3 "${SCRIPT_DIR}/submit_train.py" "${CONF_FILE}"
    ;;
  *)
    echo "[ERROR] 未知 launch_mode: ${MODE}（支持 local / platform）" >&2
    exit 1
    ;;
esac
