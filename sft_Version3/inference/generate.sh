#!/bin/bash
# 批量推理统一入口：按 common.conf [inference] launch_mode 分发。
#   local    -> 当前机器直接跑 generate.py
#   platform -> submit_infer.py 提交 Luban 平台并轮询到结束
# 前置：ckpt（train_sft）与 outputs/vocab.json（tokenizer_sid）已生成，
#       common.conf [inference] 已配置。
# 用法: bash generate.sh [common.conf]
#   不传参时默认用项目根目录的 common.conf
#   （平台任务本身执行的是 infer_platform.sh -> generate.py，不会再读
#     launch_mode，不存在递归提交）
set -e
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONF_FILE="${1:-${SCRIPT_DIR}/../common.conf}"

MODE=$(python3 - "${CONF_FILE}" <<'PY'
import configparser, sys
cp = configparser.ConfigParser()
cp.read(sys.argv[1], encoding="utf-8")
print(cp.get("inference", "launch_mode", fallback="local").strip() or "local")
PY
)
echo ">>> conf_file=${CONF_FILE}"
echo ">>> launch_mode=${MODE}"

case "${MODE}" in
  local)
    echo ">>> python=$(which python)"
    cd "${SCRIPT_DIR}"
    exec python generate.py "${CONF_FILE}"
    ;;
  platform)
    exec python3 "${SCRIPT_DIR}/submit_infer.py" "${CONF_FILE}"
    ;;
  *)
    echo "[ERROR] 未知 launch_mode: ${MODE}（支持 local / platform）" >&2
    exit 1
    ;;
esac
