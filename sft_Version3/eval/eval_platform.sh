#!/bin/bash
# Luban 平台容器内实际执行的 eval 入口（由 submit_eval.py 提交）。
# $1 = 平台容器内的 common.conf 路径（scriptParam 传入）。
# 职责：起环境 -> 改写路径前缀生成平台版 conf -> 直接跑 run_eval.py
#（不经 run_eval.sh 的 launch_mode 分发，避免平台任务反过来又提交自己造成递归）。

set -e
set -o pipefail

# ==== 环境初始化（与 train_platform.sh 同款：NFS/本地双挂载名自适应）========
echo ">>> 正在初始化环境..."
source /etc/profile || true
if [ -d "/nfs/dataset-ofs-rank-ssl" ]; then
    DATA_PREFIX="/nfs/dataset-ofs-rank-ssl"
    echo ">>> 检测到训练平台环境 NFS，使用路径: ${DATA_PREFIX}"
else
    DATA_PREFIX="/home/luban/rank-ssl"
    echo ">>> 检测到本地 SSH 环境，使用路径: ${DATA_PREFIX}"
fi
ENV_PATH="${DATA_PREFIX}/chenpinyuan/miniconda_base/envs/SFT_A6000"
export PATH="${ENV_PATH}/bin:$PATH"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
echo ">>> 当前 Python 路径: $(which python)"
python --version
if [ -x "/usr/local/hadoop-current/bin/hadoop" ]; then
    echo ">>> hadoop 客户端可用: /usr/local/hadoop-current/bin/hadoop"
else
    echo ">>> [WARN] 容器内没有 hadoop 客户端！取 test 数据会失败——"
    echo ">>>        检查 [platform] image_uuid 是否为带 hadoop 的训练镜像"
fi
nvidia-smi || true
# ============================================================================

CONF_FILE="${1:?用法: eval_platform.sh <common.conf路径>}"
PROJ_DIR="$(cd "$(dirname "${CONF_FILE}")" && pwd)"
cd "${PROJ_DIR}"

# conf 里可能写着另一个挂载名的绝对路径，统一改写成本容器可见的前缀
# （同 train_platform.sh；qwen3_path 是相对路径不受影响，不用改）
RUN_CONF="${PROJ_DIR}/common.platform.conf"
sed -e "s#/home/luban/rank-ssl#${DATA_PREFIX}#g" \
    -e "s#/nfs/dataset-ofs-rank-ssl#${DATA_PREFIX}#g" \
    "${CONF_FILE}" > "${RUN_CONF}"
echo ">>> conf: ${CONF_FILE} -> ${RUN_CONF}（路径前缀已归一为 ${DATA_PREFIX}）"

mkdir -p "${PROJ_DIR}/logs"
python "${PROJ_DIR}/eval/run_eval.py" "${RUN_CONF}" \
    2>&1 | tee "logs/eval_platform_$(date +%m%d_%H%M).log"
