#!/bin/bash
# Luban 平台容器内实际执行的训练入口（由 submit_train.py 提交）。
# $1 = 平台容器内的 common.conf 路径（scriptParam 传入）。
# 职责：起环境 -> 改写路径前缀生成平台版 conf -> 跑 train_sft.sh。
set -e
set -o pipefail

# ==== 环境初始化（与 infer_Version2.sh 同款：NFS/本地双挂载名自适应）========
echo ">>> 正在初始化环境..."
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
# 如需 wandb 监控：容器内没有 ~/.netrc，在平台侧安全注入 WANDB_API_KEY；
# 不注入则训练自动降级为跳过监控，不报错。
echo ">>> 当前 Python 路径: $(which python)"
python --version
nvidia-smi || true
# ============================================================================

CONF_FILE="${1:?用法: train_platform.sh <common.conf路径>}"
PROJ_DIR="$(cd "$(dirname "${CONF_FILE}")" && pwd)"
cd "${PROJ_DIR}"

# conf 里可能写着另一个挂载名的绝对路径（如 item_map_path=/home/luban/rank-ssl/...），
# 统一改写成本容器可见的前缀，生成平台版 conf（放项目目录内，相对路径根不变）
RUN_CONF="${PROJ_DIR}/common.platform.conf"
sed -e "s#/home/luban/rank-ssl#${DATA_PREFIX}#g" \
    -e "s#/nfs/dataset-ofs-rank-ssl#${DATA_PREFIX}#g" \
    "${CONF_FILE}" > "${RUN_CONF}"
echo ">>> conf: ${CONF_FILE} -> ${RUN_CONF}（路径前缀已归一为 ${DATA_PREFIX}）"

mkdir -p "${PROJ_DIR}/logs"
bash train/train_sft.sh "${RUN_CONF}" 2>&1 | tee "logs/train_platform_$(date +%m%d_%H%M).log"
