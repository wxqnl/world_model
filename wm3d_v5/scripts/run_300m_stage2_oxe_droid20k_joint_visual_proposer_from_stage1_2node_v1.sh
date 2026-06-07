#!/usr/bin/env bash
set -euo pipefail

cd /data/Minko/world_model/wm3d_v3

WORKER_HOST="${WORKER_HOST:-root@172.27.0.7}"
MASTER_ADDR="${MASTER_ADDR:-172.27.0.6}"
MASTER_PORT="${MASTER_PORT:-29546}"
IFACE="${IFACE:-bond0.1411}"
PYTHON="${PYTHON:-/data/Minko/.venvs/wm3d/bin/python}"
CFG="v3_p64_300m_stage2_oxe_droid20k_joint_visual_proposer_from_stage1_2node_v1.yaml"
MANIFEST="oxe_droid20k_stage1_world_v1.jsonl"
ACTION_STATS="/data/Minko/datasets/cache/wm3d_v3/action_stats_oxe_droid20k_stage1_world_v1.npz"
RESUME_CKPT="${RESUME_CKPT:?RESUME_CKPT is required}"
RUN_NAME="train_300m_stage2_oxe_droid20k_joint_visual_proposer_from_stage1_2node_v1"
LOG_DIR="/data/Minko/logs"
if [[ "${RESUME_CKPT}" = /* ]]; then
  WORKER_RESUME_CKPT="${RESUME_CKPT}"
else
  WORKER_RESUME_CKPT="/data/Minko/world_model/wm3d_v3/${RESUME_CKPT}"
fi

mkdir -p "${LOG_DIR}"

rsync -a "configs/${CFG}" "${WORKER_HOST}:/data/Minko/world_model/wm3d_v3/configs/${CFG}"
rsync -a "manifests/${MANIFEST}" "${WORKER_HOST}:/data/Minko/world_model/wm3d_v3/manifests/${MANIFEST}"
rsync -a "${ACTION_STATS}" "${WORKER_HOST}:${ACTION_STATS}"
ssh "${WORKER_HOST}" "mkdir -p $(dirname "${WORKER_RESUME_CKPT}")"
rsync -a "${RESUME_CKPT}" "${WORKER_HOST}:${WORKER_RESUME_CKPT}"

COMMON_ENV=(
  "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7"
  "PYTHONUNBUFFERED=1"
  "WM3D_DDP_BACKEND=nccl"
  "NCCL_DEBUG=INFO"
  "NCCL_IB_DISABLE=0"
  "NCCL_NVLS_ENABLE=0"
  "NCCL_NET_GDR_LEVEL=2"
  "NCCL_SOCKET_IFNAME=${IFACE}"
  "GLOO_SOCKET_IFNAME=${IFACE}"
  "TORCH_NCCL_ASYNC_ERROR_HANDLING=1"
  "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
)

ssh "${WORKER_HOST}" "cd /data/Minko/world_model/wm3d_v3 && mkdir -p ${LOG_DIR} && (setsid env ${COMMON_ENV[*]} ${PYTHON} -m torch.distributed.run --nnodes=2 --nproc_per_node=8 --node_rank=1 --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} -m wm3d_v3.training.train --cfg configs/${CFG} --resume ${WORKER_RESUME_CKPT} --reset_optim --print_every 25 > ${LOG_DIR}/${RUN_NAME}_node1.log 2>&1 < /dev/null & echo \$! > ${LOG_DIR}/${RUN_NAME}_node1.pid)"

nohup env "${COMMON_ENV[@]}" "${PYTHON}" -m torch.distributed.run \
  --nnodes=2 \
  --nproc_per_node=8 \
  --node_rank=0 \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  -m wm3d_v3.training.train \
  --cfg "configs/${CFG}" \
  --resume "${RESUME_CKPT}" \
  --reset_optim \
  --print_every 25 \
  > "${LOG_DIR}/${RUN_NAME}_node0.log" 2>&1 &
echo $! > "${LOG_DIR}/${RUN_NAME}_node0.pid"

echo "node0_pid=$(cat "${LOG_DIR}/${RUN_NAME}_node0.pid")"
echo "node1_pid=$(ssh "${WORKER_HOST}" "cat ${LOG_DIR}/${RUN_NAME}_node1.pid")"
echo "node0_log=${LOG_DIR}/${RUN_NAME}_node0.log"
echo "node1_log=${WORKER_HOST}:${LOG_DIR}/${RUN_NAME}_node1.log"
