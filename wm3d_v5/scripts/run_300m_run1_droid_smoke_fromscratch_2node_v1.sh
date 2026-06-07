#!/usr/bin/env bash
set -euo pipefail

cd /data/Minko/world_model/wm3d_v3

WORKER_HOST="${WORKER_HOST:-root@172.27.0.7}"
MASTER_ADDR="${MASTER_ADDR:-172.27.0.6}"
MASTER_PORT="${MASTER_PORT:-29543}"
IFACE="${IFACE:-bond0.1411}"
PYTHON="${PYTHON:-/data/Minko/.venvs/wm3d/bin/python}"
CFG="v3_p64_300m_run1_droid_smoke_fromscratch_2node_v1.yaml"
MANIFEST="droid_smoke64_cached_rgb_geom_v1.jsonl"
CACHE_ROOT="/data/Minko/datasets/cache/wm3d_v3_droid_smoke"
RUN_NAME="train_300m_run1_droid_smoke_fromscratch_2node_v1"
LOG_DIR="/data/Minko/logs"

mkdir -p "${LOG_DIR}"

rsync -a "configs/${CFG}" "${WORKER_HOST}:/data/Minko/world_model/wm3d_v3/configs/${CFG}"
rsync -a "manifests/${MANIFEST}" "${WORKER_HOST}:/data/Minko/world_model/wm3d_v3/manifests/${MANIFEST}"
rsync -a "${CACHE_ROOT}/" "${WORKER_HOST}:${CACHE_ROOT}/"

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
)

launch_node1() {
  ssh "${WORKER_HOST}" "cd /data/Minko/world_model/wm3d_v3 && mkdir -p ${LOG_DIR} && (setsid env ${COMMON_ENV[*]} ${PYTHON} -m torch.distributed.run --nnodes=2 --nproc_per_node=8 --node_rank=1 --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} -m wm3d_v3.training.train --cfg configs/${CFG} --print_every 2 > ${LOG_DIR}/${RUN_NAME}_node1.log 2>&1 < /dev/null & echo \$! > ${LOG_DIR}/${RUN_NAME}_node1.pid)"
}

launch_node0() {
  nohup env "${COMMON_ENV[@]}" "${PYTHON}" -m torch.distributed.run \
    --nnodes=2 \
    --nproc_per_node=8 \
    --node_rank=0 \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    -m wm3d_v3.training.train \
    --cfg "configs/${CFG}" \
    --print_every 2 \
    > "${LOG_DIR}/${RUN_NAME}_node0.log" 2>&1 &
  echo $! > "${LOG_DIR}/${RUN_NAME}_node0.pid"
}

launch_node1
launch_node0

echo "node0_pid=$(cat "${LOG_DIR}/${RUN_NAME}_node0.pid")"
echo "node1_pid=$(ssh "${WORKER_HOST}" "cat ${LOG_DIR}/${RUN_NAME}_node1.pid")"
echo "node0_log=${LOG_DIR}/${RUN_NAME}_node0.log"
echo "node1_log=${WORKER_HOST}:${LOG_DIR}/${RUN_NAME}_node1.log"
