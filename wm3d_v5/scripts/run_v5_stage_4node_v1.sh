#!/usr/bin/env bash
set -euo pipefail

cd /data/Minko/world_model/wm3d_v5

MASTER_ADDR="${MASTER_ADDR:-172.27.0.6}"
MASTER_PORT="${MASTER_PORT:-29590}"
IFACE="${IFACE:-bond0.1411}"
PYTHON="${PYTHON:-/data/Minko/.venvs/wm3d/bin/python}"
CFG="${CFG:?CFG is required}"
RUN_NAME="${RUN_NAME:?RUN_NAME is required}"
LOG_DIR="${LOG_DIR:-/data/Minko/logs}"
RESUME_CKPT="${RESUME_CKPT:-}"
RESET_OPTIM="${RESET_OPTIM:-1}"

WORKER_HOSTS=(${WORKER_HOSTS:-root@172.27.0.7 root@172.27.0.4})
NNODES=$((1 + ${#WORKER_HOSTS[@]}))

mkdir -p "${LOG_DIR}"

sync_worker_code() {
  local host="$1"
  ssh "${host}" "mkdir -p /data/Minko/world_model/wm3d_v5 /data/Minko/logs /data/Minko/datasets/cache/wm3d_v3"
  rsync -a --delete --exclude=results --exclude=.git --exclude='__pycache__' --exclude='*.pyc' \
    ./ "${host}:/data/Minko/world_model/wm3d_v5/"
}

for host in "${WORKER_HOSTS[@]}"; do
  sync_worker_code "${host}"
done

TRAIN_ARGS=(--cfg "configs/${CFG}" --print_every 25)
if [[ -n "${RESUME_CKPT}" ]]; then
  TRAIN_ARGS+=(--resume "${RESUME_CKPT}")
  if [[ "${RESET_OPTIM}" == "1" ]]; then
    TRAIN_ARGS+=(--reset_optim)
  fi
fi

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

rank=1
for host in "${WORKER_HOSTS[@]}"; do
  worker_args=("${TRAIN_ARGS[@]}")
  if [[ -n "${RESUME_CKPT}" ]]; then
    if [[ "${RESUME_CKPT}" = /* ]]; then
      worker_resume="${RESUME_CKPT}"
    else
      worker_resume="/data/Minko/world_model/wm3d_v5/${RESUME_CKPT}"
    fi
    ssh "${host}" "mkdir -p $(dirname "${worker_resume}")"
    rsync -a "${RESUME_CKPT}" "${host}:${worker_resume}"
    worker_args=(--cfg "configs/${CFG}" --print_every 25 --resume "${worker_resume}")
    if [[ "${RESET_OPTIM}" == "1" ]]; then
      worker_args+=(--reset_optim)
    fi
  fi
  ssh "${host}" "cd /data/Minko/world_model/wm3d_v5 && mkdir -p ${LOG_DIR} && (setsid env ${COMMON_ENV[*]} ${PYTHON} -m torch.distributed.run --nnodes=${NNODES} --nproc_per_node=8 --node_rank=${rank} --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} -m wm3d_v3.training.train ${worker_args[*]} > ${LOG_DIR}/${RUN_NAME}_node${rank}.log 2>&1 < /dev/null & echo \$! > ${LOG_DIR}/${RUN_NAME}_node${rank}.pid)"
  rank=$((rank + 1))
done

nohup env "${COMMON_ENV[@]}" "${PYTHON}" -m torch.distributed.run \
  --nnodes="${NNODES}" \
  --nproc_per_node=8 \
  --node_rank=0 \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  -m wm3d_v3.training.train \
  "${TRAIN_ARGS[@]}" \
  > "${LOG_DIR}/${RUN_NAME}_node0.log" 2>&1 &
echo $! > "${LOG_DIR}/${RUN_NAME}_node0.pid"

echo "nnodes=${NNODES}"
echo "node0_pid=$(cat "${LOG_DIR}/${RUN_NAME}_node0.pid")"
echo "node0_log=${LOG_DIR}/${RUN_NAME}_node0.log"
rank=1
for host in "${WORKER_HOSTS[@]}"; do
  echo "node${rank}_pid=$(ssh "${host}" "cat ${LOG_DIR}/${RUN_NAME}_node${rank}.pid")"
  echo "node${rank}_log=${host}:${LOG_DIR}/${RUN_NAME}_node${rank}.log"
  rank=$((rank + 1))
done
