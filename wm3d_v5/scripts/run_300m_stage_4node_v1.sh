#!/usr/bin/env bash
set -euo pipefail

cd /data/Minko/world_model/wm3d_v3

MASTER_ADDR="${MASTER_ADDR:-172.27.0.6}"
MASTER_PORT="${MASTER_PORT:-29570}"
IFACE="${IFACE:-bond0.1411}"
PYTHON="${PYTHON:-/data/Minko/.venvs/wm3d/bin/python}"
CFG="${CFG:?CFG is required}"
RUN_NAME="${RUN_NAME:?RUN_NAME is required}"
MANIFEST="${MANIFEST:-manifests/oxe_droid20k_balanced_world_v2.jsonl}"
BASE_MANIFEST="${BASE_MANIFEST:-manifests/oxe_droid20k_stage1_world_v1.jsonl}"
ACTION_STATS="${ACTION_STATS:-/data/Minko/datasets/cache/wm3d_v3/action_stats_oxe_droid20k_stage1_world_v1.npz}"
LOG_DIR="${LOG_DIR:-/data/Minko/logs}"
SYNC_CACHE="${SYNC_CACHE:-0}"
RESUME_CKPT="${RESUME_CKPT:-}"
RESET_OPTIM="${RESET_OPTIM:-1}"

WORKER_HOSTS=(${WORKER_HOSTS:-root@172.27.0.7 root@172.27.0.4 root@172.27.0.5})
CACHE_WORKER_HOSTS=(${CACHE_WORKER_HOSTS:-root@172.27.0.4 root@172.27.0.5})
NNODES=$((1 + ${#WORKER_HOSTS[@]}))

mkdir -p "${LOG_DIR}"

if [[ ! -f "${MANIFEST}" ]]; then
  "${PYTHON}" scripts/build_oxe_droid_balanced_manifest_v1.py \
    --input "${BASE_MANIFEST}" \
    --output "${MANIFEST}" \
    --target_records 160000 \
    --weights "fractal=0.25,bridge=0.25,droid=0.25,small_robot=0.25" \
    --seed 606
fi

sync_worker_code() {
  local host="$1"
  ssh "${host}" "mkdir -p /data/Minko/world_model/wm3d_v3 /data/Minko/logs /data/Minko/datasets/cache/wm3d_v3"
  rsync -a --delete --exclude=results --exclude=.git --exclude='__pycache__' --exclude='*.pyc' \
    ./ "${host}:/data/Minko/world_model/wm3d_v3/"
}

for host in "${WORKER_HOSTS[@]}"; do
  sync_worker_code "${host}"
done

if [[ "${SYNC_CACHE}" == "1" ]]; then
  for host in "${CACHE_WORKER_HOSTS[@]}"; do
    WORKER_HOST="${host}" MANIFEST="${MANIFEST}" ACTION_STATS="${ACTION_STATS}" \
      scripts/sync_manifest_cache_files_v1.sh
  done
fi
for host in "${WORKER_HOSTS[@]}"; do
  rsync -a "${MANIFEST}" "${host}:/data/Minko/world_model/wm3d_v3/${MANIFEST}"
  rsync -a "${ACTION_STATS}" "${host}:${ACTION_STATS}"
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
      worker_resume="/data/Minko/world_model/wm3d_v3/${RESUME_CKPT}"
    fi
    ssh "${host}" "mkdir -p $(dirname "${worker_resume}")"
    rsync -a "${RESUME_CKPT}" "${host}:${worker_resume}"
    worker_args=(--cfg "configs/${CFG}" --print_every 25 --resume "${worker_resume}")
    if [[ "${RESET_OPTIM}" == "1" ]]; then
      worker_args+=(--reset_optim)
    fi
  fi
  ssh "${host}" "cd /data/Minko/world_model/wm3d_v3 && mkdir -p ${LOG_DIR} && (setsid env ${COMMON_ENV[*]} ${PYTHON} -m torch.distributed.run --nnodes=${NNODES} --nproc_per_node=8 --node_rank=${rank} --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} -m wm3d_v3.training.train ${worker_args[*]} > ${LOG_DIR}/${RUN_NAME}_node${rank}.log 2>&1 < /dev/null & echo \$! > ${LOG_DIR}/${RUN_NAME}_node${rank}.pid)"
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
