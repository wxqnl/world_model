#!/usr/bin/env bash
set -euo pipefail

cd /data/Minko/world_model/wm3d_v3

CFG="v3_p64_140m_stage0_visual_depth_stabilized_noreset_resume7500_to10000_eval_v1.yaml"
RUN_NAME="train_140m_stage0_visual_depth_stabilized_noreset_resume7500_to10000_eval_v1"
MASTER_ADDR="${MASTER_ADDR:-172.27.0.6}"
MASTER_PORT="${MASTER_PORT:-29745}"
IFACE="${IFACE:-bond0.1411}"
PYTHON="${PYTHON:-/data/Minko/.venvs/wm3d/bin/python}"
LOG_DIR="${LOG_DIR:-/data/Minko/logs}"
MANIFEST="manifests/oxe_droid20k_depthplus_world_v1.jsonl"
ACTION_STATS="/data/Minko/datasets/cache/wm3d_v3/action_stats_oxe_droid20k_stage1_world_v1.npz"
RESUME_REL="results/wm3d_v3_p64_140m_stage0_visual_depth_stabilized_4node_v1/ckpt/step_00007500.pt"
WORKER_HOSTS=(root@172.27.0.7 root@172.27.0.4 root@172.27.0.5)
NNODES=$((1 + ${#WORKER_HOSTS[@]}))

mkdir -p "${LOG_DIR}"

for host in "${WORKER_HOSTS[@]}"; do
  ssh "${host}" "mkdir -p /data/Minko/world_model/wm3d_v3/configs /data/Minko/world_model/wm3d_v3/manifests /data/Minko/world_model/wm3d_v3/$(dirname "${RESUME_REL}") /data/Minko/logs /data/Minko/datasets/cache/wm3d_v3"
  rsync -a "configs/${CFG}" "${host}:/data/Minko/world_model/wm3d_v3/configs/${CFG}"
  rsync -a "${MANIFEST}" "${host}:/data/Minko/world_model/wm3d_v3/${MANIFEST}"
  rsync -a "${ACTION_STATS}" "${host}:${ACTION_STATS}"
  rsync -a "${RESUME_REL}" "${host}:/data/Minko/world_model/wm3d_v3/${RESUME_REL}"
done

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

TRAIN_ARGS=(--cfg "configs/${CFG}" --print_every 25 --resume "/data/Minko/world_model/wm3d_v3/${RESUME_REL}")

rank=1
for host in "${WORKER_HOSTS[@]}"; do
  ssh "${host}" "cd /data/Minko/world_model/wm3d_v3 && mkdir -p ${LOG_DIR} && (setsid env ${COMMON_ENV[*]} ${PYTHON} -m torch.distributed.run --nnodes=${NNODES} --nproc_per_node=8 --node_rank=${rank} --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} -m wm3d_v3.training.train ${TRAIN_ARGS[*]} > ${LOG_DIR}/${RUN_NAME}_node${rank}.log 2>&1 < /dev/null & echo \$! > ${LOG_DIR}/${RUN_NAME}_node${rank}.pid)"
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
