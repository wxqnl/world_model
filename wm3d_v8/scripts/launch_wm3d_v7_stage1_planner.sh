#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/Minko/world_model/wm3d_v7_actionrepair1b_20260806
PY=/data/Minko/.venvs/wm3d/bin/python
CFG=${1:?usage: launch_wm3d_v7_stage1_planner.sh CONFIG NODE_RANK}
NODE_RANK=${2:?NODE_RANK must be 0 (node44), 1 (node41), or 2 (node43)}
MASTER_ADDR=${MASTER_ADDR:-172.27.0.7}
MASTER_PORT=${MASTER_PORT:-29931}
CONFIRM=${WM3D_V7_STAGE1_PLANNER_CONFIRM:-}
EXPECTED_CONFIRM=EXECUTE_WM3D_V7_STAGE1_PLANNER_PHASE
RESUME=${WM3D_V7_STAGE1_RESUME:-}
RESUME_SHA=${WM3D_V7_STAGE1_RESUME_SHA256:-}

cd "${ROOT}"
case "${CFG}" in
  configs/wm3d_v7_stage1_planner_dynamics10k.yaml|\
  configs/wm3d_v7_stage1_planner_planner10k.yaml|\
  configs/wm3d_v7_stage1_planner_joint5k.yaml) ;;
  *) echo "unapproved Stage1-P config: ${CFG}" >&2; exit 1 ;;
esac
case "${NODE_RANK}" in 0|1|2) ;; *) echo "invalid node rank" >&2; exit 1 ;; esac
if [[ "${CONFIRM}" != "${EXPECTED_CONFIRM}" ]]; then
  echo "Stage1-P confirmation mismatch" >&2
  exit 1
fi
if pgrep -af "[w]m3d_v3.training.train.*wm3d_v7_1b_native_actionpolicy_joint_formal100k_3node24_v3" >/dev/null; then
  echo "Stage0 formal training is active; Stage1-P launch is forbidden" >&2
  exit 1
fi
if pgrep -af "[w]m3d_v3.stage1_planner.train.*${CFG}" >/dev/null; then
  echo "duplicate Stage1-P process for ${CFG}" >&2
  exit 1
fi

NAME=$(basename "${CFG}" .yaml)
LOG_DIR=${ROOT}/logs/${NAME}
LOG=${LOG_DIR}/train_rank${NODE_RANK}.log
PID_FILE=${LOG_DIR}/launcher_rank${NODE_RANK}.pid
PREFLIGHT_REPORT=${LOG_DIR}/preflight_rank${NODE_RANK}.json
mkdir -p "${LOG_DIR}"
PREFLIGHT_ARGS=(--cfg "${CFG}" --mode train --report "${PREFLIGHT_REPORT}")
TRAIN_ARGS=(--cfg "${CFG}")
if [[ -n "${RESUME}" || -n "${RESUME_SHA}" ]]; then
  if [[ -z "${RESUME}" || -z "${RESUME_SHA}" ]]; then
    echo "resume path and SHA256 must be provided together" >&2
    exit 1
  fi
  PREFLIGHT_ARGS+=(--resume "${RESUME}" --resume-sha256 "${RESUME_SHA}")
  TRAIN_ARGS+=(--resume "${RESUME}")
fi
"${PY}" scripts/preflight_wm3d_v7_stage1_planner.py "${PREFLIGHT_ARGS[@]}"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=8
case "${NODE_RANK}" in
  0|1) NCCL_IB_HCA_ALLOWLIST=mlx5_0,mlx5_1,mlx5_4,mlx5_5,mlx5_7,mlx5_8,mlx5_9,mlx5_10 ;;
  2) NCCL_IB_HCA_ALLOWLIST=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_5,mlx5_6,mlx5_7,mlx5_8 ;;
esac
export NCCL_IB_DISABLE=0
export NCCL_NET=IB
export NCCL_IB_HCA="${NCCL_IB_HCA_ALLOWLIST}"
export NCCL_NET_GDR_LEVEL=2
export NCCL_SOCKET_IFNAME=bond0.1411
export NCCL_SOCKET_FAMILY=AF_INET
export GLOO_SOCKET_IFNAME=bond0.1411
export NCCL_NVLS_ENABLE=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_NCCL_TRACE_BUFFER_SIZE=1048576
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

setsid "${PY}" -m torch.distributed.run \
  --nnodes=3 \
  --nproc_per_node=8 \
  --node_rank="${NODE_RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  -m wm3d_v3.stage1_planner.train \
  "${TRAIN_ARGS[@]}" \
  > "${LOG}" 2>&1 < /dev/null &
pid=$!
echo "${pid}" > "${PID_FILE}"
echo "${pid}"
