#!/usr/bin/env bash
set -euo pipefail

: "${CONFIG:?Set CONFIG to one materialized WM3D YAML}"
: "${REPO_ROOT:?Set REPO_ROOT to the WM3D checkout}"
: "${MASTER_ADDR:?Set MASTER_ADDR}"
: "${MASTER_PORT:?Set MASTER_PORT}"
: "${NNODES:?Set NNODES}"
: "${GPUS_PER_NODE:?Set GPUS_PER_NODE}"
: "${NODE_RANK:?Set NODE_RANK in [0,NNODES)}"
: "${RDZV_ID:?Set a unique immutable run rendezvous id}"
: "${PREFLIGHT_REPORT:?Set a unique preflight report path}"
: "${NODE_LOG:?Set a unique per-node log path}"

for path_value in "${CONFIG}" "${REPO_ROOT}" "${PREFLIGHT_REPORT}" "${NODE_LOG}"; do
  if [[ "${path_value}" != /* ]]; then
    echo "Formal WM3D paths must be absolute: ${path_value}" >&2
    exit 2
  fi
done
if [[ ! "${RDZV_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{7,127}$ ]]; then
  echo "RDZV_ID must be an explicit 8-128 character immutable run id" >&2
  exit 2
fi
if [[ ! "${MASTER_PORT}" =~ ^[0-9]+$ ]]; then
  echo "MASTER_PORT must be an integer" >&2
  exit 2
fi
if (( MASTER_PORT < 1024 || MASTER_PORT > 65534 )); then
  echo "MASTER_PORT must leave room for the training port" >&2
  exit 2
fi
cd "${REPO_ROOT}"

if [[ -L "${REPO_ROOT}" || ! -d "${REPO_ROOT}" ]]; then
  echo "REPO_ROOT must be a real directory, not a symlink" >&2
  exit 2
fi
if [[ "$(realpath -e -- "${REPO_ROOT}")" != "${REPO_ROOT}" ]]; then
  echo "REPO_ROOT must be an absolute canonical path" >&2
  exit 2
fi
if [[ ! "${NNODES}" =~ ^[1-9][0-9]*$ || ! "${GPUS_PER_NODE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "NNODES and GPUS_PER_NODE must be positive integers" >&2
  exit 2
fi
if (( NODE_RANK < 0 || NODE_RANK >= NNODES )); then
  echo "NODE_RANK is outside NNODES" >&2
  exit 2
fi
if [[ -L "${CONFIG}" || ! -f "${CONFIG}" || "$(basename "${CONFIG}")" == "latest.yaml" ]]; then
  echo "CONFIG must be an explicit materialized YAML, never latest.yaml" >&2
  exit 2
fi
if [[ "$(realpath -e -- "${CONFIG}")" != "${CONFIG}" ]]; then
  echo "CONFIG must be an absolute canonical regular-file path" >&2
  exit 2
fi
if [[ -e "${NODE_LOG}" ]]; then
  echo "Refusing to append/overwrite formal node log: ${NODE_LOG}" >&2
  exit 2
fi
PYTHON_BIN="${PYTHON_BIN:-/opt/wm3d/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-/opt/wm3d/bin/torchrun}"
if [[ ! -x "${TORCHRUN_BIN}" || ! -x "${PYTHON_BIN}" ]]; then
  echo "Pinned runtime is missing torchrun/python: ${TORCHRUN_BIN} ${PYTHON_BIN}" >&2
  exit 2
fi
mkdir -p "$(dirname "${NODE_LOG}")" "$(dirname "${PREFLIGHT_REPORT}")"
exec > >(tee "${NODE_LOG}") 2>&1

GPU_COUNT="$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits | wc -l)"
if [[ "${GPU_COUNT}" != "${GPUS_PER_NODE}" ]]; then
  echo "Expected ${GPUS_PER_NODE} visible GPUs, got ${GPU_COUNT}" >&2
  exit 2
fi

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export NCCL_IB_DISABLE=0
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT,NET,GRAPH}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
export TORCHELASTIC_RUN_ID="${RDZV_ID}"

COMMON_ARGS=(
  --nnodes="${NNODES}"
  --nproc-per-node="${GPUS_PER_NODE}"
  --node-rank="${NODE_RANK}"
  --master-addr="${MASTER_ADDR}"
  --max-restarts=0
)

echo "[$(date --iso-8601=seconds)] WM3D distributed preflight run=${RDZV_ID}"
"${TORCHRUN_BIN}" \
  "${COMMON_ARGS[@]}" \
  --master-port="${MASTER_PORT}" \
  scripts/cluster/preflight_cluster.py \
  --config "${CONFIG}" \
  --report "${PREFLIGHT_REPORT}"

TRAIN_PORT="$((MASTER_PORT + 1))"
TRAIN_ARGS=(--config "${CONFIG}")
if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
  if [[ "${RESUME_CHECKPOINT}" != /* ]]; then
    echo "RESUME_CHECKPOINT must be absolute" >&2
    exit 2
  fi
  if [[ ! "$(basename "${RESUME_CHECKPOINT}")" =~ ^step_[0-9]{8}$ ]]; then
    echo "RESUME_CHECKPOINT must be an explicit step_XXXXXXXX directory" >&2
    exit 2
  fi
  if [[ -L "${RESUME_CHECKPOINT}" || ! -d "${RESUME_CHECKPOINT}" ]]; then
    echo "RESUME_CHECKPOINT must be a real directory, not a symlink" >&2
    exit 2
  fi
  if [[ "$(realpath -e -- "${RESUME_CHECKPOINT}")" != "${RESUME_CHECKPOINT}" ]]; then
    echo "RESUME_CHECKPOINT must be an absolute canonical path" >&2
    exit 2
  fi
  if [[ -L "${RESUME_CHECKPOINT}/COMMITTED.json" || ! -f "${RESUME_CHECKPOINT}/COMMITTED.json" ]]; then
    echo "RESUME_CHECKPOINT has no COMMITTED.json" >&2
    exit 2
  fi
  TRAIN_ARGS+=(--resume "${RESUME_CHECKPOINT}")
fi

echo "[$(date --iso-8601=seconds)] WM3D formal training run=${RDZV_ID}"
"${TORCHRUN_BIN}" \
  "${COMMON_ARGS[@]}" \
  --master-port="${TRAIN_PORT}" \
  wm3d/training/train.py \
  "${TRAIN_ARGS[@]}"
