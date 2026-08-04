#!/usr/bin/env bash
set -euo pipefail

: "${CONFIG:?Set CONFIG to a materialized WM3D YAML}"
: "${CHECKPOINT:?Set CHECKPOINT to step_XXXXXXXX}"
: "${EVAL_OUTPUT_ROOT:?Set a new eval output directory}"
: "${REPO_ROOT:?Set REPO_ROOT}"
: "${MASTER_ADDR:?Set MASTER_ADDR}"
: "${MASTER_PORT:?Set MASTER_PORT}"
: "${NNODES:?Set NNODES}"
: "${GPUS_PER_NODE:?Set GPUS_PER_NODE}"
: "${NODE_RANK:?Set NODE_RANK}"
: "${NODE_LOG:?Set a new node log}"

PYTHON_BIN="${PYTHON_BIN:-/opt/wm3d/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-/opt/wm3d/bin/torchrun}"
for value in "${CONFIG}" "${CHECKPOINT}" "${EVAL_OUTPUT_ROOT}" "${REPO_ROOT}" "${NODE_LOG}"; do
  if [[ "${value}" != /* ]]; then
    echo "eval 路径必须是绝对路径：${value}" >&2
    exit 2
  fi
done
if [[ ! "${NNODES}" =~ ^[1-9][0-9]*$ || ! "${GPUS_PER_NODE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "NNODES 和 GPUS_PER_NODE 必须是正整数" >&2
  exit 2
fi
if (( NODE_RANK < 0 || NODE_RANK >= NNODES )); then
  echo "NODE_RANK 越界" >&2
  exit 2
fi
if [[ ! "$(basename "${CHECKPOINT}")" =~ ^step_[0-9]{8}$ ]] \
  || [[ -L "${CHECKPOINT}" || ! -f "${CHECKPOINT}/COMMITTED.json" ]]; then
  echo "CHECKPOINT 必须是完整编号 checkpoint" >&2
  exit 2
fi
if [[ -e "${NODE_LOG}" ]]; then
  echo "拒绝覆盖 eval 日志：${NODE_LOG}" >&2
  exit 2
fi
if [[ ! -x "${PYTHON_BIN}" || ! -x "${TORCHRUN_BIN}" ]]; then
  echo "eval runtime 不完整" >&2
  exit 2
fi

mkdir -p "$(dirname "${NODE_LOG}")"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export NCCL_IB_DISABLE=0
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
exec > >(tee "${NODE_LOG}") 2>&1

exec "${TORCHRUN_BIN}" \
  --nnodes="${NNODES}" --nproc-per-node="${GPUS_PER_NODE}" --node-rank="${NODE_RANK}" \
  --master-addr="${MASTER_ADDR}" --master-port="${MASTER_PORT}" \
  --max-restarts=0 \
  wm3d/training/eval.py \
  --config "${CONFIG}" --checkpoint "${CHECKPOINT}" \
  --output-root "${EVAL_OUTPUT_ROOT}" --steps "${EVAL_STEPS:-64}"
