#!/usr/bin/env bash
#SBATCH --job-name=wm3d-v7-eval
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=128
#SBATCH --exclusive
#SBATCH --time=08:00:00

set -euo pipefail

: "${CONFIG:?Set CONFIG}"
: "${CHECKPOINT:?Set CHECKPOINT}"
: "${EVAL_OUTPUT_ROOT:?Set EVAL_OUTPUT_ROOT}"
: "${REPO_ROOT:?Set REPO_ROOT}"
: "${EVAL_LOG_ROOT:?Set EVAL_LOG_ROOT}"

if [[ -e "${EVAL_LOG_ROOT}" || -e "${EVAL_OUTPUT_ROOT}" ]]; then
  echo "拒绝复用 eval log/output" >&2
  exit 2
fi
mkdir -p "${EVAL_LOG_ROOT}"
mapfile -t HOSTS < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
MASTER_ADDR="${HOSTS[0]}"
MASTER_PORT="${MASTER_PORT:-29600}"
NNODES="${SLURM_NNODES}"
if [[ "${NNODES}" != "8" && "${NNODES}" != "16" ]]; then
  echo "eval 需要 8 或 16 节点" >&2
  exit 2
fi
export CONFIG CHECKPOINT EVAL_OUTPUT_ROOT REPO_ROOT EVAL_LOG_ROOT
export MASTER_ADDR MASTER_PORT NNODES

srun --export=ALL --kill-on-bad-exit=1 --label bash -lc '
  set -euo pipefail
  ulimit -l unlimited
  ulimit -n 1048576
  export NODE_RANK="${SLURM_NODEID}"
  export NODE_LOG="${EVAL_LOG_ROOT}/node_$(printf "%03d" "${SLURM_NODEID}").log"
  exec "${REPO_ROOT}/scripts/scale5b/launch_eval_node.sh"
'
