#!/usr/bin/env bash
#SBATCH --job-name=wm3d
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=128
#SBATCH --exclusive
#SBATCH --time=35-00:00:00

set -euo pipefail

: "${CONFIG:?Export CONFIG as an absolute materialized YAML path}"
: "${REPO_ROOT:?Export REPO_ROOT as the shared WM3D checkout}"
: "${LOG_ROOT:?Export a new shared log directory}"
: "${RDZV_ID:?Export a unique run id}"
: "${GPUS_PER_NODE:?Export GPUS_PER_NODE}"

for path_value in "${CONFIG}" "${REPO_ROOT}" "${LOG_ROOT}"; do
  if [[ "${path_value}" != /* ]]; then
    echo "CONFIG, REPO_ROOT, and LOG_ROOT must be absolute" >&2
    exit 2
  fi
done
if [[ -e "${LOG_ROOT}" ]]; then
  echo "Refusing to reuse LOG_ROOT: ${LOG_ROOT}" >&2
  exit 2
fi
mkdir -p "${LOG_ROOT}"

mapfile -t HOSTS < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
MASTER_ADDR="${HOSTS[0]}"
MASTER_PORT="${MASTER_PORT:-29400}"
NNODES="${SLURM_NNODES}"
PREFLIGHT_REPORT="${LOG_ROOT}/cluster_preflight.json"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"

if [[ ! "${NNODES}" =~ ^[1-9][0-9]*$ || ! "${GPUS_PER_NODE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "NNODES and GPUS_PER_NODE must be positive integers" >&2
  exit 2
fi
if (( ${#HOSTS[@]} != NNODES )); then
  echo "Slurm host expansion does not match NNODES" >&2
  exit 2
fi

export CONFIG REPO_ROOT LOG_ROOT RDZV_ID
export MASTER_ADDR MASTER_PORT NNODES PREFLIGHT_REPORT RESUME_CHECKPOINT

srun --export=ALL --kill-on-bad-exit=1 --label bash -lc '
  set -euo pipefail
  ulimit -l unlimited
  ulimit -n 1048576
  export NODE_RANK="${SLURM_NODEID}"
  export NODE_LOG="${LOG_ROOT}/node_$(printf "%03d" "${SLURM_NODEID}").log"
  exec "${REPO_ROOT}/scripts/internal/launch_torchrun_node.sh"
'
