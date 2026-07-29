#!/usr/bin/env bash
# Submit with, for example: sbatch --array=0-255%64 sbatch_action_stats_array.sh
#SBATCH --job-name=wm3d-v7-stats
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00

set -euo pipefail

: "${REPO_ROOT:?Set the absolute wm3d_v7 checkout}"
: "${DATASET_ROOT:?Set the absolute prepared dataset root}"
: "${NUM_SHARDS:?Set NUM_SHARDS to the Slurm array task count}"

if [[ "${REPO_ROOT}" != /* || "${DATASET_ROOT}" != /* ]]; then
  echo "REPO_ROOT and DATASET_ROOT must be absolute" >&2
  exit 2
fi
if [[ "${SLURM_ARRAY_TASK_MIN:-}" != "0" ]]; then
  echo "The formal action-statistics array must start at zero" >&2
  exit 2
fi
if [[ "${SLURM_ARRAY_TASK_COUNT:-0}" != "${NUM_SHARDS}" ]]; then
  echo "Slurm array count must equal NUM_SHARDS" >&2
  exit 2
fi

PYTHON_BIN="${PYTHON_BIN:-/opt/wm3d/bin/python}"
PARTIAL_ROOT="${DATASET_ROOT}/control/action_stats_partials"
mkdir -p "${PARTIAL_ROOT}"
OUTPUT="${PARTIAL_ROOT}/partial_$(printf '%05d' "${SLURM_ARRAY_TASK_ID}").npz"
if [[ -e "${OUTPUT}" ]]; then
  echo "Refusing to overwrite ${OUTPUT}" >&2
  exit 2
fi

cd "${REPO_ROOT}"
exec "${PYTHON_BIN}" scripts/scale5b/build_action_stats.py partial \
  --episode-plan "${DATASET_ROOT}/control/episode_plan.jsonl" \
  --output "${OUTPUT}" \
  --shard-id "${SLURM_ARRAY_TASK_ID}" \
  --num-shards "${NUM_SHARDS}" \
  --global-sample-budget "${GLOBAL_SAMPLE_BUDGET:-8000000}"
