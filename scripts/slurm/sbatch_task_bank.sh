#!/usr/bin/env bash
#SBATCH --job-name=wm3d-task-bank
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00

set -euo pipefail

: "${REPO_ROOT:?Set the absolute WM3D checkout}"
: "${DATASET_ROOT:?Set the absolute prepared dataset root}"
: "${ENCODER_ASSET_ROOT:?Set the immutable encoder asset bundle}"
: "${TASK_MODEL_REVISION:?Set the pinned task-model revision}"

for value in "${REPO_ROOT}" "${DATASET_ROOT}" "${ENCODER_ASSET_ROOT}"; do
  if [[ "${value}" != /* ]]; then
    echo "All roots must be absolute" >&2
    exit 2
  fi
done

PYTHON_BIN="${PYTHON_BIN:-/opt/wm3d/bin/python}"
cd "${REPO_ROOT}"
exec "${PYTHON_BIN}" scripts/data/build_task_bank.py \
  --episode-plan "${DATASET_ROOT}/control/episode_plan.jsonl" \
  --output-root "${DATASET_ROOT}" \
  --asset-root "${ENCODER_ASSET_ROOT}" \
  --model "${TASK_MODEL:-google/flan-t5-xl}" \
  --revision "${TASK_MODEL_REVISION}" \
  --device cuda \
  --batch-size "${TASK_BATCH_SIZE:-32}" \
  --max-length "${TASK_MAX_LENGTH:-128}"
