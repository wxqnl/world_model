#!/usr/bin/env bash
# Submit with, for example: sbatch --array=0-1023%128 sbatch_encode_array.sh
#SBATCH --job-name=wm3d-encode
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00

set -euo pipefail

: "${REPO_ROOT:?Set the absolute WM3D checkout}"
: "${DATASET_ROOT:?Set the absolute prepared dataset root}"
: "${ENCODER_ASSET_ROOT:?Set the immutable encoder asset bundle}"
: "${VGGT_REVISION:?Set the pinned VGGT model revision}"
: "${NUM_SHARDS:?Set NUM_SHARDS to the Slurm array task count}"

for value in "${REPO_ROOT}" "${DATASET_ROOT}" "${ENCODER_ASSET_ROOT}"; do
  if [[ "${value}" != /* ]]; then
    echo "All roots must be absolute" >&2
    exit 2
  fi
done
if [[ "${RESUME_ARRAY:-0}" != "1" && "${SLURM_ARRAY_TASK_MIN:-}" != "0" ]]; then
  echo "The formal encoder array must start at zero" >&2
  exit 2
fi
if [[ "${RESUME_ARRAY:-0}" != "1" \
  && "${SLURM_ARRAY_TASK_COUNT:-0}" != "${NUM_SHARDS}" ]]; then
  echo "Slurm array count must equal NUM_SHARDS" >&2
  exit 2
fi
if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= NUM_SHARDS )); then
  echo "SLURM_ARRAY_TASK_ID is outside [0, NUM_SHARDS)" >&2
  exit 2
fi

PYTHON_BIN="${PYTHON_BIN:-/opt/wm3d/bin/python}"
export PYTHONDONTWRITEBYTECODE=1
cd "${REPO_ROOT}"
exec "${PYTHON_BIN}" scripts/internal/encode_shard.py \
  --dataset-contract "${DATASET_ROOT}/control/dataset_contract.json" \
  --episode-plan "${DATASET_ROOT}/control/episode_plan.jsonl" \
  --action-stats "${DATASET_ROOT}/control/action_stats.json" \
  --task-index "${DATASET_ROOT}/control/task_index.json" \
  --output-root "${DATASET_ROOT}" \
  --asset-root "${ENCODER_ASSET_ROOT}" \
  --shard-id "${SLURM_ARRAY_TASK_ID}" \
  --num-shards "${NUM_SHARDS}" \
  --max-part-frames "${MAX_PART_FRAMES:-512}" \
  --window-stride "${WINDOW_STRIDE:-4}" \
  --encoder-batch-frames "${ENCODER_BATCH_FRAMES:-4}" \
  --encoder-input-size "${ENCODER_INPUT_SIZE:-518}" \
  --jpeg-quality "${JPEG_QUALITY:-92}" \
  --vggt-model "${VGGT_MODEL:-facebook/VGGT-1B}" \
  --vggt-revision "${VGGT_REVISION}" \
  --device cuda
