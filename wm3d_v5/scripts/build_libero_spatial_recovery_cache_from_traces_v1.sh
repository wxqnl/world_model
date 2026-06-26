#!/usr/bin/env bash
set -euo pipefail

cd /data/Minko/world_model/wm3d_v5

TRACE_BENCH_ROOT="${TRACE_BENCH_ROOT:?TRACE_BENCH_ROOT is required}"
EPISODE_LIST="${EPISODE_LIST:?EPISODE_LIST is required}"
CACHE_ROOT="${CACHE_ROOT:-/data/Minko/world_model/wm3d_v5/cache/libero_spatial_taskboost_recovery_T16_k32_v1}"
ACTION_STATS="${ACTION_STATS:-/data/Minko/world_model/wm3d_v5/cache/libero_action_policy_all_dualcam_concat_T16_k32_s4_rot_spstat_v1/action_stats.npz}"
PY="${PY:-/data/Minko/.venvs/wm3d/bin/python}"
LIBERO_ROOT="${LIBERO_ROOT:-/data/Minko/benchmarks/LIBERO}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
CONTEXT_T="${CONTEXT_T:-16}"
POLICY_HORIZON="${POLICY_HORIZON:-32}"
STRIDE="${STRIDE:-4}"
MAX_WINDOWS="${MAX_WINDOWS:-96}"
SAMPLE_WEIGHT="${SAMPLE_WEIGHT:-4.0}"
PHASE_PRIOR_WEIGHT="${PHASE_PRIOR_WEIGHT:-0.05}"
MAX_ALIGN_DISTANCE="${MAX_ALIGN_DISTANCE:-0.0}"
OBJECT_STATE_WEIGHT="${OBJECT_STATE_WEIGHT:-0.0}"
LOG_ROOT="${LOG_ROOT:-/data/Minko/logs/libero_recovery_cache}"

mkdir -p "${CACHE_ROOT}" "${LOG_ROOT}"

export PYTHONPATH="/data/Minko/world_model/wm3d_v5:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-/data/Minko/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/data/Minko/.cache/huggingface/hub}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false
export MUJOCO_GL=egl
export __EGL_VENDOR_LIBRARY_FILENAMES=/data/Minko/egl/10_nvidia.json

HDF5_MAP="${CACHE_ROOT}/task_hdf5_map.json"
"${PY}" - "${LIBERO_ROOT}" "${HDF5_MAP}" <<'PY'
import json
import os
import sys
import yaml
from pathlib import Path

root = Path(sys.argv[1])
out = Path(sys.argv[2])
config_dir = root / ".wm3d_libero_config"
config_dir.mkdir(parents=True, exist_ok=True)
(config_dir / "config.yaml").write_text(yaml.safe_dump({
    "benchmark_root": str(root / "libero" / "libero"),
    "bddl_files": str(root / "libero" / "libero" / "bddl_files"),
    "init_states": str(root / "libero" / "libero" / "init_files"),
    "datasets": str(root / "datasets"),
    "assets": str(root / "libero" / "libero" / "assets"),
}, sort_keys=True))
os.environ["LIBERO_CONFIG_PATH"] = str(config_dir)
sys.path.insert(0, str(root))
from libero.libero.benchmark import get_benchmark
suite = get_benchmark("libero_spatial")(0)
mapping = {}
for task_id in range(suite.get_num_tasks()):
    task = suite.get_task(task_id)
    mapping[str(task_id)] = str(root / "datasets" / "libero_spatial" / f"{task.name}_demo.hdf5")
out.write_text(json.dumps(mapping, indent=2, sort_keys=True))
print(json.dumps(mapping, sort_keys=True))
PY

IFS=',' read -r -a GPU_ARR <<< "${GPUS}"
NGPU="${#GPU_ARR[@]}"

run_one() {
  local task_id="$1"
  local init_id="$2"
  local gpu="$3"
  local rollout_json="${TRACE_BENCH_ROOT}/episodes/libero_spatial_task$(printf '%02d' "${task_id}")_init$(printf '%02d' "${init_id}").json"
  local out_dir="${CACHE_ROOT}/task${task_id}/init${init_id}"
  local log_file="${LOG_ROOT}/recovery_cache_task${task_id}_init${init_id}_$(date +%Y%m%d_%H%M%S).log"
  local hdf5_path
  hdf5_path="$("${PY}" - "${HDF5_MAP}" "${task_id}" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))[str(int(sys.argv[2]))])
PY
)"
  if [[ ! -s "${rollout_json}" ]]; then
    echo "missing rollout: ${rollout_json}" >&2
    return 2
  fi
  if [[ ! -s "${hdf5_path}" ]]; then
    echo "missing hdf5: ${hdf5_path}" >&2
    return 3
  fi
  rm -rf "${out_dir}"
  mkdir -p "${out_dir}"
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 "${PY}" -m wm3d_v3.benchmarks.libero_rollout_recovery_cache \
    --rollout_json "${rollout_json}" \
    --expert_hdf5 "${hdf5_path}" \
    --demo_id "demo_${init_id}" \
    --out_dir "${out_dir}" \
    --action_stats "${ACTION_STATS}" \
    --T "${CONTEXT_T}" \
    --k "${POLICY_HORIZON}" \
    --stride "${STRIDE}" \
    --max_windows "${MAX_WINDOWS}" \
    --token_grid 8 \
    --device cuda:0 \
    --qwen_device cuda:0 \
    --phase_prior_weight "${PHASE_PRIOR_WEIGHT}" \
    --object_state_weight "${OBJECT_STATE_WEIGHT}" \
    --max_align_distance "${MAX_ALIGN_DISTANCE}" \
    --sample_weight "${SAMPLE_WEIGHT}" \
    --monotonic \
    --log_every 50 \
    > "${log_file}" 2>&1
}

active=0
ordinal=0
while read -r task_id init_id _rest; do
  [[ -z "${task_id:-}" || "${task_id:0:1}" == "#" ]] && continue
  gpu="${GPU_ARR[$((ordinal % NGPU))]}"
  run_one "${task_id}" "${init_id}" "${gpu}" &
  active=$((active + 1))
  ordinal=$((ordinal + 1))
  if (( active >= NGPU )); then
    wait -n
    active=$((active - 1))
  fi
done < "${EPISODE_LIST}"
wait

manifest="${CACHE_ROOT}/manifest.jsonl"
: > "${manifest}.tmp"
for part in "${CACHE_ROOT}"/task*/init*/manifest.jsonl; do
  [[ -s "${part}" ]] && cat "${part}" >> "${manifest}.tmp"
done
mv "${manifest}.tmp" "${manifest}"

"${PY}" - "${CACHE_ROOT}" "${manifest}" "${EPISODE_LIST}" <<'PY'
import json
import sys
from pathlib import Path
root = Path(sys.argv[1])
manifest = Path(sys.argv[2])
episode_list = Path(sys.argv[3])
rows = sum(1 for line in manifest.open() if line.strip()) if manifest.exists() else 0
episodes = sum(1 for line in episode_list.open() if line.strip() and not line.lstrip().startswith("#"))
summary = {
    "cache_root": str(root),
    "manifest": str(manifest),
    "episode_list": str(episode_list),
    "episodes": episodes,
    "cache_windows": rows,
}
(root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
print(json.dumps(summary, sort_keys=True))
PY
