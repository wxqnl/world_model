#!/usr/bin/env bash
set -euo pipefail

cd /data/Minko/world_model/wm3d_v5

export PYTHONPATH="/data/Minko/world_model/wm3d_v5:${PYTHONPATH:-}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export TOKENIZERS_PARALLELISM=false

PY="${PY:-/data/Minko/.venvs/wm3d/bin/python}"
LIBERO_DATA_ROOT="${LIBERO_DATA_ROOT:-/data/Minko/benchmarks/LIBERO/datasets}"
OUT_ROOT="${OUT_ROOT:-/0604-10T-test/wm3d_v5/cache/libero_action_policy_full_T16_k8_s4_v1}"
MANIFEST_OUT="${MANIFEST_OUT:-/data/Minko/world_model/wm3d_v5/manifests/libero_action_policy_full_T16_k8_s4_v1.jsonl}"
SOURCE_JSONL="${SOURCE_JSONL:-${OUT_ROOT}/source_windows.jsonl}"
SUMMARY_JSON="${SUMMARY_JSON:-${OUT_ROOT}/source_windows_summary.json}"
SPLIT_DIR="${SPLIT_DIR:-${OUT_ROOT}/splits}"
TASK_CACHE_DIR="${TASK_CACHE_DIR:-/0604-10T-test/wm3d_v5/cache/libero_taskemb}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
STRIDE="${STRIDE:-4}"
MAX_DEMOS_PER_FILE="${MAX_DEMOS_PER_FILE:-0}"
MAX_WINDOWS_PER_SHARD="${MAX_WINDOWS_PER_SHARD:-0}"

mkdir -p "${OUT_ROOT}" "${SPLIT_DIR}" "$(dirname "${MANIFEST_OUT}")" "${TASK_CACHE_DIR}"
export OUT_ROOT MANIFEST_OUT SOURCE_JSONL SUMMARY_JSON SPLIT_DIR TASK_CACHE_DIR GPUS

export_windows() {
  if [[ -s "${SOURCE_JSONL}" && -s "${SUMMARY_JSON}" ]]; then
    echo "[cache] source windows already exist: ${SOURCE_JSONL}"
    return
  fi
  "${PY}" -m wm3d_v3.benchmarks.libero_demo_export \
    --input \
      "${LIBERO_DATA_ROOT}/libero_10" \
      "${LIBERO_DATA_ROOT}/libero_goal" \
      "${LIBERO_DATA_ROOT}/libero_object" \
      "${LIBERO_DATA_ROOT}/libero_spatial" \
    --out_jsonl "${SOURCE_JSONL}" \
    --summary_out "${SUMMARY_JSON}" \
    --T 16 \
    --k 8 \
    --stride "${STRIDE}" \
    --max_demos_per_file "${MAX_DEMOS_PER_FILE}" \
    --camera_key agentview_rgb
}

prepare_splits_and_stats() {
  "${PY}" - <<'PY'
import json
import os
from pathlib import Path
import numpy as np

source = Path(os.environ["SOURCE_JSONL"])
split_dir = Path(os.environ["SPLIT_DIR"])
out_root = Path(os.environ["OUT_ROOT"])
gpus = [x for x in os.environ["GPUS"].split(",") if x.strip()]
world = len(gpus)
rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
if not rows:
    raise SystemExit(f"empty source jsonl: {source}")
chunks = [np.asarray(row["action_chunk"], dtype=np.float32) for row in rows]
actions = np.concatenate(chunks, axis=0)
mean = actions[:, :6].mean(axis=0).astype(np.float32)
std = np.maximum(actions[:, :6].std(axis=0), 1e-4).astype(np.float32)
pos_rate = np.asarray([(actions[:, 6] > 0.5).mean()], dtype=np.float32)
np.savez(out_root / "action_stats.npz", mean=mean, std=std, pos_rate=pos_rate)
split_dir.mkdir(parents=True, exist_ok=True)
for rank in range(world):
    path = split_dir / f"source_shard_{rank:02d}_of_{world:02d}.jsonl"
    with path.open("w") as f:
        for i, row in enumerate(rows):
            if i % world == rank:
                f.write(json.dumps(row, sort_keys=True) + "\n")
print(json.dumps({
    "rows": len(rows),
    "world": world,
    "action_stats": str(out_root / "action_stats.npz"),
    "split_dir": str(split_dir),
}, sort_keys=True))
PY
}

cache_shards() {
  IFS=',' read -r -a gpu_array <<< "${GPUS}"
  local world="${#gpu_array[@]}"
  for rank in "${!gpu_array[@]}"; do
    local gpu="${gpu_array[$rank]}"
    local shard_jsonl="${SPLIT_DIR}/source_shard_$(printf '%02d' "${rank}")_of_$(printf '%02d' "${world}").jsonl"
    local shard_out="${OUT_ROOT}/shard_$(printf '%02d' "${rank}")_of_$(printf '%02d' "${world}")"
    if [[ -s "${shard_out}/manifest.jsonl" && -f "${shard_out}/DONE" ]]; then
      echo "[cache] shard ${rank}/${world} already done"
      continue
    fi
    mkdir -p "${shard_out}"
    (
      CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" -m wm3d_v3.benchmarks.libero_expert_cache \
        --input_jsonl "${shard_jsonl}" \
        --out_dir "${shard_out}" \
        --max_windows "${MAX_WINDOWS_PER_SHARD}" \
        --T 16 \
        --token_grid 8 \
        --device cuda:0 \
        --qwen_device cuda:0 \
        --task_cache_dir "${TASK_CACHE_DIR}" \
        --action_stats "${OUT_ROOT}/action_stats.npz" \
        --include_lowdim \
        --action_history_len 16 \
        --log_every 32 \
        > "${OUT_ROOT}/cache_shard_${rank}.log" 2>&1
      touch "${shard_out}/DONE"
    ) &
  done
  wait
}

merge_manifests() {
  : > "${MANIFEST_OUT}.tmp"
  find "${OUT_ROOT}" -mindepth 2 -maxdepth 2 -path "*/manifest.jsonl" | sort | while read -r manifest; do
    cat "${manifest}" >> "${MANIFEST_OUT}.tmp"
  done
  mv "${MANIFEST_OUT}.tmp" "${MANIFEST_OUT}"
  "${PY}" - <<'PY'
import json
import os
from pathlib import Path
manifest = Path(os.environ["MANIFEST_OUT"])
rows = sum(1 for line in manifest.open() if line.strip())
summary = {
    "manifest": str(manifest),
    "rows": rows,
    "out_root": os.environ["OUT_ROOT"],
    "action_stats": str(Path(os.environ["OUT_ROOT"]) / "action_stats.npz"),
}
(Path(os.environ["OUT_ROOT"]) / "merged_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
print(json.dumps(summary, sort_keys=True))
PY
}

cmd="${1:-all}"
case "${cmd}" in
  export) export_windows ;;
  split) prepare_splits_and_stats ;;
  cache) cache_shards ;;
  merge) merge_manifests ;;
  all)
    export_windows
    prepare_splits_and_stats
    cache_shards
    merge_manifests
    ;;
  *)
    echo "usage: $0 [export|split|cache|merge|all]" >&2
    exit 2
    ;;
esac
