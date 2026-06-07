#!/usr/bin/env bash
set -euo pipefail

cd /data/Minko/world_model/wm3d_v3

WORKER_HOST="${WORKER_HOST:-root@172.27.0.7}"
CACHE_ROOT="${CACHE_ROOT:-/data/Minko/datasets/cache/wm3d_v3}"
ACTION_STATS="${ACTION_STATS:-/data/Minko/datasets/cache/wm3d_v3/action_stats_oxe_droid20k_stage1_world_v1.npz}"
MANIFEST="${MANIFEST:-manifests/oxe_droid20k_balanced_world_v2.jsonl}"

if [[ ! -f "${MANIFEST}" ]]; then
  echo "missing manifest: ${MANIFEST}" >&2
  exit 1
fi
if [[ ! -f "${ACTION_STATS}" ]]; then
  echo "missing action stats: ${ACTION_STATS}" >&2
  exit 1
fi

rsync -a "${MANIFEST}" "${WORKER_HOST}:/data/Minko/world_model/wm3d_v3/${MANIFEST}"
rsync -a "${ACTION_STATS}" "${WORKER_HOST}:${ACTION_STATS}"

mapfile -t DATASETS < <(/data/Minko/.venvs/wm3d/bin/python - "${MANIFEST}" <<'PY'
import json
import sys

seen = set()
with open(sys.argv[1]) as f:
    for line in f:
        if not line.strip():
            continue
        dataset = json.loads(line)["dataset"]
        if dataset not in seen:
            seen.add(dataset)
            print(dataset)
PY
)

SUBDIRS=(vggt_pooled vggt_geom rgb_256 actions qwen_taskemb)
for sub in "${SUBDIRS[@]}"; do
  ssh "${WORKER_HOST}" "mkdir -p ${CACHE_ROOT}/${sub}"
  for dataset in "${DATASETS[@]}"; do
    rsync -a --include="${dataset}__*" --exclude='*' \
      "${CACHE_ROOT}/${sub}/" "${WORKER_HOST}:${CACHE_ROOT}/${sub}/"
  done
done

echo "synced manifest, action stats, and cache prefixes to ${WORKER_HOST}: ${DATASETS[*]}"
