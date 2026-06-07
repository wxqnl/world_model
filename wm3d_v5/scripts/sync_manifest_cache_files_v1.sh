#!/usr/bin/env bash
set -euo pipefail

cd /data/Minko/world_model/wm3d_v3

WORKER_HOST="${WORKER_HOST:?WORKER_HOST is required}"
MANIFEST="${MANIFEST:-manifests/oxe_droid20k_balanced_world_v2.jsonl}"
CACHE_ROOT="${CACHE_ROOT:-/data/Minko/datasets/cache/wm3d_v3}"
ACTION_STATS="${ACTION_STATS:-/data/Minko/datasets/cache/wm3d_v3/action_stats_oxe_droid20k_stage1_world_v1.npz}"
PYTHON="${PYTHON:-/data/Minko/.venvs/wm3d/bin/python}"
BW_LIMIT="${BW_LIMIT:-0}"
LIST_DIR="${LIST_DIR:-/data/Minko/logs/cache_sync_lists}"

mkdir -p "${LIST_DIR}"
if [[ ! -f "${MANIFEST}" ]]; then
  echo "missing manifest: ${MANIFEST}" >&2
  exit 1
fi

host_tag="$(echo "${WORKER_HOST}" | tr '@/:' '___')"
file_list="${LIST_DIR}/cache_files_${host_tag}_$(basename "${MANIFEST}").txt"

"${PYTHON}" - "${MANIFEST}" > "${file_list}" <<'PY'
import json
import sys
from pathlib import Path

manifest = Path(sys.argv[1])
seen = set()
for line in manifest.open():
    if not line.strip():
        continue
    rec = json.loads(line)
    safe = rec["clip_id"].replace("/", "__")
    seen.add(safe)

for safe in sorted(seen):
    print(f"vggt_pooled/{safe}.npy")
    print(f"vggt_geom/{safe}.npz")
    print(f"rgb_256/{safe}.npy")
    print(f"actions/{safe}.npy")
    print(f"qwen_taskemb/{safe}.npy")
PY

rsync_args=(-a --ignore-missing-args --files-from="${file_list}")
if [[ "${BW_LIMIT}" != "0" ]]; then
  rsync_args+=(--bwlimit="${BW_LIMIT}")
fi

ssh "${WORKER_HOST}" "mkdir -p ${CACHE_ROOT} /data/Minko/logs /data/Minko/world_model/wm3d_v3/manifests"
rsync -a "${MANIFEST}" "${WORKER_HOST}:/data/Minko/world_model/wm3d_v3/${MANIFEST}"
rsync -a "${ACTION_STATS}" "${WORKER_HOST}:${ACTION_STATS}"
rsync "${rsync_args[@]}" "${CACHE_ROOT}/" "${WORKER_HOST}:${CACHE_ROOT}/"

echo "synced $(wc -l < "${file_list}") cache file entries to ${WORKER_HOST}"
echo "file_list=${file_list}"
