#!/usr/bin/env bash
set -euo pipefail

cd /data/Minko/world_model/wm3d_v3

LOG="/data/Minko/logs/watch_droid20k_cache_then_train_stage1_300m_v1.log"
PYTHON="${PYTHON:-/data/Minko/.venvs/wm3d/bin/python}"
MIN_DROID_RECORDS="${MIN_DROID_RECORDS:-18000}"
{
  echo "[watch] started $(date -Is)"
  echo "[watch] waiting for cache shard pids"
} >> "${LOG}"

while true; do
  alive=0
  for pidfile in /data/Minko/logs/cache_droid20k_stage1_shard*.pid; do
    [ -f "${pidfile}" ] || continue
    pid="$(cat "${pidfile}")"
    if ps -p "${pid}" >/dev/null 2>&1; then
      alive=$((alive + 1))
    fi
  done
  ready_records=$(cat manifests/droid20k_stage1_shard*_v1.jsonl 2>/dev/null | wc -l || true)
  echo "[watch] $(date -Is) alive=${alive} droid_records_so_far=${ready_records}" >> "${LOG}"
  if [ "${alive}" -eq 0 ]; then
    break
  fi
  sleep 300
done

echo "[watch] cache jobs complete; building mixed manifest $(date -Is)" >> "${LOG}"
"${PYTHON}" scripts/build_stage1_oxe_droid_manifest.py \
  --oxe_manifest manifests/oxe_all_trainable_cached_rgb_geom_v1.jsonl \
  --droid_manifest_glob 'manifests/droid20k_stage1_shard*_v1.jsonl' \
  --cache_root /data/Minko/datasets/cache/wm3d_v3 \
  --out_manifest manifests/oxe_droid20k_stage1_world_v1.jsonl \
  --out_action_stats /data/Minko/datasets/cache/wm3d_v3/action_stats_oxe_droid20k_stage1_world_v1.npz \
  --require_task_emb \
  --min_droid_records "${MIN_DROID_RECORDS}" \
  >> "${LOG}" 2>&1

echo "[watch] launching training $(date -Is)" >> "${LOG}"
bash scripts/run_300m_stage1_oxe_droid20k_fromscratch_2node_v1.sh >> "${LOG}" 2>&1
echo "[watch] training launch returned $(date -Is)" >> "${LOG}"
