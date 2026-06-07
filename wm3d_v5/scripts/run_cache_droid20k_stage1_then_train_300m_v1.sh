#!/usr/bin/env bash
set -euo pipefail

cd /data/Minko/world_model/wm3d_v3

PYTHON="${PYTHON:-/data/Minko/.venvs/wm3d/bin/python}"
LOG_DIR="${LOG_DIR:-/data/Minko/logs}"
DROID_ROOT="${DROID_ROOT:-/data/Minko/datasets/hf_world_model_pretrain/lerobot__droid_1.0.1}"
CACHE_ROOT="${CACHE_ROOT:-/data/Minko/datasets/cache/wm3d_v3}"
SHARDS="${SHARDS:-8}"
MAX_EPISODES_PER_SHARD="${MAX_EPISODES_PER_SHARD:-2500}"
MIN_DROID_RECORDS="${MIN_DROID_RECORDS:-18000}"
FRAME_STRIDE="${FRAME_STRIDE:-3}"
MAX_FRAMES_PER_EPISODE="${MAX_FRAMES_PER_EPISODE:-240}"
MIN_FRAMES="${MIN_FRAMES:-72}"
BATCH_FRAMES="${BATCH_FRAMES:-16}"

mkdir -p "${LOG_DIR}" manifests
rm -f manifests/droid20k_stage1_shard*_v1.jsonl
rm -f manifests/oxe_droid20k_stage1_world_v1.jsonl
rm -f "${CACHE_ROOT}/action_stats_oxe_droid20k_stage1_world_v1.npz"

for shard in $(seq 0 $((SHARDS - 1))); do
  log="${LOG_DIR}/cache_droid20k_stage1_shard${shard}.log"
  pidfile="${LOG_DIR}/cache_droid20k_stage1_shard${shard}.pid"
  rm -f "${log}" "${pidfile}"
  nohup env \
    CUDA_VISIBLE_DEVICES="${shard}" \
    PYTHONUNBUFFERED=1 \
    "${PYTHON}" scripts/cache_lerobot_droid_wm3d.py \
      --root "${DROID_ROOT}" \
      --cache_root "${CACHE_ROOT}" \
      --out_manifest "manifests/droid20k_stage1_shard${shard}_v1.jsonl" \
      --max_files 1000 \
      --episode_start "${shard}" \
      --episode_stride "${SHARDS}" \
      --max_episodes "${MAX_EPISODES_PER_SHARD}" \
      --min_frames "${MIN_FRAMES}" \
      --max_frames_per_episode "${MAX_FRAMES_PER_EPISODE}" \
      --frame_stride "${FRAME_STRIDE}" \
      --batch_frames "${BATCH_FRAMES}" \
      > "${log}" 2>&1 &
  echo $! > "${pidfile}"
  echo "shard${shard}_pid=$(cat "${pidfile}") log=${log}"
done

watch_log="${LOG_DIR}/watch_droid20k_cache_then_train_stage1_300m_v1.log"
watch_pid="${LOG_DIR}/watch_droid20k_cache_then_train_stage1_300m_v1.pid"
rm -f "${watch_log}" "${watch_pid}"
nohup env \
  PYTHON="${PYTHON}" \
  MIN_DROID_RECORDS="${MIN_DROID_RECORDS}" \
  bash scripts/watch_droid20k_cache_then_train_stage1_300m_v1.sh \
  > "${LOG_DIR}/watch_droid20k_cache_then_train_stage1_300m_v1.nohup" 2>&1 &
echo $! > "${watch_pid}"
echo "watcher_pid=$(cat "${watch_pid}") log=${watch_log}"
