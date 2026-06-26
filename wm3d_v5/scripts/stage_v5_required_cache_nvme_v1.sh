#!/usr/bin/env bash
set -euo pipefail

# Stage the complete currently-configured v5 Stage0 cache onto local NVMe.
# Run from node43: /data/Minko/world_model/wm3d_v5.

NODES=("node43:local" "node41:root@172.27.0.4" "node44:root@172.27.0.7" "node42:root@172.27.0.5")
NODE44="root@172.27.0.7"
V5_DIR="/data/Minko/world_model/wm3d_v5"
SRC_V3="/data/Minko/datasets/cache/wm3d_v3"
LOCAL_ROOT="/data/Minko/datasets/cache/wm3d_v5_stage0_sharded"
WINDOW_SUBDIR="vggt_window_geom_p64_T16_k8_s4_hw64_full_v1"
SHARD_DIR="${LOCAL_ROOT}/${WINDOW_SUBDIR}_shards"
SHARED_WINDOW="/0604-10T-test/wm3d_v5/cache/${WINDOW_SUBDIR}"
MANIFEST="${V5_DIR}/manifests/oxe_droid20k_depthplus_world_v1.jsonl"
SHARD_PLAN_DIR="${LOCAL_ROOT}/_node_shards"
LOG_DIR="/data/Minko/logs"
SSH_OPTS=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null)
PULL_SSH="ssh -i /root/.ssh/id_ed25519 -o IdentitiesOnly=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"

mkdir -p "$LOG_DIR"

run_local_bg() {
  local name="$1"; shift
  local log="${LOG_DIR}/stage_v5_nvme_${name}_20260608.log"
  setsid bash -lc "$*" > "$log" 2>&1 < /dev/null &
  echo $! > "${LOG_DIR}/stage_v5_nvme_${name}_20260608.pid"
  echo "${name}_pid=$(cat "${LOG_DIR}/stage_v5_nvme_${name}_20260608.pid")"
}

run_remote_bg() {
  local name="$1" host="$2"; shift 2
  local log="${LOG_DIR}/stage_v5_nvme_${name}_20260608.log"
  ssh "${SSH_OPTS[@]}" "$host" "mkdir -p '$LOG_DIR'; setsid bash -lc '$*' > '$log' 2>&1 < /dev/null & echo \$! > '${LOG_DIR}/stage_v5_nvme_${name}_20260608.pid'; echo ${name}_pid=\$(cat '${LOG_DIR}/stage_v5_nvme_${name}_20260608.pid')"
}

stop_old_low_priority_io() {
  ssh "${SSH_OPTS[@]}" "$NODE44" "
    for pf in /data/Minko/logs/archive_v3_cache_zstd_node44_20260608.pid /data/Minko/logs/prepare_v5_stage0_local_cache_node44_20260608.pid; do
      pid=\$(cat \"\$pf\" 2>/dev/null || true)
      if [ -n \"\$pid\" ]; then
        pkill -TERM -P \"\$pid\" 2>/dev/null || true
        kill -TERM \"\$pid\" 2>/dev/null || true
      fi
    done
    sleep 2
    for pf in /data/Minko/logs/archive_v3_cache_zstd_node44_20260608.pid /data/Minko/logs/prepare_v5_stage0_local_cache_node44_20260608.pid; do
      pid=\$(cat \"\$pf\" 2>/dev/null || true)
      if [ -n \"\$pid\" ]; then
        pkill -KILL -P \"\$pid\" 2>/dev/null || true
        kill -KILL \"\$pid\" 2>/dev/null || true
      fi
    done
  "
}

stop_stage_jobs() {
  local kill_cmd='
    for pf in /data/Minko/logs/stage_v5_nvme_*_20260608.pid; do
      [ -f "$pf" ] || continue
      pid=$(cat "$pf" 2>/dev/null || true)
      if [ -n "$pid" ]; then
        pkill -TERM -P "$pid" 2>/dev/null || true
        kill -TERM "$pid" 2>/dev/null || true
      fi
    done
    sleep 2
    for pf in /data/Minko/logs/stage_v5_nvme_*_20260608.pid; do
      [ -f "$pf" ] || continue
      pid=$(cat "$pf" 2>/dev/null || true)
      if [ -n "$pid" ]; then
        pkill -KILL -P "$pid" 2>/dev/null || true
        kill -KILL "$pid" 2>/dev/null || true
      fi
    done
  '
  bash -lc "$kill_cmd" || true
  ssh "${SSH_OPTS[@]}" root@172.27.0.4 "$kill_cmd" || true
  ssh "${SSH_OPTS[@]}" "$NODE44" "$kill_cmd" || true
  ssh "${SSH_OPTS[@]}" root@172.27.0.5 "$kill_cmd" || true
}

prepare_node_shard_plan() {
  chmod +x "${V5_DIR}/scripts/plan_v5_node_cache_shards_v1.py"
  rm -rf "${SHARD_PLAN_DIR}"
  mkdir -p "${SHARD_PLAN_DIR}"
  "${V5_DIR}/scripts/plan_v5_node_cache_shards_v1.py" \
    --manifest "${MANIFEST}" \
    --out "${SHARD_PLAN_DIR}" \
    --nodes node43,node41,node44,node42 \
    --T 16 --k 8 --stride 4
  for host in root@172.27.0.4 "$NODE44" root@172.27.0.5; do
    ssh "${SSH_OPTS[@]}" "$host" "rm -rf '${SHARD_PLAN_DIR}'; mkdir -p '${SHARD_PLAN_DIR}'"
    tar -C "${SHARD_PLAN_DIR}" -cf - . | ssh "${SSH_OPTS[@]}" "$host" "tar -C '${SHARD_PLAN_DIR}' -xf -"
  done
}

copy_builder_to_node() {
  local host="$1"
  if [ "$host" = "local" ]; then
    chmod +x "${V5_DIR}/scripts/build_window_geom_tar_shards_v1.py" "${V5_DIR}/scripts/plan_v5_node_cache_shards_v1.py"
  else
    scp "${V5_DIR}/scripts/build_window_geom_tar_shards_v1.py" "${host}:${V5_DIR}/scripts/build_window_geom_tar_shards_v1.py"
    scp "${V5_DIR}/scripts/plan_v5_node_cache_shards_v1.py" "${host}:${V5_DIR}/scripts/plan_v5_node_cache_shards_v1.py"
    ssh "${SSH_OPTS[@]}" "$host" "chmod +x '${V5_DIR}/scripts/build_window_geom_tar_shards_v1.py' '${V5_DIR}/scripts/plan_v5_node_cache_shards_v1.py'"
  fi
}

build_window_shards_to_node() {
  local name="$1" host="$2"
  copy_builder_to_node "$host"
  local node_manifest="${SHARD_PLAN_DIR}/${name}.manifest.jsonl"
  local cmd="set -euo pipefail; mkdir -p '${LOCAL_ROOT}'; rm -rf '${SHARD_DIR}'; '${V5_DIR}/scripts/build_window_geom_tar_shards_v1.py' --src '${SHARED_WINDOW}' --out '${SHARD_DIR}' --manifest '${node_manifest}' --T 16 --k 8 --stride 4 --num-shards 64 --jobs 4 --shard-strategy contiguous --overwrite --log-every 10000; echo window_shards_done \$(date)"
  if [ "$host" = "local" ]; then
    run_local_bg "${name}_build_window_shards" "$cmd"
  else
    run_remote_bg "${name}_build_window_shards" "$host" "$cmd"
  fi
}

stage_node44_base_links() {
  run_remote_bg "node44_base_links" "$NODE44" \
    "set -euo pipefail; rm -rf /data/Minko/datasets/cache/wm3d_v5_stage0_base /data/Minko/datasets/cache/wm3d_v5_stage0_local '${LOCAL_ROOT}/actions' '${LOCAL_ROOT}/rgb_256' '${LOCAL_ROOT}/vggt_geom' '${LOCAL_ROOT}/qwen_taskemb' '${LOCAL_ROOT}/action_stats_oxe_droid20k_stage1_world_v1.npz'; mkdir -p '${LOCAL_ROOT}'; ln -s '${SRC_V3}/actions' '${LOCAL_ROOT}/actions'; ln -s '${SRC_V3}/rgb_256' '${LOCAL_ROOT}/rgb_256'; ln -s '${SRC_V3}/vggt_geom' '${LOCAL_ROOT}/vggt_geom'; ln -s '${SRC_V3}/qwen_taskemb' '${LOCAL_ROOT}/qwen_taskemb'; ln -s '${SRC_V3}/action_stats_oxe_droid20k_stage1_world_v1.npz' '${LOCAL_ROOT}/action_stats_oxe_droid20k_stage1_world_v1.npz'; echo done"
}

stage_base_to_node() {
  local name="$1" host="$2"
  local list_path="${SHARD_PLAN_DIR}/${name}.base_files.null"
  local cmd="set -euo pipefail; rm -rf /data/Minko/datasets/cache/wm3d_v5_stage0_base /data/Minko/datasets/cache/wm3d_v5_stage0_local '${LOCAL_ROOT}/actions' '${LOCAL_ROOT}/rgb_256' '${LOCAL_ROOT}/vggt_geom' '${LOCAL_ROOT}/qwen_taskemb' '${LOCAL_ROOT}/action_stats_oxe_droid20k_stage1_world_v1.npz'; mkdir -p '${LOCAL_ROOT}'; ${PULL_SSH} ${NODE44} \"tar --null -C '${SRC_V3}' -cf - -T '${list_path}'\" | tar -C '${LOCAL_ROOT}' -xf -; echo base_done \$(date)"
  if [ "$host" = "local" ]; then
    run_local_bg "${name}_base" "$cmd"
  else
    run_remote_bg "${name}_base" "$host" "$cmd"
  fi
}

stage_shards_to_node() {
  local name="$1" host="$2"
  local cmd="set -euo pipefail; mkdir -p '${LOCAL_ROOT}'; rm -rf '${SHARD_DIR}'; ${PULL_SSH} ${NODE44} \"tar -C '${LOCAL_ROOT}' -cf - '${WINDOW_SUBDIR}_shards'\" | tar -C '${LOCAL_ROOT}' -xf -; echo shards_done \$(date)"
  if [ "$host" = "local" ]; then
    run_local_bg "${name}_window_shards" "$cmd"
  else
    run_remote_bg "${name}_window_shards" "$host" "$cmd"
  fi
}

case "${1:-start}" in
  start)
    stop_stage_jobs
    stop_old_low_priority_io
    prepare_node_shard_plan
    stage_node44_base_links
    stage_base_to_node node43 local
    stage_base_to_node node41 root@172.27.0.4
    stage_base_to_node node42 root@172.27.0.5
    build_window_shards_to_node node43 local
    build_window_shards_to_node node41 root@172.27.0.4
    build_window_shards_to_node node44 root@172.27.0.7
    build_window_shards_to_node node42 root@172.27.0.5
    ;;
  build-shards-local)
    prepare_node_shard_plan
    build_window_shards_to_node node43 local
    build_window_shards_to_node node41 root@172.27.0.4
    build_window_shards_to_node node44 root@172.27.0.7
    build_window_shards_to_node node42 root@172.27.0.5
    ;;
  distribute-shards)
    stage_shards_to_node node43 local
    stage_shards_to_node node41 root@172.27.0.4
    stage_shards_to_node node42 root@172.27.0.5
    ;;
  *)
    echo "usage: $0 [start|build-shards-local|distribute-shards]" >&2
    exit 2
    ;;
esac
