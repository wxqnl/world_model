#!/usr/bin/env bash
set -euo pipefail

script="/data/Minko/world_model/wm3d_v5/scripts/prepare_v5_stage0_local_cache_node_v1.sh"
logs="/data/Minko/logs"

launch_local() {
  local node="$1"
  mkdir -p "$logs"
  setsid "$script" "$node" > "${logs}/prepare_v5_stage0_local_cache_${node}_20260608.log" 2>&1 < /dev/null &
  echo $! > "${logs}/prepare_v5_stage0_local_cache_${node}_20260608.pid"
  echo "${node}_pid=$(cat "${logs}/prepare_v5_stage0_local_cache_${node}_20260608.pid")"
}

launch_remote() {
  local node="$1"
  local host="$2"
  scp "$script" "${host}:${script}"
  ssh "$host" "chmod +x '$script'; mkdir -p '$logs'; setsid '$script' '$node' > '${logs}/prepare_v5_stage0_local_cache_${node}_20260608.log' 2>&1 < /dev/null & echo \$! > '${logs}/prepare_v5_stage0_local_cache_${node}_20260608.pid'; echo ${node}_pid=\$(cat '${logs}/prepare_v5_stage0_local_cache_${node}_20260608.pid')"
}

launch_local node43
launch_remote node44 root@172.27.0.7
launch_remote node41 root@172.27.0.4
launch_remote node42 root@172.27.0.5
