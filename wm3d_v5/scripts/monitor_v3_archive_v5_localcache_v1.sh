#!/usr/bin/env bash
set -euo pipefail

nodes=(
  "node43 local delete_duplicate_v3_cache_node43_20260608"
  "node41 root@172.27.0.4 delete_duplicate_v3_cache_node41_20260608"
  "node44 root@172.27.0.7 archive_v3_cache_zstd_node44_20260608"
  "node42 root@172.27.0.5 delete_duplicate_v3_cache_node42_20260608"
)

check_node() {
  local name="$1"
  local host="$2"
  local stem="$3"
  local cmd='
set -euo pipefail
pidfile="/data/Minko/logs/'"$stem"'.pid"
log="/data/Minko/logs/'"$stem"'.log"
marker_glob="/data/Minko/datasets/cache/*MOVED*"
pid="$(cat "$pidfile" 2>/dev/null || true)"
prep_log="/data/Minko/logs/prepare_v5_stage0_local_cache_'"$name"'_20260608.log"
prep_pidfile="/data/Minko/logs/prepare_v5_stage0_local_cache_'"$name"'_20260608.pid"
prep_pid="$(cat "$prep_pidfile" 2>/dev/null || true)"
child=""
if [ -n "$pid" ]; then
  child="$(pgrep -P "$pid" 2>/dev/null || true)"
fi
echo "pid=$pid children=$(echo "$child" | tr "\n" " ")"
for p in $pid $child; do
  if [ -n "$p" ] && [ -d "/proc/$p" ]; then
    ps -o pid,ppid,stat,etimes,wchan:24,cmd -p "$p"
    grep -E "rchar|wchar|read_bytes|write_bytes" "/proc/$p/io" || true
  fi
done
echo "old_tar_left:"
pgrep -af "datasets_cache_wm3d_v3.tar.partial wm3d_v3" | grep -v "pgrep -af" || true
echo "v3_source:"
if [ -d /data/Minko/datasets/cache/wm3d_v3 ]; then
  echo "wm3d_v3_source_present"
else
  echo "wm3d_v3_source_absent"
fi
echo "v5_local_cache:"
if [ -d /data/Minko/datasets/cache/wm3d_v5_stage0_local/vggt_window_geom_p64_T16_k8_s4_hw64_full_v1 ]; then
  echo "window_local_dir_present"
else
  echo "window_local_dir_absent"
fi
echo "prepare:"
echo "prep_pid=$prep_pid"
if [ -n "$prep_pid" ] && [ -d "/proc/$prep_pid" ]; then
  ps -o pid,ppid,stat,etimes,wchan:24,cmd -p "$prep_pid"
  pgrep -P "$prep_pid" -a || true
fi
tail -n 20 "$prep_log" 2>/dev/null | tr "\r" "\n" | tail -n 8 || true
if ls $marker_glob >/dev/null 2>&1; then
  echo "markers:"
  ls $marker_glob
fi
echo "log_tail:"
tail -n 20 "$log" 2>/dev/null | tr "\r" "\n" | tail -n 12 || true
echo "disk:"
timeout 8 df -h /data /0604-10T-test || true
'
  echo "===== ${name} ====="
  if [ "$host" = "local" ]; then
    bash -lc "$cmd"
  else
    ssh "$host" "$cmd"
  fi
}

for item in "${nodes[@]}"; do
  # shellcheck disable=SC2086
  set -- $item
  check_node "$1" "$2" "$3"
done
