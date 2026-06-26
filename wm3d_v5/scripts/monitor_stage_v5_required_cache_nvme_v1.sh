#!/usr/bin/env bash
set -euo pipefail

LOG_DIR="/data/Minko/logs"
nodes=(
  "node43 local"
  "node41 root@172.27.0.4"
  "node44 root@172.27.0.7"
  "node42 root@172.27.0.5"
)

check_host() {
  local name="$1" host="$2"
  local cmd='
set +e
echo disk
df -h /data | tail -1
echo jobs
for pf in /data/Minko/logs/stage_v5_nvme_*_20260608.pid; do
  [ -f "$pf" ] || continue
  pid=$(cat "$pf" 2>/dev/null || true)
  stem=$(basename "$pf" .pid)
  echo "$stem pid=$pid"
  if [ -n "$pid" ] && [ -d "/proc/$pid" ]; then
    ps -o pid,ppid,stat,etimes,wchan:24,cmd -p "$pid"
    pgrep -P "$pid" -a || true
  fi
  log="/data/Minko/logs/${stem}.log"
  tail -n 8 "$log" 2>/dev/null | tr "\r" "\n" | tail -n 8 || true
done
echo local_cache
root=/data/Minko/datasets/cache/wm3d_v5_stage0_sharded
for p in actions rgb_256 vggt_geom qwen_taskemb action_stats_oxe_droid20k_stage1_world_v1.npz vggt_window_geom_p64_T16_k8_s4_hw64_full_v1_shards/index.tsv; do
  [ -e "$root/$p" ] && echo "present $p" || echo "missing $p"
done
shard_root="$root/vggt_window_geom_p64_T16_k8_s4_hw64_full_v1_shards"
if [ -d "$shard_root" ]; then
  echo shard_summary
  [ -f "$shard_root/summary.txt" ] && cat "$shard_root/summary.txt" || true
  find "$shard_root" -maxdepth 1 -type f \( -name "window_geom_*.tar" -o -name "window_geom_*.tar.tmp" \) 2>/dev/null | wc -l | awk "{print \"shard_files=\" \$1}"
fi
echo gpu
nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader,nounits 2>/dev/null || true
'
  echo "===== $name ====="
  if [ "$host" = "local" ]; then
    bash -lc "$cmd"
  else
    ssh -n "$host" "$cmd"
  fi
}

for item in "${nodes[@]}"; do
  set -- $item
  check_host "$1" "$2"
done
