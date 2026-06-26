#!/usr/bin/env bash
# Wait until the NFS->local cache copy finishes, then (re)launch the 8-GPU head SFT
# against the LOCAL cache (fast file opens, no NFS rpc_wait stalls).
set -uo pipefail
LOGD=/data/Minko/logs/wm3d_v5_p64_1b_libero_action_policy_v1
LOC=/data/Minko/wm3d_v5_cache_local/libero_action_policy_full_T16_k8_s4_v1
MAN_LOC=/data/Minko/world_model/wm3d_v5/manifests/libero_action_policy_full_T16_k8_s4_v1_local.jsonl
CL=$LOGD/copy_cache_local.log
RLOG=$LOGD/restart_local.log
mkdir -p "$LOGD"
echo "[$(date -Is)] waiting for copy DONE_COPY in $CL" >> "$RLOG"
while ! grep -q DONE_COPY "$CL" 2>/dev/null; do sleep 20; done
n=$(find "$LOC" -name '*.npz' | wc -l)
echo "[$(date -Is)] copy done; local_npz=$n; clearing GPUs + launching local training" >> "$RLOG"
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do kill -9 "$p" 2>/dev/null; done
sleep 3
cd /data/Minko/world_model/wm3d_v5
ACTION_MANIFEST="$MAN_LOC" ACTION_CACHE_ROOT="$LOC" bash scripts/run_v5_libero_action_policy_v1.sh train >> "$RLOG" 2>&1
echo "[$(date -Is)] local training launched (see $LOGD/train_action_policy.log)" >> "$RLOG"
