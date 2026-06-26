#!/usr/bin/env bash
# Wait until the LIBERO token cache (manifest + action_stats) is fully built,
# then auto-launch the 8-GPU head-only SFT. Launch this detached (setsid) so it
# survives SSH disconnects; training itself is setsid-launched by the run script.
set -euo pipefail
cd /data/Minko/world_model/wm3d_v5

MANIFEST="${MANIFEST:-/data/Minko/world_model/wm3d_v5/manifests/libero_action_policy_full_T16_k8_s4_v1.jsonl}"
STATS="${STATS:-/0604-10T-test/wm3d_v5/cache/libero_action_policy_full_T16_k8_s4_v1/action_stats.npz}"
LOG_DIR="${LOG_DIR:-/data/Minko/logs/wm3d_v5_p64_1b_libero_action_policy_v1}"
WLOG="${LOG_DIR}/watch_train_when_cache_ready.log"
mkdir -p "${LOG_DIR}"
log(){ echo "[$(date -Is)] $*" | tee -a "${WLOG}"; }

log "waiting for cache manifest=${MANIFEST}"
while [[ ! -s "${MANIFEST}" || ! -s "${STATS}" ]]; do sleep 30; done
log "cache ready: $(wc -l < "${MANIFEST}") windows -> launching 8-GPU head-only SFT"
bash scripts/run_v5_libero_action_policy_v1.sh train 2>&1 | tee -a "${WLOG}"
log "training launched; monitor ${LOG_DIR}/train_action_policy.log"
