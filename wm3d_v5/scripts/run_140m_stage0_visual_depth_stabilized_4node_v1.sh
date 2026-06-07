#!/usr/bin/env bash
set -euo pipefail

cd /data/Minko/world_model/wm3d_v3

CFG="v3_p64_140m_stage0_visual_depth_stabilized_4node_v1.yaml" \
RUN_NAME="train_140m_stage0_visual_depth_stabilized_4node_v1" \
MASTER_PORT="${MASTER_PORT:-29740}" \
MANIFEST="manifests/oxe_droid20k_depthplus_world_v1.jsonl" \
ACTION_STATS="/data/Minko/datasets/cache/wm3d_v3/action_stats_oxe_droid20k_stage1_world_v1.npz" \
WORKER_HOSTS="${WORKER_HOSTS:-root@172.27.0.7 root@172.27.0.4 root@172.27.0.5}" \
CACHE_WORKER_HOSTS="${CACHE_WORKER_HOSTS:-root@172.27.0.4 root@172.27.0.5}" \
scripts/run_300m_stage_4node_v1.sh
