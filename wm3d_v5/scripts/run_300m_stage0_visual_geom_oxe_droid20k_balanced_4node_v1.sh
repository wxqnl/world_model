#!/usr/bin/env bash
set -euo pipefail

cd /data/Minko/world_model/wm3d_v3

CFG="v3_p64_300m_stage0_visual_geom_oxe_droid20k_balanced_fromscratch_2node_v1.yaml" \
RUN_NAME="train_300m_stage0_visual_geom_oxe_droid20k_balanced_fromscratch_4node_v1" \
MASTER_PORT="${MASTER_PORT:-29570}" \
WORKER_HOSTS="${WORKER_HOSTS:-root@172.27.0.7 root@172.27.0.4 root@172.27.0.5}" \
CACHE_WORKER_HOSTS="${CACHE_WORKER_HOSTS:-root@172.27.0.4 root@172.27.0.5}" \
scripts/run_300m_stage_4node_v1.sh
