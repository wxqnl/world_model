#!/usr/bin/env bash
set -euo pipefail

cd /data/Minko/world_model/wm3d_v3

CFG="v3_p64_300m_stage0_visual_geom_oxe_droid20k_balanced_fromscratch_2node_v1.yaml" \
RUN_NAME="train_300m_stage0_visual_geom_oxe_droid20k_balanced_fromscratch_2node_v1" \
MASTER_PORT="${MASTER_PORT:-29560}" \
scripts/run_300m_stage_2node_v2.sh
