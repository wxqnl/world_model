#!/usr/bin/env bash
set -euo pipefail

cd /data/Minko/world_model/wm3d_v3

RESUME_CKPT="${RESUME_CKPT:-results/wm3d_v3_p64_300m_stage0_visual_geom_oxe_droid20k_balanced_fromscratch_2node_v1/ckpt/best.pt}"

if [[ ! -f "${RESUME_CKPT}" ]]; then
  echo "missing Stage0 checkpoint: ${RESUME_CKPT}" >&2
  exit 1
fi

CFG="v3_p64_300m_stage1_world_oxe_droid20k_balanced_from_stage0_2node_v2.yaml" \
RUN_NAME="train_300m_stage1_world_oxe_droid20k_balanced_from_stage0_2node_v2" \
MASTER_PORT="${MASTER_PORT:-29561}" \
RESUME_CKPT="${RESUME_CKPT}" \
scripts/run_300m_stage_2node_v2.sh
