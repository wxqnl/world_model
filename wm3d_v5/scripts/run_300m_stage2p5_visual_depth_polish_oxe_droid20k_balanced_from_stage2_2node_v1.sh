#!/usr/bin/env bash
set -euo pipefail

cd /data/Minko/world_model/wm3d_v3

RESUME_CKPT="${RESUME_CKPT:-results/wm3d_v3_p64_300m_stage2_oxe_droid20k_balanced_joint_visual_proposer_from_stage1_2node_v2/ckpt/best.pt}"

if [[ ! -f "${RESUME_CKPT}" ]]; then
  echo "missing Stage2 checkpoint: ${RESUME_CKPT}" >&2
  exit 1
fi

CFG="v3_p64_300m_stage2p5_visual_depth_polish_oxe_droid20k_balanced_from_stage2_2node_v1.yaml" \
RUN_NAME="train_300m_stage2p5_visual_depth_polish_oxe_droid20k_balanced_from_stage2_2node_v1" \
MASTER_PORT="${MASTER_PORT:-29563}" \
RESUME_CKPT="${RESUME_CKPT}" \
scripts/run_300m_stage_2node_v2.sh
