#!/usr/bin/env bash
set -euo pipefail

cd /data/Minko/world_model/wm3d_v3

STAGE1P5_ROOT="${STAGE1P5_ROOT:-results/wm3d_v3_p64_300m_stage1p5_hunyuan_bridge_oxe_droid20k_balanced_from_stage1_2node_v1}"

if [[ -z "${RESUME_CKPT:-}" ]]; then
  if [[ -f "${STAGE1P5_ROOT}/ckpt/best.pt" ]]; then
    RESUME_CKPT="${STAGE1P5_ROOT}/ckpt/best.pt"
  elif [[ -f "${STAGE1P5_ROOT}/ckpt/latest.pt" ]]; then
    RESUME_CKPT="${STAGE1P5_ROOT}/ckpt/latest.pt"
  else
    echo "missing Stage1.5 checkpoint for Stage2 progress+proposer scaffold: expected ${STAGE1P5_ROOT}/ckpt/best.pt or ${STAGE1P5_ROOT}/ckpt/latest.pt; set RESUME_CKPT to override" >&2
    exit 1
  fi
fi

if [[ ! -f "${RESUME_CKPT}" ]]; then
  echo "missing explicit RESUME_CKPT for Stage2 progress+proposer scaffold: ${RESUME_CKPT}" >&2
  exit 1
fi

CFG="v3_p64_300m_stage2_oxe_droid20k_balanced_joint_visual_proposer_from_stage1_2node_v2.yaml" \
RUN_NAME="train_300m_stage2_oxe_droid20k_balanced_joint_visual_proposer_from_stage1_2node_v2" \
MASTER_PORT="${MASTER_PORT:-29562}" \
RESUME_CKPT="${RESUME_CKPT}" \
scripts/run_300m_stage_2node_v2.sh
