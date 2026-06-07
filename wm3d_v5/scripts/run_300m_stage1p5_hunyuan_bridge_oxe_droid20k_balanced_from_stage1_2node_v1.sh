#!/usr/bin/env bash
set -euo pipefail

cd /data/Minko/world_model/wm3d_v3

STAGE1_ROOT="${STAGE1_ROOT:-results/wm3d_v3_p64_300m_stage1_world_oxe_droid20k_balanced_from_stage0_2node_v2}"

if [[ -z "${RESUME_CKPT:-}" ]]; then
  if [[ -f "${STAGE1_ROOT}/ckpt/best.pt" ]]; then
    RESUME_CKPT="${STAGE1_ROOT}/ckpt/best.pt"
  elif [[ -f "${STAGE1_ROOT}/ckpt/latest.pt" ]]; then
    RESUME_CKPT="${STAGE1_ROOT}/ckpt/latest.pt"
  else
    echo "missing Stage1 checkpoint for Run1.5 Hunyuan bridge: expected ${STAGE1_ROOT}/ckpt/best.pt or ${STAGE1_ROOT}/ckpt/latest.pt; set RESUME_CKPT to override" >&2
    exit 1
  fi
fi

if [[ ! -f "${RESUME_CKPT}" ]]; then
  echo "missing explicit RESUME_CKPT for Run1.5 Hunyuan bridge: ${RESUME_CKPT}" >&2
  exit 1
fi

CFG="v3_p64_300m_stage1p5_hunyuan_bridge_oxe_droid20k_balanced_from_stage1_2node_v1.yaml" \
RUN_NAME="train_300m_stage1p5_hunyuan_bridge_oxe_droid20k_balanced_from_stage1_2node_v1" \
MASTER_PORT="${MASTER_PORT:-29565}" \
RESUME_CKPT="${RESUME_CKPT}" \
scripts/run_300m_stage_2node_v2.sh
