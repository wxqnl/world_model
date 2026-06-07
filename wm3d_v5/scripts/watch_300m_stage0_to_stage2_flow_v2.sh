#!/usr/bin/env bash
set -euo pipefail

cd /data/Minko/world_model/wm3d_v3

PYTHON="${PYTHON:-/data/Minko/.venvs/wm3d/bin/python}"
LOG_DIR="${LOG_DIR:-/data/Minko/logs}"
WATCH_LOG="${WATCH_LOG:-${LOG_DIR}/watch_300m_stage0_to_stage2_flow_v2.log}"
WORKER_HOST="${WORKER_HOST:-root@172.27.0.7}"
RUN_GENERATION_CANARY="${RUN_GENERATION_CANARY:-0}"

STAGE0_RUN="train_300m_stage0_visual_geom_oxe_droid20k_balanced_fromscratch_2node_v1"
STAGE0_CFG="configs/v3_p64_300m_stage0_visual_geom_oxe_droid20k_balanced_fromscratch_2node_v1.yaml"
STAGE0_ROOT="results/wm3d_v3_p64_300m_stage0_visual_geom_oxe_droid20k_balanced_fromscratch_2node_v1"

STAGE1_RUN="train_300m_stage1_world_oxe_droid20k_balanced_from_stage0_2node_v2"
STAGE1_CFG="configs/v3_p64_300m_stage1_world_oxe_droid20k_balanced_from_stage0_2node_v2.yaml"
STAGE1_ROOT="results/wm3d_v3_p64_300m_stage1_world_oxe_droid20k_balanced_from_stage0_2node_v2"
STAGE1_SCRIPT="scripts/run_300m_stage1_world_oxe_droid20k_balanced_from_stage0_2node_v2.sh"

STAGE1P5_RUN="train_300m_stage1p5_hunyuan_bridge_oxe_droid20k_balanced_from_stage1_2node_v1"
STAGE1P5_CFG="configs/v3_p64_300m_stage1p5_hunyuan_bridge_oxe_droid20k_balanced_from_stage1_2node_v1.yaml"
STAGE1P5_ROOT="results/wm3d_v3_p64_300m_stage1p5_hunyuan_bridge_oxe_droid20k_balanced_from_stage1_2node_v1"
STAGE1P5_SCRIPT="scripts/run_300m_stage1p5_hunyuan_bridge_oxe_droid20k_balanced_from_stage1_2node_v1.sh"

STAGE2_RUN="train_300m_stage2_oxe_droid20k_balanced_joint_visual_proposer_from_stage1_2node_v2"
STAGE2_CFG="configs/v3_p64_300m_stage2_oxe_droid20k_balanced_joint_visual_proposer_from_stage1_2node_v2.yaml"
STAGE2_ROOT="results/wm3d_v3_p64_300m_stage2_oxe_droid20k_balanced_joint_visual_proposer_from_stage1_2node_v2"
STAGE2_SCRIPT="scripts/run_300m_stage2_oxe_droid20k_balanced_joint_visual_proposer_from_stage1_2node_v2.sh"

mkdir -p "${LOG_DIR}"

log() {
  echo "[$(date -Is)] $*" | tee -a "${WATCH_LOG}"
}

pid_alive() {
  local pidfile="$1"
  [[ -f "${pidfile}" ]] || return 1
  local pid
  pid="$(cat "${pidfile}" 2>/dev/null || true)"
  [[ -n "${pid}" ]] || return 1
  ps -p "${pid}" >/dev/null 2>&1
}

worker_pid_alive() {
  local run="$1"
  ssh -o BatchMode=yes -o ConnectTimeout=8 "${WORKER_HOST}" \
    "test -f ${LOG_DIR}/${run}_node1.pid && ps -p \$(cat ${LOG_DIR}/${run}_node1.pid) >/dev/null 2>&1"
}

wait_stage() {
  local run="$1"
  log "wait_${run}"
  while pid_alive "${LOG_DIR}/${run}_node0.pid"; do
    grep "\[rank0\] step" "${LOG_DIR}/${run}_node0.log" 2>/dev/null | tail -n 1 | tee -a "${WATCH_LOG}" || true
    sleep 300
  done
  if worker_pid_alive "${run}"; then
    log "waiting_${run}_node1"
    while worker_pid_alive "${run}"; do
      sleep 120
    done
  fi
  grep -E "Traceback|RuntimeError|CUDA out|NCCL WARN|NCCL ERROR|Error|Exception" \
    "${LOG_DIR}/${run}_node0.log" | tail -n 60 >> "${WATCH_LOG}" || true
  log "${run}_all_exited"
}

best_or_latest() {
  local root="$1"
  if [[ -f "${root}/ckpt/best.pt" ]]; then
    echo "${root}/ckpt/best.pt"
  elif [[ -f "${root}/ckpt/latest.pt" ]]; then
    echo "${root}/ckpt/latest.pt"
  else
    return 1
  fi
}

eval_stage() {
  local label="$1"
  local cfg="$2"
  local ckpt="$3"
  local root="$4"
  local out_dir="${root}/basic_eval_after_${label}"
  mkdir -p "${out_dir}"

  log "start_${label}_basic_eval ckpt=${ckpt}"
  CUDA_VISIBLE_DEVICES=0 "${PYTHON}" -m wm3d_v3.eval.run_eval \
    --cfg "${cfg}" \
    --ckpt "${ckpt}" \
    --out "${out_dir}/eval_rgb_depth_64b.json" \
    --max_batches 64 \
    --batch_size 4 \
    > "${LOG_DIR}/${label}_basic_eval_300m_v2.log" 2>&1
  log "done_${label}_basic_eval out=${out_dir}/eval_rgb_depth_64b.json"

  log "start_${label}_native3d_benchmark"
  GPU=0 CFG="${cfg}" CKPT="${ckpt}" OUT_DIR="${out_dir}/native3d_benchmark_v2" \
    MAX_BATCHES_PER_DATASET=4 N_VIZ=6 \
    bash scripts/run_world3d_native_benchmark_v2.sh \
    > "${LOG_DIR}/${label}_native3d_benchmark_300m_v2.log" 2>&1 \
    || log "${label}_native3d_benchmark_failed log=${LOG_DIR}/${label}_native3d_benchmark_300m_v2.log"
  log "done_${label}_native3d_benchmark"

  if [[ "${RUN_GENERATION_CANARY}" == "1" ]]; then
    log "start_${label}_generation_canary"
    CUDA_VISIBLE_DEVICES=0 \
      CFG="${cfg}" \
      CKPT="${ckpt}" \
      OUT_DIR="${out_dir}/generation_canary" \
      MAX_BATCHES=16 \
      BATCH_SIZE=1 \
      N_GIFS=3 \
      N_HUNYUAN_GIFS=2 \
      bash scripts/run_generation_canary_v1.sh \
      > "${LOG_DIR}/${label}_generation_canary_300m_v2.log" 2>&1 \
      || log "${label}_generation_canary_failed log=${LOG_DIR}/${label}_generation_canary_300m_v2.log"
    log "done_${label}_generation_canary"
  else
    log "skip_${label}_generation_canary RUN_GENERATION_CANARY=0"
  fi
}

log "watch_start"

wait_stage "${STAGE0_RUN}"
STAGE0_CKPT="$(best_or_latest "${STAGE0_ROOT}")" || { log "missing_stage0_checkpoint"; exit 2; }
eval_stage "stage0" "${STAGE0_CFG}" "${STAGE0_CKPT}" "${STAGE0_ROOT}"

if ! pid_alive "${LOG_DIR}/${STAGE1_RUN}_node0.pid"; then
  log "start_stage1 resume=${STAGE0_CKPT}"
  RESUME_CKPT="${STAGE0_CKPT}" bash "${STAGE1_SCRIPT}" >> "${WATCH_LOG}" 2>&1
fi
wait_stage "${STAGE1_RUN}"
STAGE1_CKPT="$(best_or_latest "${STAGE1_ROOT}")" || { log "missing_stage1_checkpoint"; exit 3; }
eval_stage "stage1" "${STAGE1_CFG}" "${STAGE1_CKPT}" "${STAGE1_ROOT}"

if ! pid_alive "${LOG_DIR}/${STAGE1P5_RUN}_node0.pid"; then
  log "start_stage1p5_hunyuan_bridge resume=${STAGE1_CKPT}"
  RESUME_CKPT="${STAGE1_CKPT}" bash "${STAGE1P5_SCRIPT}" >> "${WATCH_LOG}" 2>&1
fi
wait_stage "${STAGE1P5_RUN}"
STAGE1P5_CKPT="$(best_or_latest "${STAGE1P5_ROOT}")" || { log "missing_stage1p5_checkpoint"; exit 4; }
eval_stage "stage1p5" "${STAGE1P5_CFG}" "${STAGE1P5_CKPT}" "${STAGE1P5_ROOT}"

if ! pid_alive "${LOG_DIR}/${STAGE2_RUN}_node0.pid"; then
  log "start_stage2_progress_proposer resume=${STAGE1P5_CKPT}"
  RESUME_CKPT="${STAGE1P5_CKPT}" bash "${STAGE2_SCRIPT}" >> "${WATCH_LOG}" 2>&1
fi
wait_stage "${STAGE2_RUN}"
STAGE2_CKPT="$(best_or_latest "${STAGE2_ROOT}")" || { log "missing_stage2_checkpoint"; exit 5; }
eval_stage "stage2" "${STAGE2_CFG}" "${STAGE2_CKPT}" "${STAGE2_ROOT}"

# ---- Stage3 (optional): text->video generation (Hunyuan DiT-control) on the stage2 world ----
# Gated OFF by default so the canonical stage0->stage2 flow is unchanged.
# Enable with RUN_STAGE3_GENERATION=1. Re-point to a native3d ckpt by editing WM_CFG/WM_CKPT below.
RUN_STAGE3_GENERATION="${RUN_STAGE3_GENERATION:-0}"
if [[ "${RUN_STAGE3_GENERATION}" == "1" ]]; then
  STAGE3_LAUNCHER="/data/Minko/world_model/wm3d_v5/scripts/run_stage3_generation_hunyuan_dit_control_v1.sh"
  STAGE3_OUT="${STAGE3_OUT:-/data/Minko/world_model/wm3d_v5/results/wm3d_stage3_generation_hunyuan_dit_control_from_stage2_v1}"
  STAGE2_CFG_ABS="/data/Minko/world_model/wm3d_v3/${STAGE2_CFG}"
  STAGE2_CKPT_ABS="/data/Minko/world_model/wm3d_v3/${STAGE2_CKPT}"
  log "start_stage3_generation wm_ckpt=${STAGE2_CKPT_ABS} out=${STAGE3_OUT}"
  WM_CFG="${STAGE2_CFG_ABS}" WM_CKPT="${STAGE2_CKPT_ABS}" OUT_DIR="${STAGE3_OUT}" \
    bash "${STAGE3_LAUNCHER}" >> "${WATCH_LOG}" 2>&1 \
    || log "stage3_generation_failed (see ${WATCH_LOG})"
  log "done_stage3_generation out=${STAGE3_OUT}"
else
  log "skip_stage3_generation RUN_STAGE3_GENERATION=0 (enable=1, or run scripts/run_stage3_generation_hunyuan_dit_control_v1.sh standalone)"
fi

log "flow_done stage2_ckpt=${STAGE2_CKPT}; stage2 is progress+proposer scaffold; inspect fixed GIF/depth before optional Stage2.5"
