#!/usr/bin/env bash
set -euo pipefail

cd /data/Minko/world_model/wm3d_v3

PYTHON="${PYTHON:-/data/Minko/.venvs/wm3d/bin/python}"
LOG_DIR="${LOG_DIR:-/data/Minko/logs}"
WORKER_HOST="${WORKER_HOST:-root@172.27.0.7}"
SYNC_CACHE="${SYNC_CACHE:-0}"
RUN_NATIVE3D="${RUN_NATIVE3D:-1}"

WATCH_LOG="${WATCH_LOG:-${LOG_DIR}/watch_140m_mini10_stage0_to_stage2_flow_v1.log}"

STAGE0_RUN="train_140m_stage0_visual_depth_mini10_2node_v1"
STAGE0_CFG="configs/v3_p64_140m_stage0_visual_depth_mini10_2node_v1.yaml"
STAGE0_ROOT="results/wm3d_v3_p64_140m_stage0_visual_depth_mini10_2node_v1"

STAGE1_RUN="train_140m_stage1_dynamics_visual_replay_mini10_2node_v1"
STAGE1_CFG="configs/v3_p64_140m_stage1_dynamics_visual_replay_mini10_2node_v1.yaml"
STAGE1_ROOT="results/wm3d_v3_p64_140m_stage1_dynamics_visual_replay_mini10_2node_v1"

STAGE1P5_RUN="train_140m_stage1p5_hunyuan_bridge_mini10_2node_v1"
STAGE1P5_CFG="configs/v3_p64_140m_stage1p5_hunyuan_bridge_mini10_2node_v1.yaml"
STAGE1P5_ROOT="results/wm3d_v3_p64_140m_stage1p5_hunyuan_bridge_mini10_2node_v1"

STAGE2_RUN="train_140m_stage2_action_scaffold_mini10_2node_v1"
STAGE2_CFG="configs/v3_p64_140m_stage2_action_scaffold_mini10_2node_v1.yaml"
STAGE2_ROOT="results/wm3d_v3_p64_140m_stage2_action_scaffold_mini10_2node_v1"

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

launch_stage() {
  local run="$1"
  local cfg="$2"
  local port="$3"
  local resume="${4:-}"
  log "launch_${run} cfg=${cfg} resume=${resume:-from_scratch}"
  local cfg_name
  cfg_name="$(basename "${cfg}")"
  if [[ -n "${resume}" ]]; then
    SYNC_CACHE="${SYNC_CACHE}" CFG="${cfg_name}" RUN_NAME="${run}" MASTER_PORT="${port}" RESUME_CKPT="${resume}" \
      bash scripts/run_300m_stage_2node_v2.sh >> "${WATCH_LOG}" 2>&1
  else
    SYNC_CACHE="${SYNC_CACHE}" CFG="${cfg_name}" RUN_NAME="${run}" MASTER_PORT="${port}" \
      bash scripts/run_300m_stage_2node_v2.sh >> "${WATCH_LOG}" 2>&1
  fi
}

wait_stage() {
  local run="$1"
  log "wait_${run}"
  while pid_alive "${LOG_DIR}/${run}_node0.pid"; do
    grep "\[rank0\] step" "${LOG_DIR}/${run}_node0.log" 2>/dev/null | tail -n 1 | tee -a "${WATCH_LOG}" || true
    sleep 30
  done
  if worker_pid_alive "${run}"; then
    log "waiting_${run}_node1"
    while worker_pid_alive "${run}"; do
      sleep 30
    done
  fi
  grep -E "Traceback|RuntimeError|CUDA out|NCCL WARN|NCCL ERROR|Error|Exception|non-finite|NaN" \
    "${LOG_DIR}/${run}_node0.log" | tail -n 80 >> "${WATCH_LOG}" || true
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
  local make_hunyuan="${5:-0}"
  local out_dir="${root}/mini10_eval_after_${label}"
  mkdir -p "${out_dir}"

  log "eval_${label}_basic ckpt=${ckpt}"
  CUDA_VISIBLE_DEVICES=0 "${PYTHON}" -m wm3d_v3.eval.run_eval \
    --cfg "${cfg}" \
    --ckpt "${ckpt}" \
    --out "${out_dir}/eval_rgb_depth_24b.json" \
    --max_batches 24 \
    --batch_size 4 \
    > "${LOG_DIR}/${label}_mini10_basic_eval.log" 2>&1

  log "eval_${label}_demo_gif"
  CUDA_VISIBLE_DEVICES=0 "${PYTHON}" -m wm3d_v3.eval.make_demo_gif \
    --cfg "${cfg}" \
    --ckpt "${ckpt}" \
    --out_dir "${out_dir}/demo_gifs" \
    --n_clips 4 \
    > "${LOG_DIR}/${label}_mini10_demo_gif.log" 2>&1 || log "${label}_demo_gif_failed"

  if [[ "${RUN_NATIVE3D}" == "1" ]]; then
    log "eval_${label}_native3d"
    GPU=0 CFG="${cfg}" CKPT="${ckpt}" OUT_DIR="${out_dir}/native3d_benchmark_v2" \
      MAX_BATCHES_PER_DATASET=1 N_VIZ=4 \
      bash scripts/run_world3d_native_benchmark_v2.sh \
      > "${LOG_DIR}/${label}_mini10_native3d.log" 2>&1 || log "${label}_native3d_failed"
  fi

  if [[ "${make_hunyuan}" == "1" ]]; then
    log "eval_${label}_hunyuan_latent_demo"
    CUDA_VISIBLE_DEVICES=0 "${PYTHON}" -m wm3d_v3.eval.make_hunyuan_latent_demo \
      --cfg "${cfg}" \
      --ckpt "${ckpt}" \
      --out_dir "${out_dir}/hunyuan_latent_demos" \
      --n_clips 2 \
      > "${LOG_DIR}/${label}_mini10_hunyuan_latent_demo.log" 2>&1 || log "${label}_hunyuan_latent_demo_failed"
  fi

  find "${out_dir}" -maxdepth 4 -type f | sort > "${out_dir}/artifacts.txt"
  log "eval_${label}_done out=${out_dir}"
}

log "flow_start 140m mini10 stage0->stage2"

launch_stage "${STAGE0_RUN}" "${STAGE0_CFG}" 29610
wait_stage "${STAGE0_RUN}"
STAGE0_CKPT="$(best_or_latest "${STAGE0_ROOT}")" || { log "missing_stage0_ckpt"; exit 2; }
eval_stage "stage0" "${STAGE0_CFG}" "${STAGE0_CKPT}" "${STAGE0_ROOT}" 0

launch_stage "${STAGE1_RUN}" "${STAGE1_CFG}" 29611 "${STAGE0_CKPT}"
wait_stage "${STAGE1_RUN}"
STAGE1_CKPT="$(best_or_latest "${STAGE1_ROOT}")" || { log "missing_stage1_ckpt"; exit 3; }
eval_stage "stage1" "${STAGE1_CFG}" "${STAGE1_CKPT}" "${STAGE1_ROOT}" 0

launch_stage "${STAGE1P5_RUN}" "${STAGE1P5_CFG}" 29612 "${STAGE1_CKPT}"
wait_stage "${STAGE1P5_RUN}"
STAGE1P5_CKPT="$(best_or_latest "${STAGE1P5_ROOT}")" || { log "missing_stage1p5_ckpt"; exit 4; }
eval_stage "stage1p5" "${STAGE1P5_CFG}" "${STAGE1P5_CKPT}" "${STAGE1P5_ROOT}" 1

launch_stage "${STAGE2_RUN}" "${STAGE2_CFG}" 29613 "${STAGE1P5_CKPT}"
wait_stage "${STAGE2_RUN}"
STAGE2_CKPT="$(best_or_latest "${STAGE2_ROOT}")" || { log "missing_stage2_ckpt"; exit 5; }
eval_stage "stage2" "${STAGE2_CFG}" "${STAGE2_CKPT}" "${STAGE2_ROOT}" 0

log "flow_done stage0=${STAGE0_CKPT} stage1=${STAGE1_CKPT} stage1p5=${STAGE1P5_CKPT} stage2=${STAGE2_CKPT}"
