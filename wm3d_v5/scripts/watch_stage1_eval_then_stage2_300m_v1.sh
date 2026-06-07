#!/usr/bin/env bash
set -euo pipefail

cd /data/Minko/world_model/wm3d_v3

PYTHON="${PYTHON:-/data/Minko/.venvs/wm3d/bin/python}"
LOG_DIR="${LOG_DIR:-/data/Minko/logs}"
WATCH_LOG="${WATCH_LOG:-${LOG_DIR}/watch_stage1_eval_then_stage2_300m_v1.log}"
STAGE1_RUN="train_300m_stage1_world_oxe_droid20k_fromscratch_2node_v1"
STAGE1_CFG="configs/v3_p64_300m_stage1_world_oxe_droid20k_fromscratch_2node_v1.yaml"
STAGE1_ROOT="results/wm3d_v3_p64_300m_stage1_world_oxe_droid20k_fromscratch_2node_v1"
STAGE2_RUN_SCRIPT="scripts/run_300m_stage2_oxe_droid20k_joint_visual_proposer_from_stage1_2node_v1.sh"
STAGE2_PIDFILE="${LOG_DIR}/train_300m_stage2_oxe_droid20k_joint_visual_proposer_from_stage1_2node_v1_node0.pid"
WORKER_HOST="${WORKER_HOST:-root@172.27.0.7}"

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

log "watch_start"
while pid_alive "${LOG_DIR}/${STAGE1_RUN}_node0.pid"; do
  grep "\[rank0\] step" "${LOG_DIR}/${STAGE1_RUN}_node0.log" 2>/dev/null | tail -n 1 | tee -a "${WATCH_LOG}" || true
  sleep 300
done
log "stage1_node0_exited"

if ssh -o BatchMode=yes -o ConnectTimeout=8 "${WORKER_HOST}" "test -f ${LOG_DIR}/${STAGE1_RUN}_node1.pid && ps -p \$(cat ${LOG_DIR}/${STAGE1_RUN}_node1.pid) >/dev/null 2>&1"; then
  log "waiting_stage1_node1"
  while ssh -o BatchMode=yes -o ConnectTimeout=8 "${WORKER_HOST}" "ps -p \$(cat ${LOG_DIR}/${STAGE1_RUN}_node1.pid) >/dev/null 2>&1"; do
    sleep 120
  done
fi
log "stage1_all_exited"

grep -E "Traceback|RuntimeError|CUDA out|NCCL WARN|NCCL ERROR|Error|Exception" "${LOG_DIR}/${STAGE1_RUN}_node0.log" | tail -n 40 >> "${WATCH_LOG}" || true

STAGE1_CKPT="${STAGE1_ROOT}/ckpt/best.pt"
if [[ ! -f "${STAGE1_CKPT}" ]]; then
  STAGE1_CKPT="${STAGE1_ROOT}/ckpt/latest.pt"
fi
if [[ ! -f "${STAGE1_CKPT}" ]]; then
  log "missing_stage1_checkpoint"
  exit 2
fi
log "stage1_ckpt=${STAGE1_CKPT}"

EVAL_DIR="${STAGE1_ROOT}/basic_eval_after_stage1"
mkdir -p "${EVAL_DIR}"
log "start_stage1_basic_eval"
CUDA_VISIBLE_DEVICES=0 "${PYTHON}" -m wm3d_v3.eval.run_eval \
  --cfg "${STAGE1_CFG}" \
  --ckpt "${STAGE1_CKPT}" \
  --out "${EVAL_DIR}/eval_rgb_depth_64b.json" \
  --max_batches 64 \
  --batch_size 4 \
  > "${LOG_DIR}/stage1_basic_eval_300m_oxe_droid20k.log" 2>&1
log "done_stage1_basic_eval out=${EVAL_DIR}/eval_rgb_depth_64b.json"

log "start_stage1_world3d_claim_eval"
CUDA_VISIBLE_DEVICES=0 "${PYTHON}" -m wm3d_v3.eval.world3d_claim_eval \
  --cfg "${STAGE1_CFG}" \
  --ckpt "${STAGE1_CKPT}" \
  --out "${EVAL_DIR}/world3d_claim_32b.json" \
  --max_batches 32 \
  --batch_size 4 \
  --variants zero sign_flip grip_toggle \
  > "${LOG_DIR}/stage1_world3d_claim_eval_300m_oxe_droid20k.log" 2>&1
log "done_stage1_world3d_claim_eval out=${EVAL_DIR}/world3d_claim_32b.json"

log "start_stage1_generation_canary"
CUDA_VISIBLE_DEVICES=0 \
CFG="${STAGE1_CFG}" \
CKPT="${STAGE1_CKPT}" \
OUT_DIR="${EVAL_DIR}/generation_canary" \
MAX_BATCHES=16 \
BATCH_SIZE=1 \
N_GIFS=2 \
N_HUNYUAN_GIFS=1 \
bash scripts/run_generation_canary_v1.sh \
  > "${LOG_DIR}/stage1_generation_canary_300m_oxe_droid20k.log" 2>&1 \
  || log "stage1_generation_canary_failed log=${LOG_DIR}/stage1_generation_canary_300m_oxe_droid20k.log"
log "done_stage1_generation_canary"

if [[ -f "${STAGE2_PIDFILE}" ]] && ps -p "$(cat "${STAGE2_PIDFILE}")" >/dev/null 2>&1; then
  log "stage2_already_running pid=$(cat "${STAGE2_PIDFILE}")"
  exit 0
fi

log "start_stage2 resume=${STAGE1_CKPT}"
RESUME_CKPT="${STAGE1_CKPT}" bash "${STAGE2_RUN_SCRIPT}" >> "${WATCH_LOG}" 2>&1
log "stage2_launch_done"
