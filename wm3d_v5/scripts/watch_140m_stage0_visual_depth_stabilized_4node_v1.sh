#!/usr/bin/env bash
set -euo pipefail

cd /data/Minko/world_model/wm3d_v3

PYTHON="${PYTHON:-/data/Minko/.venvs/wm3d/bin/python}"
LOG_DIR="${LOG_DIR:-/data/Minko/logs}"
RUN="train_140m_stage0_visual_depth_stabilized_4node_v1"
CFG="configs/v3_p64_140m_stage0_visual_depth_stabilized_4node_v1.yaml"
ROOT="results/wm3d_v3_p64_140m_stage0_visual_depth_stabilized_4node_v1"
WATCH_LOG="${WATCH_LOG:-${LOG_DIR}/watch_140m_stage0_visual_depth_stabilized_4node_v1.log}"
WORKER_HOSTS=(${WORKER_HOSTS:-root@172.27.0.7 root@172.27.0.4 root@172.27.0.5})

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

worker_alive() {
  local rank="$1"
  local host="$2"
  ssh -o BatchMode=yes -o ConnectTimeout=8 "${host}" \
    "test -f ${LOG_DIR}/${RUN}_node${rank}.pid && ps -p \$(cat ${LOG_DIR}/${RUN}_node${rank}.pid) >/dev/null 2>&1"
}

any_worker_alive() {
  local rank=1
  local host
  for host in "${WORKER_HOSTS[@]}"; do
    if worker_alive "${rank}" "${host}"; then
      return 0
    fi
    rank=$((rank + 1))
  done
  return 1
}

best_or_latest() {
  if [[ -f "${ROOT}/ckpt/best.pt" ]]; then
    echo "${ROOT}/ckpt/best.pt"
  elif [[ -f "${ROOT}/ckpt/latest.pt" ]]; then
    echo "${ROOT}/ckpt/latest.pt"
  else
    return 1
  fi
}

log "watch_start run=${RUN}"
while pid_alive "${LOG_DIR}/${RUN}_node0.pid"; do
  grep "\[rank0\] step" "${LOG_DIR}/${RUN}_node0.log" 2>/dev/null | tail -n 1 | tee -a "${WATCH_LOG}" || true
  sleep 60
done

while any_worker_alive; do
  log "waiting_workers"
  sleep 60
done

log "training_exited"
grep -E "Traceback|RuntimeError|CUDA out|NCCL WARN|NCCL ERROR|Exception|non-finite|NaN" \
  "${LOG_DIR}/${RUN}_node0.log" | tail -n 120 >> "${WATCH_LOG}" || true

CKPT="$(best_or_latest)" || { log "missing_checkpoint"; exit 2; }
OUT_DIR="${ROOT}/eval_after_stage0_stabilized"
mkdir -p "${OUT_DIR}"

log "eval_basic ckpt=${CKPT}"
CUDA_VISIBLE_DEVICES=0 "${PYTHON}" -m wm3d_v3.eval.run_eval \
  --cfg "${CFG}" \
  --ckpt "${CKPT}" \
  --out "${OUT_DIR}/eval_rgb_depth_48b.json" \
  --max_batches 48 \
  --batch_size 4 \
  > "${LOG_DIR}/${RUN}_eval_basic.log" 2>&1

log "eval_demo_gif"
CUDA_VISIBLE_DEVICES=0 "${PYTHON}" -m wm3d_v3.eval.make_demo_gif \
  --cfg "${CFG}" \
  --ckpt "${CKPT}" \
  --out_dir "${OUT_DIR}/demo_gifs" \
  --n_clips 8 \
  > "${LOG_DIR}/${RUN}_eval_demo_gif.log" 2>&1 || log "demo_gif_failed"

log "eval_native3d"
GPU=0 CFG="${CFG}" CKPT="${CKPT}" OUT_DIR="${OUT_DIR}/native3d_benchmark_v2" \
  MAX_BATCHES_PER_DATASET=2 N_VIZ=8 \
  bash scripts/run_world3d_native_benchmark_v2.sh \
  > "${LOG_DIR}/${RUN}_eval_native3d.log" 2>&1 || log "native3d_failed"

find "${OUT_DIR}" -maxdepth 4 -type f | sort > "${OUT_DIR}/artifacts.txt"
log "watch_done ckpt=${CKPT} out=${OUT_DIR}"
