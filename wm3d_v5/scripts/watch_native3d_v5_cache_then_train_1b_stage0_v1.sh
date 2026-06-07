#!/usr/bin/env bash
set -euo pipefail

cd /data/Minko/world_model/wm3d_v5

PYTHON="${PYTHON:-/data/Minko/.venvs/wm3d/bin/python}"
LOG_DIR="${LOG_DIR:-/data/Minko/logs}"
CACHE_ROOT="${CACHE_ROOT:-/0604-10T-test/wm3d_v5/cache}"
MANIFEST="${MANIFEST:-/data/Minko/world_model/wm3d_v5/manifests/oxe_droid20k_depthplus_world_v1.jsonl}"
WINDOW_SUBDIR="${WINDOW_SUBDIR:-vggt_window_geom_p64_native3d_v5_w1_v1}"
CACHE_RUN="${CACHE_RUN:-cache_native3d_v5_w1_stage0_1b_v1}"
TRAIN_RUN="${TRAIN_RUN:-train_v5_p64_1b_stage0_native3d_wsd_4node_v1}"
CFG="${CFG:-v5_p64_1b_stage0_native3d_wsd_4node_v1.yaml}"
GEOM_T="${GEOM_T:-16}"
GEOM_K="${GEOM_K:-8}"
GEOM_STRIDE="${GEOM_STRIDE:-4}"
GEOM_HW="${GEOM_HW:-64}"
GEOM_MAX_WINDOWS="${GEOM_MAX_WINDOWS:-0}"
MASTER_PORT="${MASTER_PORT:-29590}"
AUTO_TRAIN="${AUTO_TRAIN:-0}"
WORKER_HOSTS=(${WORKER_HOSTS:-root@172.27.0.7 root@172.27.0.4})
GPUS=(${GPUS:-0 1 2 3 4 5 6 7})
NNODES=$((1 + ${#WORKER_HOSTS[@]}))
WORLD=$((NNODES * ${#GPUS[@]}))
SHARED_CACHE="${SHARED_CACHE:-1}"
VALIDATE_CACHE="${VALIDATE_CACHE:-1}"
CACHE_P="${CACHE_P:-64}"
CACHE_D="${CACHE_D:-2048}"

mkdir -p "${LOG_DIR}" "${CACHE_ROOT}/${WINDOW_SUBDIR}"

sync_worker_code() {
  local host="$1"
  ssh "${host}" "mkdir -p /data/Minko/world_model/wm3d_v5 ${LOG_DIR} ${CACHE_ROOT}/${WINDOW_SUBDIR}"
  rsync -a --delete --exclude=results --exclude=.git --exclude='__pycache__' --exclude='*.pyc' \
    ./ "${host}:/data/Minko/world_model/wm3d_v5/"
}

echo "[watch] syncing v5 code to workers"
for host in "${WORKER_HOSTS[@]}"; do
  sync_worker_code "${host}"
done

launch_cache_one() {
  local host="$1"
  local shard="$2"
  local gpu="$3"
  local log="${LOG_DIR}/${CACHE_RUN}_shard${shard}.log"
  local pid="${LOG_DIR}/${CACHE_RUN}_shard${shard}.pid"
  local cmd="cd /data/Minko/world_model/wm3d_v5 && (setsid env CUDA_VISIBLE_DEVICES=${gpu} PYTHONUNBUFFERED=1 ${PYTHON} scripts/cache_oxe.py --manifest ${MANIFEST} --cache_root ${CACHE_ROOT} --skip_qwen --no_geom --no_rgb --geom_extra_subdir ${WINDOW_SUBDIR} --geom_extra_T ${GEOM_T} --geom_extra_k ${GEOM_K} --geom_extra_stride ${GEOM_STRIDE} --geom_extra_hw ${GEOM_HW} --geom_extra_max_windows_per_episode ${GEOM_MAX_WINDOWS} --shard ${shard} --world ${WORLD} > ${log} 2>&1 < /dev/null & echo \$! > ${pid})"
  if [[ "${host}" == "local" ]]; then
    bash -lc "${cmd}"
  else
    ssh "${host}" "${cmd}"
  fi
}

echo "[watch] launching ${WORLD} window-geom cache shards into ${WINDOW_SUBDIR}"
shard=0
for gpu in "${GPUS[@]}"; do
  launch_cache_one local "${shard}" "${gpu}"
  shard=$((shard + 1))
done
for host in "${WORKER_HOSTS[@]}"; do
  for gpu in "${GPUS[@]}"; do
    launch_cache_one "${host}" "${shard}" "${gpu}"
    shard=$((shard + 1))
  done
done

is_alive() {
  local host="$1"
  local pid_path="$2"
  if [[ "${host}" == "local" ]]; then
    local pid
    pid="$(cat "${pid_path}" 2>/dev/null || true)"
    [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null
  else
    ssh "${host}" "pid=\$(cat ${pid_path} 2>/dev/null || true); [[ -n \"\$pid\" ]] && kill -0 \"\$pid\" 2>/dev/null"
  fi
}

echo "[watch] waiting for cache shards"
while true; do
  alive=0
  shard=0
  for _gpu in "${GPUS[@]}"; do
    if is_alive local "${LOG_DIR}/${CACHE_RUN}_shard${shard}.pid"; then alive=$((alive + 1)); fi
    shard=$((shard + 1))
  done
  for host in "${WORKER_HOSTS[@]}"; do
    for _gpu in "${GPUS[@]}"; do
      if is_alive "${host}" "${LOG_DIR}/${CACHE_RUN}_shard${shard}.pid"; then alive=$((alive + 1)); fi
      shard=$((shard + 1))
    done
  done
  echo "[watch] alive_cache_shards=${alive}/${WORLD}"
  if [[ "${alive}" == "0" ]]; then
    break
  fi
  sleep 300
done

echo "[watch] checking cache logs for hard failures"
if grep -R -E "Traceback|RuntimeError|CUDA out of memory|KeyError" "${LOG_DIR}/${CACHE_RUN}"_shard*.log; then
  echo "[watch] local cache logs contain hard failures" >&2
  exit 1
fi
shard_offset=${#GPUS[@]}
for host in "${WORKER_HOSTS[@]}"; do
  if ssh "${host}" "grep -R -E 'Traceback|RuntimeError|CUDA out of memory|KeyError' ${LOG_DIR}/${CACHE_RUN}_shard*.log"; then
    echo "[watch] ${host} cache logs contain hard failures" >&2
    exit 1
  fi
  shard_offset=$((shard_offset + ${#GPUS[@]}))
done

if [[ "${SHARED_CACHE}" == "1" ]]; then
  echo "[watch] shared cache enabled; skipping gather/fanout rsync"
else
  echo "[watch] gathering worker window cache to node0"
  for host in "${WORKER_HOSTS[@]}"; do
    rsync -a "${host}:${CACHE_ROOT}/${WINDOW_SUBDIR}/" "${CACHE_ROOT}/${WINDOW_SUBDIR}/"
  done

  echo "[watch] fanout full window cache to workers"
  for host in "${WORKER_HOSTS[@]}"; do
    ssh "${host}" "mkdir -p ${CACHE_ROOT}/${WINDOW_SUBDIR}"
    rsync -a "${CACHE_ROOT}/${WINDOW_SUBDIR}/" "${host}:${CACHE_ROOT}/${WINDOW_SUBDIR}/"
  done
fi

if [[ "${VALIDATE_CACHE}" == "1" ]]; then
  echo "[watch] validating window cache before training"
  "${PYTHON}" scripts/validate_native3d_window_cache.py \
    --manifest "${MANIFEST}" \
    --cache_root "${CACHE_ROOT}" \
    --window_subdir "${WINDOW_SUBDIR}" \
    --T "${GEOM_T}" \
    --k "${GEOM_K}" \
    --stride "${GEOM_STRIDE}" \
    --max_windows_per_episode "${GEOM_MAX_WINDOWS}" \
    --P "${CACHE_P}" \
    --D "${CACHE_D}" \
    --hw "${GEOM_HW}" \
    --require_task_emb \
    --sample_dataset \
    --strict
fi

if [[ "${AUTO_TRAIN}" != "1" ]]; then
  echo "[watch] cache complete and synced; AUTO_TRAIN=${AUTO_TRAIN}, not launching training"
  exit 0
fi

echo "[watch] launching v5 stage0 training"
MASTER_PORT="${MASTER_PORT}" RUN_NAME="${TRAIN_RUN}" CFG="${CFG}" WORKER_HOSTS="${WORKER_HOSTS[*]}" \
  scripts/run_v5_stage_4node_v1.sh

echo "[watch] training launched: ${TRAIN_RUN}"
