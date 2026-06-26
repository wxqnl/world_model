#!/usr/bin/env bash
set -euo pipefail

cd /data/Minko/world_model/wm3d_v5

export PYTHONPATH="/data/Minko/world_model/wm3d_v5:${PYTHONPATH:-}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/data/Minko/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/data/Minko/.cache/huggingface/hub}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export LIBGL_DRIVERS_PATH="${LIBGL_DRIVERS_PATH:-/usr/lib/x86_64-linux-gnu/dri/}"
export __EGL_VENDOR_LIBRARY_FILENAMES="${__EGL_VENDOR_LIBRARY_FILENAMES:-/data/Minko/egl/10_nvidia.json}"

PY="${PY:-/data/Minko/.venvs/wm3d/bin/python}"
LIBERO_PY="${LIBERO_PY:-/data/Minko/.conda-envs/libero-py38/bin/python}"
LIBERO_ROOT="${LIBERO_ROOT:-/data/Minko/benchmarks/LIBERO}"
BASE_CFG="${BASE_CFG:-configs/v5_p64_1b_libero_action_policy_fastwam_spatial_dualcam_concat_padded_earlyw_rgb_lastobs_k32_base_v1.yaml}"
CKPT="${CKPT:-/0604-10T-test/wm3d_v5/results/wm3d_v5_p64_1b_libero_spatial_fastwam_dualcam_concat_evalmatch_w0_rgb_rawxyz_lastobs_k32_sft_v1/ckpt/best.pt}"
BENCH_ROOT="${BENCH_ROOT:-/0604-10T-test/wm3d_v5/results/wm3d_v5_p64_1b_libero_spatial_fastwam_dualcam_concat_evalmatch_w0_rgb_rawxyz_lastobs_k32_sft_v1/libero_spatial_resumable_h10_w0_node41_v1}"
LOG_DIR="${LOG_DIR:-${BENCH_ROOT}/logs}"
EP_DIR="${EP_DIR:-${BENCH_ROOT}/episodes}"
TASK_CACHE_DIR="${TASK_CACHE_DIR:-/0604-10T-test/wm3d_v5/cache/libero_taskemb_online}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
BASE_PORT="${BASE_PORT:-11880}"
SUITE="${SUITE:-libero_spatial}"
MAX_TASKS="${MAX_TASKS:-10}"
TASK_IDS="${TASK_IDS:-}"
INIT_START="${INIT_START:-0}"
INIT_END="${INIT_END:-49}"
INIT_IDS="${INIT_IDS:-}"
EPISODE_LIST="${EPISODE_LIST:-}"
MAX_STEPS="${MAX_STEPS:-400}"
CAMERA_KEY="${CAMERA_KEY:-agentview_image}"
CAMERA_KEYS="${CAMERA_KEYS:-agentview_image,robot0_eye_in_hand_image}"
CAMERA_FUSION="${CAMERA_FUSION:-concat}"
CAMERA_SIZE="${CAMERA_SIZE:-256}"
ROTATE_180="${ROTATE_180:-1}"
CONTEXT_T="${CONTEXT_T:-16}"
WARMUP_STEPS="${WARMUP_STEPS:-0}"
GRIPPER_MODE="${GRIPPER_MODE:-closed01_to_libero}"
GRIPPER_CLOSED_THRESHOLD="${GRIPPER_CLOSED_THRESHOLD:-0.35}"
USE_POLICY_GRIPPER_PROB="${USE_POLICY_GRIPPER_PROB:-1}"
GRIPPER_MIN_CLOSE_Z="${GRIPPER_MIN_CLOSE_Z:-0.0}"
ACTION_HISTORY_LEN="${ACTION_HISTORY_LEN:-0}"
SEND_PROGRESS="${SEND_PROGRESS:-0}"
SEND_OBJECT_STATE="${SEND_OBJECT_STATE:-0}"
TRACE_OBJECT_STATE="${TRACE_OBJECT_STATE:-0}"
SEND_PLAN_STATE="${SEND_PLAN_STATE:-0}"
PLAN_STATE_DIM="${PLAN_STATE_DIM:-8}"
SAVE_FRAMES_ROOT="${SAVE_FRAMES_ROOT:-}"
SAVE_FRAME_EVERY="${SAVE_FRAME_EVERY:-0}"
EXEC_HORIZON="${EXEC_HORIZON:-10}"
SELECTION_MODE="${SELECTION_MODE:-direct}"
MAX_RETRIES="${MAX_RETRIES:-4}"
EPISODE_TIMEOUT="${EPISODE_TIMEOUT:-1800}"
RETRY_SLEEP="${RETRY_SLEEP:-5}"
BENCHMARK_LABEL="${BENCHMARK_LABEL:-LIBERO spatial closed-loop (WM3D-v5 eval-match RGB/rawxyz k32 SFT, resumable)}"

mkdir -p "${BENCH_ROOT}" "${LOG_DIR}" "${EP_DIR}" "${TASK_CACHE_DIR}" "${LOG_DIR}/episodes"
IFS=',' read -r -a GPU_ARR <<< "${GPUS}"
NGPU="${#GPU_ARR[@]}"
EPISODE_LIST_ABS=""
if [[ -n "${EPISODE_LIST}" ]]; then
  EPISODE_LIST_ABS="$(readlink -f "${EPISODE_LIST}")"
fi

log() { echo "[$(date -Is)] $*"; }

require_inputs() {
  [[ -s "${CKPT}" ]] || { log "missing checkpoint: ${CKPT}"; exit 6; }
}

episode_ok() {
  local path="$1"
  [[ -s "${path}" ]] || return 1
  "${PY}" - "${path}" >/dev/null 2>&1 <<'PY'
import json, sys
with open(sys.argv[1]) as fh:
    data = json.load(fh)
results = data.get("results") or []
assert len(results) == 1
PY
}

SERVER_PIDS=()
start_servers() {
  log "starting ${NGPU} policy servers cfg=${BASE_CFG} ckpt=${CKPT} selection_mode=${SELECTION_MODE}"
  for idx in "${!GPU_ARR[@]}"; do
    local gpu="${GPU_ARR[$idx]}"
    local port=$((BASE_PORT + idx))
    CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 nohup "${PY}" -m wm3d_v3.policy.http_policy_server \
      --cfg "${BASE_CFG}" --ckpt "${CKPT}" --host 127.0.0.1 --port "${port}" \
      --device cuda:0 --qwen_device cuda:0 \
      --camera_fusion "${CAMERA_FUSION}" \
      --task_cache_dir "${TASK_CACHE_DIR}" --selection_mode "${SELECTION_MODE}" \
      > "${LOG_DIR}/policy_server_gpu${gpu}_port${port}.log" 2>&1 &
    SERVER_PIDS+=("$!")
    log "  server idx=${idx} gpu=${gpu} port=${port} pid=$!"
  done
}

wait_servers_ready() {
  for idx in "${!GPU_ARR[@]}"; do
    local port=$((BASE_PORT + idx))
    local ok=0
    for _ in $(seq 1 300); do
      if curl -fsS "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
        ok=1
        break
      fi
      sleep 2
    done
    [[ "${ok}" == "1" ]] || { log "server port=${port} did not become ready"; exit 7; }
    log "  server port=${port} ready"
  done
}

stop_servers() {
  for pid in "${SERVER_PIDS[@]:-}"; do
    [[ -n "${pid:-}" ]] && kill "${pid}" 2>/dev/null || true
  done
}
trap stop_servers EXIT

run_episode() {
  local worker_idx="$1"
  local gpu="$2"
  local port="$3"
  local task_id="$4"
  local init_id="$5"
  local out="${EP_DIR}/${SUITE}_task$(printf '%02d' "${task_id}")_init$(printf '%02d' "${init_id}").json"
  if episode_ok "${out}"; then
    return 0
  fi

  local attempt
  for attempt in $(seq 1 "${MAX_RETRIES}"); do
    local tmp="${out}.tmp.worker${worker_idx}.try${attempt}"
    local ep_log="${LOG_DIR}/episodes/${SUITE}_task$(printf '%02d' "${task_id}")_init$(printf '%02d' "${init_id}")_try${attempt}.log"
    local rotate_flag=()
    local progress_flag=()
    local gripper_prob_flag=()
    local gripper_gate_flag=()
    local send_object_state_flag=()
    local trace_object_state_flag=()
    local send_plan_state_flag=()
    local save_frames_flag=()
    [[ "${ROTATE_180}" == "1" ]] && rotate_flag=(--rotate_180)
    [[ "${SEND_PROGRESS}" == "1" ]] && progress_flag=(--send_progress)
    [[ "${USE_POLICY_GRIPPER_PROB}" == "1" ]] && gripper_prob_flag=(--use_policy_gripper_prob)
    [[ "${GRIPPER_MIN_CLOSE_Z}" != "0" && "${GRIPPER_MIN_CLOSE_Z}" != "0.0" ]] && gripper_gate_flag=(--gripper_min_close_z "${GRIPPER_MIN_CLOSE_Z}")
    [[ "${SEND_OBJECT_STATE}" == "1" ]] && send_object_state_flag=(--send_object_state)
    [[ "${TRACE_OBJECT_STATE}" == "1" ]] && trace_object_state_flag=(--trace_object_state)
    [[ "${SEND_PLAN_STATE}" == "1" ]] && send_plan_state_flag=(--send_plan_state --plan_state_dim "${PLAN_STATE_DIM}")
    if [[ -n "${SAVE_FRAMES_ROOT}" ]]; then
      save_frames_flag=(
        --save_frames_dir "${SAVE_FRAMES_ROOT}/${SUITE}_task$(printf '%02d' "${task_id}")_init$(printf '%02d' "${init_id}")"
        --save_frame_every "${SAVE_FRAME_EVERY}"
      )
    fi
    rm -f "${tmp}"
    log "worker=${worker_idx} gpu=${gpu} task=${task_id} init=${init_id} attempt=${attempt}"
    set +e
    CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 LIBERO_SKIP_ENV_CLOSE="${LIBERO_SKIP_ENV_CLOSE:-1}" \
      timeout --kill-after=30s "${EPISODE_TIMEOUT}" \
      "${LIBERO_PY}" -m wm3d_v3.benchmarks.libero_remote_runner \
        --libero_root "${LIBERO_ROOT}" \
        --server_url "http://127.0.0.1:${port}" \
        --suite "${SUITE}" \
        --task_ids "${task_id}" \
        --max_tasks "${MAX_TASKS}" \
        --init_states "${init_id}" \
        --max_steps "${MAX_STEPS}" \
        --camera_key "${CAMERA_KEY}" \
        --camera_keys "${CAMERA_KEYS}" \
        --camera_size "${CAMERA_SIZE}" \
        "${rotate_flag[@]}" \
        --context_T "${CONTEXT_T}" \
        --warmup_steps "${WARMUP_STEPS}" \
        --gripper_mode "${GRIPPER_MODE}" \
        --gripper_closed_threshold "${GRIPPER_CLOSED_THRESHOLD}" \
        "${gripper_prob_flag[@]}" \
        "${gripper_gate_flag[@]}" \
        --exec_horizon "${EXEC_HORIZON}" \
        --send_lowdim \
        "${send_object_state_flag[@]}" \
        "${trace_object_state_flag[@]}" \
        "${send_plan_state_flag[@]}" \
        --action_history_len "${ACTION_HISTORY_LEN}" \
        "${progress_flag[@]}" \
        "${save_frames_flag[@]}" \
        --out "${tmp}" \
      > "${ep_log}" 2>&1
    local code=$?
    set -e
    if episode_ok "${tmp}"; then
      mv "${tmp}" "${out}"
      log "done worker=${worker_idx} gpu=${gpu} task=${task_id} init=${init_id} code=${code} out=${out}"
      return 0
    fi
    log "retry worker=${worker_idx} gpu=${gpu} task=${task_id} init=${init_id} attempt=${attempt} code=${code} log=${ep_log}"
    rm -f "${tmp}"
    sleep "${RETRY_SLEEP}"
  done
  log "FAILED worker=${worker_idx} gpu=${gpu} task=${task_id} init=${init_id}"
  return 1
}

run_worker() {
  local worker_idx="$1"
  local gpu="${GPU_ARR[$worker_idx]}"
  local port=$((BASE_PORT + worker_idx))
  local failed=0
  local ordinal=0
  local task_id
  local init_id
  local task_ids=()
  local init_ids=()
  if [[ -n "${EPISODE_LIST_ABS}" ]]; then
    while read -r task_id init_id _rest; do
      [[ -z "${task_id:-}" || "${task_id:0:1}" == "#" ]] && continue
      if (( ordinal % NGPU == worker_idx )); then
        run_episode "${worker_idx}" "${gpu}" "${port}" "${task_id}" "${init_id}" || failed=1
      fi
      ordinal=$((ordinal + 1))
    done < "${EPISODE_LIST_ABS}"
    return "${failed}"
  fi
  if [[ -n "${TASK_IDS}" ]]; then
    IFS=',' read -r -a task_ids <<< "${TASK_IDS}"
  else
    mapfile -t task_ids < <(seq 0 $((MAX_TASKS - 1)))
  fi
  if [[ -n "${INIT_IDS}" ]]; then
    IFS=',' read -r -a init_ids <<< "${INIT_IDS}"
  else
    mapfile -t init_ids < <(seq "${INIT_START}" "${INIT_END}")
  fi
  for task_id in "${task_ids[@]}"; do
    task_id="$(echo "${task_id}" | xargs)"
    [[ -z "${task_id}" ]] && continue
    for init_id in "${init_ids[@]}"; do
      init_id="$(echo "${init_id}" | xargs)"
      [[ -z "${init_id}" ]] && continue
      if (( ordinal % NGPU == worker_idx )); then
        run_episode "${worker_idx}" "${gpu}" "${port}" "${task_id}" "${init_id}" || failed=1
      fi
      ordinal=$((ordinal + 1))
    done
  done
  return "${failed}"
}

summarize() {
  local task_count="${MAX_TASKS}"
  local init_count=$((INIT_END - INIT_START + 1))
  local expected
  if [[ -n "${EPISODE_LIST_ABS}" ]]; then
    expected=$(awk 'NF >= 2 && $1 !~ /^#/ {n++} END {print n+0}' "${EPISODE_LIST_ABS}")
  else
    if [[ -n "${INIT_IDS}" ]]; then
      local tmp_inits=()
      IFS=',' read -r -a tmp_inits <<< "${INIT_IDS}"
      init_count=0
      local init_id
      for init_id in "${tmp_inits[@]}"; do
        init_id="$(echo "${init_id}" | xargs)"
        [[ -n "${init_id}" ]] && init_count=$((init_count + 1))
      done
    fi
  if [[ -n "${TASK_IDS}" ]]; then
    local tmp_ids=()
    IFS=',' read -r -a tmp_ids <<< "${TASK_IDS}"
    task_count=0
    local task_id
    for task_id in "${tmp_ids[@]}"; do
      task_id="$(echo "${task_id}" | xargs)"
      [[ -n "${task_id}" ]] && task_count=$((task_count + 1))
    done
  fi
    expected=$((task_count * init_count))
  fi
  local completed
  completed=$(find "${EP_DIR}" -maxdepth 1 -type f -name "${SUITE}_task*_init*.json" | wc -l | tr -d ' ')
  log "completed episodes=${completed}/${expected}"
  if [[ "${completed}" == "0" ]]; then
    log "no completed episodes to summarize"
    exit 9
  fi
  "${PY}" -m wm3d_v3.benchmarks.libero_trace_summary \
    --input "${EP_DIR}/${SUITE}_task"*_init*.json \
    --out "${BENCH_ROOT}/${SUITE}_summary.json" \
    --jsonl_out "${BENCH_ROOT}/${SUITE}_trace.jsonl"
  SUITE="${SUITE}" BENCH_ROOT="${BENCH_ROOT}" EXPECTED="${expected}" COMPLETED="${completed}" \
  MAX_TASKS="${MAX_TASKS}" TASK_IDS="${TASK_IDS}" INIT_START="${INIT_START}" INIT_END="${INIT_END}" INIT_IDS="${INIT_IDS}" EPISODE_LIST="${EPISODE_LIST_ABS}" MAX_STEPS="${MAX_STEPS}" \
  EXEC_HORIZON="${EXEC_HORIZON}" WARMUP_STEPS="${WARMUP_STEPS}" CAMERA_KEYS="${CAMERA_KEYS}" \
  CAMERA_FUSION="${CAMERA_FUSION}" ROTATE_180="${ROTATE_180}" ACTION_HISTORY_LEN="${ACTION_HISTORY_LEN}" \
  SEND_PROGRESS="${SEND_PROGRESS}" SEND_OBJECT_STATE="${SEND_OBJECT_STATE}" TRACE_OBJECT_STATE="${TRACE_OBJECT_STATE}" SEND_PLAN_STATE="${SEND_PLAN_STATE}" \
  SAVE_FRAMES_ROOT="${SAVE_FRAMES_ROOT}" SAVE_FRAME_EVERY="${SAVE_FRAME_EVERY}" USE_POLICY_GRIPPER_PROB="${USE_POLICY_GRIPPER_PROB}" \
  GRIPPER_CLOSED_THRESHOLD="${GRIPPER_CLOSED_THRESHOLD}" GRIPPER_MIN_CLOSE_Z="${GRIPPER_MIN_CLOSE_Z}" \
  BENCHMARK_LABEL="${BENCHMARK_LABEL}" "${PY}" - <<'PY'
import json, os
from pathlib import Path
root = Path(os.environ["BENCH_ROOT"])
suite = os.environ["SUITE"]
summary = json.loads((root / f"{suite}_summary.json").read_text())
completed = int(os.environ["COMPLETED"])
expected = int(os.environ["EXPECTED"])
rate = summary.get("success_rate")
card = {
    "benchmark": os.environ["BENCHMARK_LABEL"],
    "protocol": {
        "suite": suite,
        "tasks": int(os.environ["MAX_TASKS"]),
        "task_ids": [int(x.strip()) for x in os.environ.get("TASK_IDS", "").split(",") if x.strip()],
        "init_start": int(os.environ["INIT_START"]),
        "init_end": int(os.environ["INIT_END"]),
        "init_ids": [int(x.strip()) for x in os.environ.get("INIT_IDS", "").split(",") if x.strip()],
        "episode_list": os.environ.get("EPISODE_LIST") or None,
        "expected_episodes": expected,
        "completed_episodes": completed,
        "complete": completed == expected,
        "max_steps": int(os.environ["MAX_STEPS"]),
        "exec_horizon": int(os.environ["EXEC_HORIZON"]),
        "warmup_steps": int(os.environ["WARMUP_STEPS"]),
        "conditioning": ["wm_tokens", "task_emb", "lowdim"],
        "camera_keys": [x for x in os.environ["CAMERA_KEYS"].split(",") if x],
        "camera_fusion": os.environ["CAMERA_FUSION"],
        "rotate_180": os.environ["ROTATE_180"] == "1",
        "action_history_len": int(os.environ["ACTION_HISTORY_LEN"]),
        "send_progress": os.environ["SEND_PROGRESS"] == "1",
        "send_object_state": os.environ["SEND_OBJECT_STATE"] == "1",
        "trace_object_state": os.environ["TRACE_OBJECT_STATE"] == "1",
        "send_plan_state": os.environ["SEND_PLAN_STATE"] == "1",
        "save_frames_root": os.environ.get("SAVE_FRAMES_ROOT") or None,
        "save_frame_every": int(os.environ["SAVE_FRAME_EVERY"]),
        "use_policy_gripper_prob": os.environ["USE_POLICY_GRIPPER_PROB"] == "1",
        "gripper_closed_threshold": float(os.environ["GRIPPER_CLOSED_THRESHOLD"]),
        "gripper_min_close_z": float(os.environ["GRIPPER_MIN_CLOSE_Z"]),
    },
    "per_suite": [{
        "suite": suite,
        "success_rate": rate,
        "episodes": summary.get("episodes"),
        "successes": summary.get("successes"),
    }],
    "pooled": {
        "success_rate": rate,
        "episodes": summary.get("episodes"),
        "successes": summary.get("successes"),
    },
}
(root / "final_scorecard.json").write_text(json.dumps(card, indent=2, sort_keys=True))
lines = [
    f"# LIBERO Final Scorecard - {card['benchmark']}",
    "",
    f"- suite: {suite}",
    f"- success_rate: {rate}",
    f"- successes: {summary.get('successes')}",
    f"- episodes: {summary.get('episodes')}",
    f"- complete: {completed == expected} ({completed}/{expected})",
]
(root / "final_scorecard.md").write_text("\n".join(lines) + "\n")
print(json.dumps(card, sort_keys=True))
PY
}

run_all() {
  require_inputs
  ulimit -c 0 || true
  start_servers
  wait_servers_ready
  local pids=()
  for idx in "${!GPU_ARR[@]}"; do
    run_worker "${idx}" > "${LOG_DIR}/worker${idx}.log" 2>&1 &
    pids+=("$!")
    log "worker idx=${idx} gpu=${GPU_ARR[$idx]} pid=$!"
  done
  local failed=0
  local pid
  for pid in "${pids[@]}"; do
    wait "${pid}" || failed=1
  done
  summarize
  [[ "${failed}" == "0" ]] || exit 10
}

case "${1:-all}" in
  all) run_all ;;
  summarize) summarize ;;
  *) echo "usage: $0 [all|summarize]" >&2; exit 2 ;;
esac
