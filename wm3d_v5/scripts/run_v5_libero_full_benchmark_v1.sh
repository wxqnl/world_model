#!/usr/bin/env bash
# Full LIBERO closed-loop benchmark for the WM3D-v5 1B frozen-trunk action-policy head.
#
# Protocol (paper-grade, same as OpenVLA/LIBERO):
#   4 suites x 10 tasks x 50 init states x <=300 steps  = 2000 episodes.
# Topology:
#   8 persistent policy servers (one per GPU on node1), loaded ONCE and reused
#   across all 4 suites. Per suite, the 50 init states are sharded across the 8
#   GPUs and the runners execute in parallel (server + MuJoCo sim share one GPU).
# Conditioning (standard / non-privileged, train==serve):
#   WM tokens + task emb + lowdim proprio + action_history + progress.
#   object_state / plan_state are intentionally NOT sent.
set -euo pipefail

cd /data/Minko/world_model/wm3d_v5
export PYTHONPATH="/data/Minko/world_model/wm3d_v5:${PYTHONPATH:-}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export LIBGL_DRIVERS_PATH="${LIBGL_DRIVERS_PATH:-/usr/lib/x86_64-linux-gnu/dri/}"
export __EGL_VENDOR_LIBRARY_FILENAMES="${__EGL_VENDOR_LIBRARY_FILENAMES:-/data/Minko/egl/10_nvidia.json}"

PY="${PY:-/data/Minko/.venvs/wm3d/bin/python}"
LIBERO_PY="${LIBERO_PY:-/data/Minko/.conda-envs/libero-py38/bin/python}"
LIBERO_ROOT="${LIBERO_ROOT:-/data/Minko/benchmarks/LIBERO}"
BASE_CFG="${BASE_CFG:-configs/v5_p64_1b_libero_action_policy_base_from_stage2_v1.yaml}"
SFT_ROOT="${SFT_ROOT:-results/wm3d_v5_p64_1b_libero_action_policy_sft_from_stage2_v1}"
CKPT="${CKPT:-${SFT_ROOT}/ckpt/best.pt}"
BENCH_ROOT="${BENCH_ROOT:-${SFT_ROOT}/libero_full_4suite_10task_50init_300step_v1}"
LOG_DIR="${LOG_DIR:-${BENCH_ROOT}/logs}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
BASE_PORT="${BASE_PORT:-8901}"
SUITES="${SUITES:-libero_spatial libero_object libero_goal libero_10}"
MAX_TASKS="${MAX_TASKS:-10}"
MAX_STEPS="${MAX_STEPS:-300}"
CAMERA_KEY="${CAMERA_KEY:-agentview_image}"
CAMERA_SIZE="${CAMERA_SIZE:-256}"
CONTEXT_T="${CONTEXT_T:-16}"
WARMUP_STEPS="${WARMUP_STEPS:-5}"
GRIPPER_MODE="${GRIPPER_MODE:-closed01_to_libero}"
ACTION_HISTORY_LEN="${ACTION_HISTORY_LEN:-16}"
EXEC_HORIZON="${EXEC_HORIZON:-1}"
# Serve conditioning toggles (train==serve). Training feeds progress=None and,
# for the copycat-fixed model, action_history is dropped. Set SEND_PROGRESS=0 to
# omit progress; ACTION_HISTORY_LEN=0 omits action_history.
SEND_PROGRESS="${SEND_PROGRESS:-1}"
SEND_PROGRESS_FLAG=""; [[ "${SEND_PROGRESS}" == "1" ]] && SEND_PROGRESS_FLAG="--send_progress"
# 50 init states (0..49) split across 8 GPUs (7*7 + 1):
INIT_STATE_SHARDS="${INIT_STATE_SHARDS:-0,1,2,3,4,5,6;7,8,9,10,11,12,13;14,15,16,17,18,19,20;21,22,23,24,25,26,27;28,29,30,31,32,33,34;35,36,37,38,39,40,41;42,43,44,45,46,47,48;49}"
TASK_CACHE_DIR="${TASK_CACHE_DIR:-/0604-10T-test/wm3d_v5/cache/libero_taskemb_online}"

mkdir -p "${BENCH_ROOT}" "${LOG_DIR}" "${TASK_CACHE_DIR}"
IFS=',' read -r -a GPU_ARR <<< "${GPUS}"
IFS=';' read -r -a SHARD_ARR <<< "${INIT_STATE_SHARDS}"
NGPU="${#GPU_ARR[@]}"

log() { echo "[$(date -Is)] $*"; }

require_ckpt() {
  if [[ ! -s "${CKPT}" ]]; then
    log "missing action-policy checkpoint: ${CKPT} (train first)"; exit 6
  fi
}

SERVER_PIDS=()
start_servers() {
  log "starting ${NGPU} policy servers cfg=${BASE_CFG} ckpt=${CKPT}"
  for idx in "${!GPU_ARR[@]}"; do
    local gpu="${GPU_ARR[$idx]}"; local port=$((BASE_PORT + idx))
    CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 nohup "${PY}" -m wm3d_v3.policy.http_policy_server \
      --cfg "${BASE_CFG}" --ckpt "${CKPT}" --host 127.0.0.1 --port "${port}" \
      --device cuda:0 --qwen_device cuda:0 \
      --task_cache_dir "${TASK_CACHE_DIR}" --selection_mode action_policy \
      > "${LOG_DIR}/policy_server_gpu${gpu}_port${port}.log" 2>&1 &
    SERVER_PIDS+=("$!")
    log "  server idx=${idx} gpu=${gpu} port=${port} pid=$!"
  done
}

wait_servers_ready() {
  for idx in "${!GPU_ARR[@]}"; do
    local port=$((BASE_PORT + idx)); local ok=0
    for _ in $(seq 1 300); do
      if curl -fsS "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then ok=1; break; fi
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

run_suite() {
  local suite="$1"
  log "=== suite ${suite}: ${NGPU}-way init-state shard (10 tasks x 50 init) ==="
  local pids=()
  for idx in "${!GPU_ARR[@]}"; do
    local gpu="${GPU_ARR[$idx]}"; local port=$((BASE_PORT + idx)); local shard="${SHARD_ARR[$idx]:-}"
    [[ -z "${shard}" ]] && continue
    local out="${BENCH_ROOT}/${suite}_shard${idx}.json"
    if [[ -s "${out}" ]]; then log "  ${suite} shard ${idx} exists, skip"; continue; fi
    (
      CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 "${LIBERO_PY}" -m wm3d_v3.benchmarks.libero_remote_runner \
        --libero_root "${LIBERO_ROOT}" \
        --server_url "http://127.0.0.1:${port}" \
        --suite "${suite}" \
        --max_tasks "${MAX_TASKS}" \
        --init_states "${shard}" \
        --max_steps "${MAX_STEPS}" \
        --camera_key "${CAMERA_KEY}" \
        --camera_size "${CAMERA_SIZE}" \
        --context_T "${CONTEXT_T}" \
        --warmup_steps "${WARMUP_STEPS}" \
        --gripper_mode "${GRIPPER_MODE}" \
        --exec_horizon "${EXEC_HORIZON}" \
        --send_lowdim --action_history_len "${ACTION_HISTORY_LEN}" ${SEND_PROGRESS_FLAG} \
        --out "${out}"
    ) > "${LOG_DIR}/${suite}_shard${idx}.log" 2>&1 &
    pids+=("$!")
    log "  launched ${suite} shard ${idx} gpu=${gpu} port=${port} init=[${shard}]"
  done
  local failed=0
  for pid in "${pids[@]:-}"; do [[ -n "${pid:-}" ]] && { wait "${pid}" || failed=1; }; done
  [[ "${failed}" == "0" ]] || log "WARNING: a ${suite} shard failed (see ${LOG_DIR}/${suite}_shard*.log)"
  "${PY}" -m wm3d_v3.benchmarks.libero_trace_summary \
    --input "${BENCH_ROOT}/${suite}_shard"*.json \
    --out "${BENCH_ROOT}/${suite}_summary.json" \
    --jsonl_out "${BENCH_ROOT}/${suite}_trace.jsonl" || true
  log "  ${suite} summary -> ${BENCH_ROOT}/${suite}_summary.json"
}

scorecard() {
  SUITES="${SUITES}" BENCH_ROOT="${BENCH_ROOT}" "${PY}" - <<'PY'
import json, os
from pathlib import Path
root = Path(os.environ["BENCH_ROOT"])
suites = os.environ["SUITES"].split()
rows = []
for s in suites:
    p = root / f"{s}_summary.json"
    if not p.exists():
        rows.append({"suite": s, "success_rate": None, "episodes": 0, "successes": 0}); continue
    d = json.loads(p.read_text())
    rows.append({"suite": s, "success_rate": d.get("success_rate"),
                 "episodes": d.get("episodes"), "successes": d.get("successes")})
rates = [r["success_rate"] for r in rows if r["success_rate"] is not None]
avg = sum(rates) / len(rates) if rates else None
total_ep = sum(r["episodes"] or 0 for r in rows)
total_su = sum(r["successes"] or 0 for r in rows)
card = {
    "benchmark": "LIBERO closed-loop (frozen WM3D-v5 1B trunk + BC action-policy head)",
    "protocol": {"suites": suites, "tasks_per_suite": 10, "init_states_per_task": 50,
                 "max_steps": 300, "selection_mode": "action_policy",
                 "conditioning": ["wm_tokens", "task_emb", "lowdim", "action_history", "progress"],
                 "camera": "agentview only"},
    "per_suite": rows,
    "suite_mean_success_rate": avg,
    "pooled": {"successes": total_su, "episodes": total_ep,
               "success_rate": (total_su / total_ep) if total_ep else None},
}
(root / "final_scorecard.json").write_text(json.dumps(card, indent=2, sort_keys=True))
lines = ["# LIBERO Final Scorecard - WM3D-v5 1B (frozen trunk + BC head)", "",
         "| suite | success | episodes | rate |", "|---|---|---|---|"]
for r in rows:
    rr = "n/a" if r["success_rate"] is None else f"{r['success_rate']:.4f}"
    lines.append(f"| {r['suite']} | {r['successes']} | {r['episodes']} | {rr} |")
lines += ["", f"**Suite-mean success rate:** {('n/a' if avg is None else f'{avg:.4f}')}",
          f"**Pooled:** {total_su}/{total_ep} = {('n/a' if not total_ep else f'{total_su/total_ep:.4f}')}",
          "", "Protocol: 4 suites x 10 tasks x 50 init states, max_steps=300, agentview-only,",
          "frozen 1B 3D-native world-model trunk + lightweight BC action-policy head;",
          "conditioning = WM tokens + task emb + proprio lowdim + action history + progress",
          "(no privileged object-state / plan-state)."]
(root / "final_scorecard.md").write_text("\n".join(lines) + "\n")
print(json.dumps({"suite_mean": avg, "pooled": card["pooled"]}, sort_keys=True))
PY
}

main() {
  require_ckpt
  start_servers
  wait_servers_ready
  for suite in ${SUITES}; do run_suite "${suite}"; done
  scorecard
  log "DONE scorecard=${BENCH_ROOT}/final_scorecard.md"
}

cmd="${1:-all}"
case "${cmd}" in
  servers) start_servers; wait_servers_ready; log "servers up; ctrl-c to stop"; wait ;;
  suite) require_ckpt; start_servers; wait_servers_ready; run_suite "${2:?usage: $0 suite <name>}"; scorecard ;;
  scorecard) scorecard ;;
  all) main ;;
  *) echo "usage: $0 [all|servers|suite <name>|scorecard]" >&2; exit 2 ;;
esac
