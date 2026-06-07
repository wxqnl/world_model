#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/data/Minko}"
PROJECT_ROOT="${PROJECT_ROOT:-$ROOT/world_model/wm3d_v3}"
WM3D_PY="${WM3D_PY:-$ROOT/.venvs/wm3d/bin/python}"
LIBERO_PY="${LIBERO_PY:-$ROOT/.conda-envs/libero-py38/bin/python}"
LIBERO_ROOT="${LIBERO_ROOT:-$ROOT/benchmarks/LIBERO}"
CFG="${CFG:-configs/v3_p64_140m_actioncond_context_motion_p0_heads_candrank_ce.yaml}"
CKPT="${CKPT:-results/wm3d_v3_p64_140m_actioncond_context_motion_p0_heads_candrank_ce_v1/ckpt/best.pt}"
OUT="${OUT:-results/wm3d_v3_p64_140m_actioncond_context_motion_p0_heads_candrank_ce_v1/system_eval_best/libero_remote_rollout_smoke.json}"
SERVER_LOG="${SERVER_LOG:-$ROOT/logs/wm3d_policy_server_smoke.log}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8765}"
DEVICE="${DEVICE:-cuda:0}"
SELECTION_MODE="${SELECTION_MODE:-ranked}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
MAX_TASKS="${MAX_TASKS:-1}"
TASK_IDS="${TASK_IDS:-}"
INIT_STATES="${INIT_STATES:-0}"
INIT_STATE_HDF5="${INIT_STATE_HDF5:-}"
INIT_STATE_DEMO_ID="${INIT_STATE_DEMO_ID:-demo_0}"
EXPERT_ACTION_HDF5="${EXPERT_ACTION_HDF5:-}"
EXPERT_ACTION_DEMO_ID="${EXPERT_ACTION_DEMO_ID:-demo_0}"
EXPERT_ACTION_PREFIX_STEPS="${EXPERT_ACTION_PREFIX_STEPS:-0}"
EXPERT_ACTION_FULL_REPLAY="${EXPERT_ACTION_FULL_REPLAY:-0}"
MAX_STEPS="${MAX_STEPS:-20}"
CAMERA_SIZE="${CAMERA_SIZE:-128}"
CONTEXT_T="${CONTEXT_T:-16}"
WARMUP_STEPS="${WARMUP_STEPS:-}"
GRIPPER_MODE="${GRIPPER_MODE:-closed01_to_libero}"
POSE_SCALE="${POSE_SCALE:-1.0}"
MAX_POSE_NORM="${MAX_POSE_NORM:-0.0}"
CAMERA_KEY="${CAMERA_KEY:-agentview_image}"
SEND_LOWDIM="${SEND_LOWDIM:-0}"
SEND_OBJECT_STATE="${SEND_OBJECT_STATE:-0}"
SEND_PLAN_STATE="${SEND_PLAN_STATE:-0}"
PLAN_STATE_DIM="${PLAN_STATE_DIM:-8}"
FORCE_PLAN_STAGE3_GRIP_CLOSED="${FORCE_PLAN_STAGE3_GRIP_CLOSED:-0}"
ACTION_HISTORY_LEN="${ACTION_HISTORY_LEN:-0}"
SEND_PROGRESS="${SEND_PROGRESS:-0}"
PROGRESS_DENOMINATOR="${PROGRESS_DENOMINATOR:-0}"
TRACE_OBJECT_STATE="${TRACE_OBJECT_STATE:-0}"
SAVE_FRAMES_DIR="${SAVE_FRAMES_DIR:-}"
SAVE_FRAME_EVERY="${SAVE_FRAME_EVERY:-0}"
SUMMARY_OUT="${SUMMARY_OUT:-${OUT%.json}_summary.json}"
JSONL_OUT="${JSONL_OUT:-${OUT%.json}.jsonl}"

cd "$PROJECT_ROOT"
mkdir -p "$(dirname "$SERVER_LOG")" "$(dirname "$OUT")"

SERVER_PID=""
cleanup() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" \
TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}" \
"$WM3D_PY" -m wm3d_v3.policy.http_policy_server \
  --cfg "$CFG" \
  --ckpt "$CKPT" \
  --host "$HOST" \
  --port "$PORT" \
  --device "$DEVICE" \
  --qwen_device "$DEVICE" \
  --selection_mode "$SELECTION_MODE" \
  > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

RUNNER_EXTRA_ARGS=()
if [[ -n "$SAVE_FRAMES_DIR" ]]; then
  RUNNER_EXTRA_ARGS+=(--save_frames_dir "$SAVE_FRAMES_DIR" --save_frame_every "$SAVE_FRAME_EVERY")
fi
if [[ -n "$TASK_IDS" ]]; then
  RUNNER_EXTRA_ARGS+=(--task_ids "$TASK_IDS")
fi
if [[ -n "$INIT_STATE_HDF5" ]]; then
  RUNNER_EXTRA_ARGS+=(--init_state_hdf5 "$INIT_STATE_HDF5" --init_state_demo_id "$INIT_STATE_DEMO_ID")
  if [[ -z "$WARMUP_STEPS" ]]; then
    WARMUP_STEPS="0"
  fi
fi
if [[ -n "$EXPERT_ACTION_HDF5" ]]; then
  RUNNER_EXTRA_ARGS+=(--expert_action_hdf5 "$EXPERT_ACTION_HDF5" --expert_action_demo_id "$EXPERT_ACTION_DEMO_ID" --expert_action_prefix_steps "$EXPERT_ACTION_PREFIX_STEPS")
fi
if [[ "$EXPERT_ACTION_FULL_REPLAY" == "1" || "$EXPERT_ACTION_FULL_REPLAY" == "true" ]]; then
  RUNNER_EXTRA_ARGS+=(--expert_action_full_replay)
fi
if [[ -n "$WARMUP_STEPS" ]]; then
  RUNNER_EXTRA_ARGS+=(--warmup_steps "$WARMUP_STEPS")
fi
if [[ "$SEND_LOWDIM" == "1" || "$SEND_LOWDIM" == "true" ]]; then
  RUNNER_EXTRA_ARGS+=(--send_lowdim)
fi
if [[ "$SEND_OBJECT_STATE" == "1" || "$SEND_OBJECT_STATE" == "true" ]]; then
  RUNNER_EXTRA_ARGS+=(--send_object_state)
fi
if [[ "$SEND_PLAN_STATE" == "1" || "$SEND_PLAN_STATE" == "true" ]]; then
  RUNNER_EXTRA_ARGS+=(--send_plan_state --plan_state_dim "$PLAN_STATE_DIM")
fi
if [[ "$FORCE_PLAN_STAGE3_GRIP_CLOSED" == "1" || "$FORCE_PLAN_STAGE3_GRIP_CLOSED" == "true" ]]; then
  RUNNER_EXTRA_ARGS+=(--force_plan_stage3_grip_closed)
fi
if [[ "$ACTION_HISTORY_LEN" != "0" ]]; then
  RUNNER_EXTRA_ARGS+=(--action_history_len "$ACTION_HISTORY_LEN")
fi
if [[ "$SEND_PROGRESS" == "1" || "$SEND_PROGRESS" == "true" ]]; then
  RUNNER_EXTRA_ARGS+=(--send_progress --progress_denominator "$PROGRESS_DENOMINATOR")
fi
if [[ "$TRACE_OBJECT_STATE" == "1" || "$TRACE_OBJECT_STATE" == "true" ]]; then
  RUNNER_EXTRA_ARGS+=(--trace_object_state)
fi
if [[ "$POSE_SCALE" != "1.0" ]]; then
  RUNNER_EXTRA_ARGS+=(--pose_scale "$POSE_SCALE")
fi
if [[ "$MAX_POSE_NORM" != "0.0" ]]; then
  RUNNER_EXTRA_ARGS+=(--max_pose_norm "$MAX_POSE_NORM")
fi

"$WM3D_PY" - <<PY
import json, time, urllib.request
url = "http://${HOST}:${PORT}/health"
for _ in range(180):
    try:
        with urllib.request.urlopen(url, timeout=2) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        if data.get("ok"):
            print(json.dumps({"policy_server": "ready", "url": url}))
            break
    except Exception:
        time.sleep(1)
else:
    raise SystemExit("policy server did not become ready")
PY

xvfb-run -a -s "-screen 0 1024x768x24" \
  env MUJOCO_GL=glfw PYTHONPATH="$PROJECT_ROOT" \
  "$LIBERO_PY" -m wm3d_v3.benchmarks.libero_remote_runner \
    --libero_root "$LIBERO_ROOT" \
    --server_url "http://${HOST}:${PORT}" \
    --suite libero_10 \
    --max_tasks "$MAX_TASKS" \
    --init_states "$INIT_STATES" \
    --max_steps "$MAX_STEPS" \
    --camera_key "$CAMERA_KEY" \
    --camera_size "$CAMERA_SIZE" \
    --context_T "$CONTEXT_T" \
    --gripper_mode "$GRIPPER_MODE" \
    "${RUNNER_EXTRA_ARGS[@]}" \
    --out "$OUT"

"$WM3D_PY" -m wm3d_v3.benchmarks.libero_trace_summary \
  --input "$OUT" \
  --out "$SUMMARY_OUT" \
  --jsonl_out "$JSONL_OUT"

echo "wrote $OUT"
echo "wrote $SUMMARY_OUT"
echo "wrote $JSONL_OUT"
