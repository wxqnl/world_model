#!/usr/bin/env bash
set -euo pipefail

cd /data/Minko/world_model/wm3d_v3

VENV_BIN=${VENV_BIN:-/data/Minko/.venvs/wm3d/bin}
PY=${PY:-$VENV_BIN/python}
TORCHRUN=${TORCHRUN:-$VENV_BIN/torchrun}
LOG_DIR=${LOG_DIR:-/data/Minko/logs}
FLOW_NAME=${FLOW_NAME:-oxe500m_staged_flow_after_stage_a_v1}
FLOW_LOG=${FLOW_LOG:-$LOG_DIR/${FLOW_NAME}.log}
FLOW_PIDFILE=${FLOW_PIDFILE:-$LOG_DIR/${FLOW_NAME}.pid}

STAGE_A_PIDFILE=${STAGE_A_PIDFILE:-$LOG_DIR/train_oxe_500m_stage_a_world_visual_hunyuan_oxeall_v1_8gpu.pid}
STAGE_A_CKPT=${STAGE_A_CKPT:-results/wm3d_v3_p64_500m_stage_a_world_visual_hunyuan_oxeall_v1_8gpu/ckpt/best.pt}
STAGE_A_CFG=${STAGE_A_CFG:-configs/v3_p64_500m_stage_a_world_visual_hunyuan_oxeall_v1_8gpu.yaml}

STAGE_B_CFG=${STAGE_B_CFG:-configs/v3_p64_500m_stage_b_oxe_joint_visual_proposer_v1_7gpu.yaml}
STAGE_B_ROOT=${STAGE_B_ROOT:-results/wm3d_v3_p64_500m_stage_b_oxe_joint_visual_proposer_v1_7gpu}
STAGE_B_LOG=${STAGE_B_LOG:-$LOG_DIR/train_oxe_500m_stage_b_oxe_joint_visual_proposer_v1_7gpu.log}

STAGE_C_CFG=${STAGE_C_CFG:-configs/v3_p64_500m_stage_c_oxe_direct_policy_v1_8gpu.yaml}
STAGE_C_ROOT=${STAGE_C_ROOT:-results/wm3d_v3_p64_500m_stage_c_oxe_direct_policy_v1_8gpu}
STAGE_C_LOG=${STAGE_C_LOG:-$LOG_DIR/train_oxe_500m_stage_c_oxe_direct_policy_v1_8gpu.log}

STAGE_D_CFG=${STAGE_D_CFG:-configs/libero_action_policy_partial4_stage_d_from_oxe500m_stage_c_v1_8gpu.yaml}
STAGE_D_ROOT=${STAGE_D_ROOT:-results/wm3d_libero_action_policy_partial4_stage_d_from_oxe500m_stage_c_v1_8gpu}
STAGE_D_LOG=${STAGE_D_LOG:-$LOG_DIR/train_libero_action_policy_partial4_stage_d_from_oxe500m_stage_c_v1_8gpu.log}
STAGE_D_ROLLOUT_LOG=${STAGE_D_ROLLOUT_LOG:-$LOG_DIR/libero_rollout_stage_d_best_hdf5init_task1_demo0.log}

TRAIN_GPUS_7=${TRAIN_GPUS_7:-0,1,2,3,4,5,6}
TRAIN_GPUS_8=${TRAIN_GPUS_8:-0,1,2,3,4,5,6,7}

mkdir -p "$LOG_DIR"

run_torch_stage() {
  local label="$1"
  local nproc="$2"
  local gpus="$3"
  local cfg="$4"
  local resume="$5"
  local log="$6"
  local root="$7"
  echo "[$(date -Is)] start_${label} cfg=$cfg resume=$resume root=$root"
  rm -rf "$root"
  WM3D_DDP_BACKEND=nccl \
  NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}" \
  NCCL_IB_HCA="${NCCL_IB_HCA:-^mlx5_bond_0}" \
  NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}" \
  NCCL_ASYNC_ERROR_HANDLING=1 \
  OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" \
  PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
  CUDA_VISIBLE_DEVICES="$gpus" \
  "$TORCHRUN" --standalone --nproc_per_node="$nproc" \
    -m wm3d_v3.training.train \
    --cfg "$cfg" \
    --resume "$resume" \
    --reset_optim \
    --print_every 25 \
    > "$log" 2>&1
  echo "[$(date -Is)] done_${label}"
}

run_canary() {
  local label="$1"
  local cfg="$2"
  local ckpt="$3"
  local out="$4"
  local log="$5"
  echo "[$(date -Is)] start_canary_${label} ckpt=$ckpt out=$out"
  CUDA_VISIBLE_DEVICES=7 \
  CFG="$cfg" \
  CKPT="$ckpt" \
  OUT_DIR="$out" \
  MAX_BATCHES=16 \
  N_GIFS=3 \
  N_HUNYUAN_GIFS=2 \
  scripts/run_generation_canary_v1.sh > "$log" 2>&1
  echo "[$(date -Is)] done_canary_${label}"
}

run_system_harness() {
  local label="$1"
  local cfg="$2"
  local ckpt="$3"
  local out="$4"
  local log="$5"
  echo "[$(date -Is)] start_system_harness_${label} ckpt=$ckpt out=$out"
  "$PY" -m wm3d_v3.eval.system_harness \
    --cfg "$cfg" \
    --ckpt "$ckpt" \
    --out_dir "$out" \
    --max_eval_batches 80 \
    --max_action_batches 80 \
    --max_ttc_batches 40 \
    --max_policy_batches 80 \
    --max_offline_replay_tasks 1 \
    --libero_trace_input results/wm3d_libero_action_policy_task1_teacher_intervention_suffix_v9/closed_loop_hdf5init_demo0.json \
    > "$log" 2>&1 || echo "[$(date -Is)] system_harness_${label}_failed status=$?"
}

check_stage_a_gate() {
  local eval_json="$1"
  "$PY" - "$eval_json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
    raise SystemExit(f"missing_stage_a_gate_eval={path}")
data = json.loads(path.read_text())
metrics = (data.get("metrics") or {}).get("ALL") or {}
thresholds = {
    "L_rgb_L1": 0.040,
    "L_rgb_lpips": 0.135,
    "L_depth_rel_L1": 0.080,
}
missing = [key for key in thresholds if key not in metrics]
if missing:
    raise SystemExit(f"missing_stage_a_gate_metrics={missing}")
failed = {
    key: {"value": float(metrics[key]), "threshold": limit}
    for key, limit in thresholds.items()
    if float(metrics[key]) > limit
}
print(json.dumps({
    "stage_a_gate": "pass" if not failed else "fail",
    "metrics": {key: float(metrics[key]) for key in thresholds},
    "thresholds": thresholds,
    "failed": failed,
}, sort_keys=True))
if failed:
    raise SystemExit(11)
PY
}

{
  echo "[$(date -Is)] flow_start"
  if [[ -f "$STAGE_A_PIDFILE" ]]; then
    stage_a_pid="$(cat "$STAGE_A_PIDFILE" || true)"
    if [[ -n "${stage_a_pid:-}" ]] && ps -p "$stage_a_pid" >/dev/null 2>&1; then
      echo "[$(date -Is)] waiting_stage_a_pid=$stage_a_pid"
      while ps -p "$stage_a_pid" >/dev/null 2>&1; do
        grep "\\[rank0\\] step" /data/Minko/logs/train_oxe_500m_stage_a_world_visual_hunyuan_oxeall_v1_8gpu.log | tail -n 1 || true
        sleep 300
      done
    fi
  fi
  if [[ ! -f "$STAGE_A_CKPT" ]]; then
    echo "missing_stage_a_ckpt=$STAGE_A_CKPT"
    exit 2
  fi

  STAGE_A_FINAL_CANARY=results/wm3d_v3_p64_500m_stage_a_world_visual_hunyuan_oxeall_v1_8gpu/canary_final
  run_canary stage_a_final "$STAGE_A_CFG" "$STAGE_A_CKPT" \
    "$STAGE_A_FINAL_CANARY" \
    "$LOG_DIR/canary_stage_a_final.log"
  check_stage_a_gate "$STAGE_A_FINAL_CANARY/eval_rgb_depth.json"

  run_torch_stage stage_b 7 "$TRAIN_GPUS_7" "$STAGE_B_CFG" "$STAGE_A_CKPT" "$STAGE_B_LOG" "$STAGE_B_ROOT"
  if [[ ! -f "$STAGE_B_ROOT/ckpt/best.pt" ]]; then
    echo "missing_stage_b_best=$STAGE_B_ROOT/ckpt/best.pt"
    exit 3
  fi
  run_canary stage_b "$STAGE_B_CFG" "$STAGE_B_ROOT/ckpt/best.pt" \
    "$STAGE_B_ROOT/canary_best" \
    "$LOG_DIR/canary_stage_b_best.log"
  run_system_harness stage_b "$STAGE_B_CFG" "$STAGE_B_ROOT/ckpt/best.pt" \
    "$STAGE_B_ROOT/system_harness_best" \
    "$LOG_DIR/system_harness_stage_b_best.log"

  run_torch_stage stage_c 8 "$TRAIN_GPUS_8" "$STAGE_C_CFG" "$STAGE_B_ROOT/ckpt/best.pt" "$STAGE_C_LOG" "$STAGE_C_ROOT"
  if [[ ! -f "$STAGE_C_ROOT/ckpt/best.pt" ]]; then
    echo "missing_stage_c_best=$STAGE_C_ROOT/ckpt/best.pt"
    exit 4
  fi

  echo "[$(date -Is)] start_stage_d cfg=$STAGE_D_CFG"
  rm -rf "$STAGE_D_ROOT"
  WM3D_DDP_BACKEND=nccl \
  NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}" \
  NCCL_IB_HCA="${NCCL_IB_HCA:-^mlx5_bond_0}" \
  NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}" \
  NCCL_ASYNC_ERROR_HANDLING=1 \
  OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" \
  PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
  CUDA_VISIBLE_DEVICES="$TRAIN_GPUS_8" \
  "$TORCHRUN" --standalone --nproc_per_node=8 \
    -m wm3d_v3.training.train_libero_action_policy \
    --cfg "$STAGE_D_CFG" \
    --print_every 25 \
    > "$STAGE_D_LOG" 2>&1
  echo "[$(date -Is)] done_stage_d"

  if [[ -f "$STAGE_D_ROOT/ckpt/best.pt" ]]; then
    echo "[$(date -Is)] final_stage_d_best=$STAGE_D_ROOT/ckpt/best.pt"
    echo "[$(date -Is)] start_stage_d_libero_rollout"
    CUDA_VISIBLE_DEVICES=0 \
    DEVICE=cuda:0 \
    CFG="$STAGE_C_CFG" \
    CKPT="$STAGE_D_ROOT/ckpt/best.pt" \
    SELECTION_MODE=direct \
    PORT=8785 \
    MAX_TASKS=1 \
    INIT_STATE_HDF5=/data/Minko/benchmarks/LIBERO/datasets/libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket_demo.hdf5 \
    INIT_STATE_DEMO_ID=demo_0 \
    MAX_STEPS=300 \
    CAMERA_SIZE=128 \
    CONTEXT_T=16 \
    WARMUP_STEPS=0 \
    GRIPPER_MODE=closed01_to_libero \
    SEND_LOWDIM=1 \
    ACTION_HISTORY_LEN=16 \
    OUT="$STAGE_D_ROOT/libero_remote_rollout_hdf5init_task1_demo0_best_300step.json" \
    SAVE_FRAMES_DIR="$STAGE_D_ROOT/hdf5init_task1_demo0_best_frames" \
    SAVE_FRAME_EVERY=25 \
    SERVER_LOG="$LOG_DIR/wm3d_policy_server_stage_d_task1_demo0_best.log" \
    bash scripts/run_libero_remote_smoke.sh > "$STAGE_D_ROLLOUT_LOG" 2>&1 \
      || echo "[$(date -Is)] stage_d_libero_rollout_failed status=$? log=$STAGE_D_ROLLOUT_LOG"
    echo "[$(date -Is)] done_stage_d_libero_rollout"
  else
    echo "missing_stage_d_best=$STAGE_D_ROOT/ckpt/best.pt"
    exit 5
  fi
  echo "[$(date -Is)] flow_done"
} >> "$FLOW_LOG" 2>&1 &

echo "$!" > "$FLOW_PIDFILE"
echo "started_flow_pid=$(cat "$FLOW_PIDFILE") log=$FLOW_LOG"
