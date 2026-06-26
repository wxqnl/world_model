#!/usr/bin/env bash
set -euo pipefail

cd /data/Minko/world_model/wm3d_v5

ROOT="/0604-10T-test/wm3d_v5/results/wm3d_v5_p64_1b_libero_spatial_fastwam_dualcam_concat_padded_earlyw_rgb_rawxyz_lastobs_k32_sft_v1"
LOG_ROOT="/0604-10T-test/wm3d_v5/logs/libero_spatial_fastwam_dualcam_concat_padded_earlyw_rgb_rawxyz_lastobs_k32_v1"
CFG="configs/v5_p64_1b_libero_spatial_fastwam_dualcam_concat_padded_earlyw_rgb_rawxyz_lastobs_k32_sft_v1.yaml"
BASE_CFG="configs/v5_p64_1b_libero_action_policy_fastwam_spatial_dualcam_concat_padded_earlyw_rgb_lastobs_k32_base_v1.yaml"
CACHE_ROOT="/0604-10T-test/wm3d_v5/cache/libero_action_policy_spatial_dualcam_concat_padded_lastobs_T16_k32_s4_rot_v1"
CACHE_MANIFEST="/0604-10T-test/wm3d_v5/manifests/libero_action_policy_spatial_dualcam_concat_padded_lastobs_T16_k32_s4_rot_v1.jsonl"
CAMERA_FUSION="concat"

mkdir -p "${LOG_ROOT}" "${ROOT}"

run_cache() {
  if [[ -s "${CACHE_MANIFEST}" ]]; then
    echo "[pipeline] concat cache manifest exists: ${CACHE_MANIFEST}"
    return
  fi
  echo "[pipeline] building dualcam concat padded spatial k32 cache"
  OUT_ROOT="${CACHE_ROOT}" \
  MANIFEST_OUT="${CACHE_MANIFEST}" \
  SOURCE_JSONL="${CACHE_ROOT}/source_windows.jsonl" \
  SUMMARY_JSON="${CACHE_ROOT}/source_windows_summary.json" \
  SPLIT_DIR="${CACHE_ROOT}/splits" \
  CAMERA_FUSION="${CAMERA_FUSION}" \
  TARGET_OFFSET=-1 \
  PAD_EPISODE_START=1 \
  GPUS="${GPUS:-0,1,2,3,4,5,6,7}" \
  bash scripts/build_v5_libero_action_policy_cache_spatial_dualcam_k32_v1.sh all \
    > "${LOG_ROOT}/cache_$(date +%Y%m%d_%H%M%S).log" 2>&1
}

run_train() {
  CFG="${CFG}" \
  BASE_CFG="${BASE_CFG}" \
  ROOT="${ROOT}" \
  LOG_ROOT="${LOG_ROOT}" \
  RUN_ID="dualcam_concat_padded_earlyw_rgb_rawxyz_lastobs_k32_train_$(date +%Y%m%d_%H%M%S)" \
  MAX_STEPS="${MAX_STEPS:-12000}" \
  GPUS="${GPUS:-0,1,2,3,4,5,6,7}" \
  NPROC="${NPROC:-8}" \
  FORCE_TRAIN="${FORCE_TRAIN:-1}" \
  PRINT_EVERY="${PRINT_EVERY:-25}" \
  bash scripts/run_v5_libero_deployforward_unfreeze_sft_eval_v1.sh train
}

run_task1_smoke() {
  local ckpt_name="${1:-best}"
  local port="${2:-10757}"
  local exec_horizon="${EXEC_HORIZON:-10}"
  local warmup_steps="${WARMUP_STEPS:-30}"
  local max_pose_norm="${MAX_POSE_NORM:-0}"
  local run_id="dualcam_concat_padded_earlyw_rgb_rawxyz_lastobs_k32_task1_init0_h${exec_horizon}_w${warmup_steps}_mp${max_pose_norm}_${ckpt_name}_$(date +%Y%m%d_%H%M%S)"
  nohup bash -s > "${LOG_ROOT}/${run_id}.log" 2>&1 <<INNER
set -euo pipefail
cd /data/Minko/world_model/wm3d_v5
ROOT="${ROOT}"
LOG_ROOT="${LOG_ROOT}"
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 /data/Minko/.venvs/wm3d/bin/python -m wm3d_v3.policy.http_policy_server \\
  --cfg "${BASE_CFG}" \\
  --ckpt "\${ROOT}/ckpt/${ckpt_name}.pt" --host 127.0.0.1 --port ${port} \\
  --device cuda:0 --qwen_device cuda:0 \\
  --camera_fusion "${CAMERA_FUSION}" \\
  --task_cache_dir /0604-10T-test/wm3d_v5/cache/libero_taskemb_online \\
  --selection_mode direct > "\${LOG_ROOT}/dualcam_concat_padded_earlyw_rgb_rawxyz_lastobs_k32_task1_${ckpt_name}_server.log" 2>&1 &
spid=\$!
trap 'kill \${spid} 2>/dev/null || true' EXIT
for _ in \$(seq 1 240); do curl -fsS http://127.0.0.1:${port}/health >/dev/null 2>&1 && break; sleep 2; done
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 MUJOCO_GL=egl LIBGL_DRIVERS_PATH=/usr/lib/x86_64-linux-gnu/dri/ __EGL_VENDOR_LIBRARY_FILENAMES=/data/Minko/egl/10_nvidia.json \\
  /data/Minko/.conda-envs/libero-py38/bin/python -m wm3d_v3.benchmarks.libero_remote_runner \\
  --libero_root /data/Minko/benchmarks/LIBERO --server_url http://127.0.0.1:${port} \\
  --suite libero_spatial --task_ids 1 --init_states 0 --max_steps 400 \\
  --camera_keys agentview_image,robot0_eye_in_hand_image --rotate_180 --camera_size 256 --context_T 16 --warmup_steps ${warmup_steps} \\
  --gripper_mode closed01_to_libero --exec_horizon ${exec_horizon} --max_pose_norm ${max_pose_norm} --send_lowdim --action_history_len 0 \\
  --out "\${ROOT}/debug_task1_init0_dualcam_concat_padded_earlyw_rgb_rawxyz_lastobs_k32_h${exec_horizon}_w${warmup_steps}_mp${max_pose_norm}_${ckpt_name}_v1.json"
INNER
}

run_full_eval() {
  CFG="${CFG}" \
  BASE_CFG="${BASE_CFG}" \
  ROOT="${ROOT}" \
  LOG_ROOT="${LOG_ROOT}" \
  RUN_ID="dualcam_concat_padded_earlyw_rgb_rawxyz_lastobs_k32_eval_$(date +%Y%m%d_%H%M%S)" \
  GPUS="${GPUS:-0,1,2,3,4,5,6,7}" \
  BASE_PORT="${BASE_PORT:-10761}" \
  EXEC_HORIZON="${EXEC_HORIZON:-10}" \
  EVAL_MAX_STEPS="${EVAL_MAX_STEPS:-400}" \
  CAMERA_KEYS="agentview_image,robot0_eye_in_hand_image" \
  CAMERA_FUSION="${CAMERA_FUSION}" \
  ROTATE_180=1 \
  CAMERA_SIZE=256 \
  CONTEXT_T=16 \
  FORCE_TRAIN=0 \
  bash scripts/run_v5_libero_deployforward_unfreeze_sft_eval_v1.sh eval
}

case "${1:-all}" in
  cache) run_cache ;;
  train) run_train ;;
  smoke) run_task1_smoke best 10757 ;;
  eval) run_full_eval ;;
  all)
    run_cache
    run_train
    run_task1_smoke best 10757
    run_task1_smoke latest 10758
    run_full_eval
    ;;
  *)
    echo "usage: $0 [cache|train|smoke|eval|all]" >&2
    exit 2
    ;;
esac
