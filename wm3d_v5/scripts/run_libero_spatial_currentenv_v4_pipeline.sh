#!/usr/bin/env bash
set -euo pipefail

cd /data/Minko/world_model/wm3d_v5

ROOT="/data/Minko/world_model/wm3d_v5/results/wm3d_v5_p64_1b_libero_spatial_action_policy_currentenv_dense_unfreeze_sft_v4"
LOG_ROOT="/data/Minko/logs/libero_spatial_currentenv_v4"
CFG="configs/v5_p64_1b_libero_spatial_action_policy_currentenv_dense_unfreeze_sft_v4.yaml"

mkdir -p "${LOG_ROOT}"

CFG="${CFG}" \
ROOT="${ROOT}" \
LOG_ROOT="${LOG_ROOT}" \
RUN_ID="v4_train_$(date +%Y%m%d_%H%M%S)" \
MAX_STEPS=9000 \
FORCE_TRAIN=1 \
bash scripts/run_v5_libero_deployforward_unfreeze_sft_eval_v1.sh train

run_one() {
  local ckpt_name="$1"
  local port="$2"
  local run_id="v4_task1_init0_h8_${ckpt_name}_$(date +%Y%m%d_%H%M%S)"
  nohup bash -s > "${LOG_ROOT}/${run_id}.log" 2>&1 <<INNER
set -euo pipefail
cd /data/Minko/world_model/wm3d_v5
ROOT="${ROOT}"
LOG_ROOT="${LOG_ROOT}"
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 /data/Minko/.venvs/wm3d/bin/python -m wm3d_v3.policy.http_policy_server \\
  --cfg configs/v5_p64_1b_libero_action_policy_base_stage2p5_noprog_nohist_v1.yaml \\
  --ckpt "\${ROOT}/ckpt/${ckpt_name}.pt" --host 127.0.0.1 --port ${port} \\
  --device cuda:0 --qwen_device cuda:0 \\
  --task_cache_dir /0604-10T-test/wm3d_v5/cache/libero_taskemb_online \\
  --selection_mode direct > "\${LOG_ROOT}/v4_task1_${ckpt_name}_server.log" 2>&1 &
spid=\$!
trap 'kill \${spid} 2>/dev/null || true' EXIT
for _ in \$(seq 1 180); do curl -fsS http://127.0.0.1:${port}/health >/dev/null 2>&1 && break; sleep 2; done
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 MUJOCO_GL=egl LIBGL_DRIVERS_PATH=/usr/lib/x86_64-linux-gnu/dri/ __EGL_VENDOR_LIBRARY_FILENAMES=/data/Minko/egl/10_nvidia.json \\
  /data/Minko/.conda-envs/libero-py38/bin/python -m wm3d_v3.benchmarks.libero_remote_runner \\
  --libero_root /data/Minko/benchmarks/LIBERO --server_url http://127.0.0.1:${port} \\
  --suite libero_spatial --task_ids 1 --init_states 0 --max_steps 300 \\
  --camera_key agentview_image --camera_size 256 --context_T 16 --warmup_steps 5 \\
  --gripper_mode closed01_to_libero --exec_horizon 8 --send_lowdim --action_history_len 0 \\
  --use_policy_gripper_prob --gripper_closed_threshold 0.35 \\
  --out "\${ROOT}/debug_task1_init0_h8_prob035_${ckpt_name}_v1.json"
INNER
}

run_one best 10631
wait
run_one latest 10632
wait
