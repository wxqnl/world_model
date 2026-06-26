#!/usr/bin/env bash
set -euo pipefail

cd /data/Minko

export MASTER_ADDR=172.27.0.6
export MASTER_PORT=29699
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_NVLS_ENABLE=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1
export PYTHONPATH=/data/Minko/world_model/wm3d_v5:/data/Minko/external/HunyuanVideo:${PYTHONPATH:-}

RUN_DIR=/0604-10T-test/wm3d_v5/results/hunyuan_dit_control_v5_1b_oxe_stage98step800_rgbfeat_stage99_16gpu_20260618_0245
STAGE98=/0604-10T-test/wm3d_v5/results/hunyuan_dit_control_v5_1b_oxe_stage97step1000_signflip_stage98_16gpu_20260618_0215
mkdir -p "${RUN_DIR}/logs"

{
  echo "launch_time=$(date -Is)"
  echo "host=$(hostname)"
  echo "node_rank=0"
  echo "run_dir=${RUN_DIR}"
  echo "master=${MASTER_ADDR}:${MASTER_PORT}"
  echo "adapter_use_rough=0"
  echo "wm_pixel=1"
  echo "return_rgb_features=1"
  echo "path_type=context"
  echo "init_control_ckpt=${STAGE98}/ckpt/step_00000800.pt"
  echo "velocity_target_sign=-1.0"
  echo "rgb_feature_dim=576"
  echo "rgb_feature_gain=0.25"
  echo "latent_motion_mask_source=gt_rgb"
  echo "velocity_dynamic_weight=12 velocity_static_weight=1"
  echo "action_rank_weight=0"
  env | grep -E '^(CUDA_VISIBLE_DEVICES|NCCL_NVLS_ENABLE|PYTORCH_CUDA_ALLOC_CONF|OMP_NUM_THREADS|PYTHONPATH|MASTER_)=' | sort
} | tee "${RUN_DIR}/logs/node43_env.log"

/data/Minko/.venvs/wm3d/bin/torchrun \
  --nnodes=2 \
  --nproc_per_node=8 \
  --node_rank=0 \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  /data/Minko/world_model/wm3d_v5/scripts/train_hunyuan_dit_control_adapter.py \
  --wm_cfg /0604-10T-test/wm3d_v5/hunyuan_train/oxe_full_hunyuan_joint_v5_1b_stage2_full103965.yaml \
  --wm_ckpt /0604-10T-test/wm3d_v5/results/hunyuan_joint_v5_1b_oxe_stage25step500_latentscale118_hunyuan_owned_rgb_stage28_16gpu_20260615_1620/ckpt/wm_step_00000650.pt \
  --control_ckpt "${STAGE98}/ckpt/step_00000800.pt" \
  --out_dir "${RUN_DIR}" \
  --hunyuan_model_base /data/Minko/models/hunyuan_video \
  --epochs 3 \
  --max_steps 16000 \
  --batch_size_per_gpu 1 \
  --num_workers 2 \
  --lr 5e-6 \
  --weight_decay 0.01 \
  --warmup_steps 200 \
  --grad_clip 1.0 \
  --max_train_windows 0 \
  --max_val_windows 1024 \
  --eval_batches 8 \
  --eval_every_steps 100 \
  --ckpt_every_steps 100 \
  --keep_last_checkpoints 3 \
  --milestone_every_steps 1000 \
  --print_every 10 \
  --path_type context \
  --velocity_mse_weight 1.0 \
  --velocity_l1_weight 0.05 \
  --velocity_target_sign -1.0 \
  --latent_motion_mask_source gt_rgb \
  --latent_motion_threshold 0.02 \
  --latent_motion_dilate 1 \
  --velocity_dynamic_weight 12.0 \
  --velocity_static_weight 1.0 \
  --action_rank_weight 0.0 \
  --use_block_action_film \
  --block_action_film_scale 1.0 \
  --block_action_film_hidden 192 \
  --control_scale 1.0 \
  --use_rgb_features \
  --rgb_feature_dim 576 \
  --rgb_feature_gain 0.25 \
  2>&1 | tee -a "${RUN_DIR}/logs/node43_train.log"
