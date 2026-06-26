#!/usr/bin/env bash
set -euo pipefail

cd /data/Minko

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_NVLS_ENABLE=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8
export PYTHONPATH=/data/Minko/world_model/wm3d_v5

source /data/Minko/.venvs/wm3d/bin/activate

RUN_DIR=${RUN_DIR:-/0604-10T-test/wm3d_v5/results/hunyuan_latent_decoder_v5_1b_oxe_stage107c_contextpixel_teacher_stage108b_16gpu_20260618_1510}
WM_CFG=${WM_CFG:-/data/Minko/world_model/wm3d_v5/configs/v5_p64_1b_oxe_context_pixel_scaffold_actionrank_stage107c.yaml}
WM_CKPT=${WM_CKPT:-/0604-10T-test/wm3d_v5/results/wm3d_v5_p64_1b_oxe_context_pixel_scaffold_actionrank_stage107c_16gpu_20260618_1430/ckpt/step_00002500.pt}
INIT_CKPT=${INIT_CKPT:-/0604-10T-test/wm3d_v5/results/hunyuan_latent_decoder_v5_1b_oxe_stage75best_bgprotect_teacher_stage106b_16gpu_20260618_1250/ckpt/best.pt}
mkdir -p "${RUN_DIR}/logs"

{
  echo "launch_time=$(date -Is)"
  echo "host=$(hostname)"
  echo "node_rank=0"
  echo "run_dir=${RUN_DIR}"
  echo "wm_cfg=${WM_CFG}"
  echo "wm_ckpt=${WM_CKPT}"
  echo "init_ckpt=${INIT_CKPT}"
  echo "wm_pixel=1"
  echo "adapter_use_rgb_features=1 dim=96 gain=0.35"
  echo "counterfactual_teacher=alternate delta=35 temporal=15"
  env | grep -E '^(CUDA_VISIBLE_DEVICES|NCCL_NVLS_ENABLE|PYTORCH_CUDA_ALLOC_CONF|OMP_NUM_THREADS|PYTHONPATH)=' | sort
} | tee "${RUN_DIR}/logs/node43_env.log"

torchrun \
  --nnodes=2 \
  --nproc_per_node=8 \
  --node_rank=0 \
  --master_addr=172.27.0.6 \
  --master_port=29636 \
  world_model/wm3d_v5/scripts/train_hunyuan_latent_adapter_stage37.py \
  --wm_cfg "${WM_CFG}" \
  --wm_ckpt "${WM_CKPT}" \
  --adapter_init_ckpt "${INIT_CKPT}" \
  --out_dir "${RUN_DIR}" \
  --hunyuan_repo /data/Minko/external/HunyuanVideo \
  --hunyuan_model_base /data/Minko/models/hunyuan_video \
  --vae_precision fp32 \
  --precision bf16 \
  --vae_trainable_prefixes decoder.up_blocks.3,decoder.conv_norm_out,decoder.conv_out \
  --vae_lr 8e-08 \
  --vae_weight_decay 0.0 \
  --epochs 8 \
  --max_steps 16000 \
  --batch_size_per_gpu 2 \
  --num_workers 2 \
  --lr 2e-06 \
  --weight_decay 0.01 \
  --warmup_steps 160 \
  --grad_clip 1.0 \
  --latent_mse_weight 0.02 \
  --latent_l1_weight 0.002 \
  --latent_dynamic_l1_weight 1.8 \
  --latent_static_l1_weight 0.25 \
  --latent_delta_l1_weight 1.8 \
  --latent_delta_temporal_l1_weight 14.0 \
  --latent_delta_from_first_l1_weight 28.0 \
  --mask_l1_weight 0.2 \
  --mask_bce_weight 0.35 \
  --mask_area_weight 0.001 \
  --mask_tv_weight 0.01 \
  --decoded_l1_weight 0.02 \
  --decoded_motion_l1_weight 95.0 \
  --decoded_temporal_l1_weight 30.0 \
  --decoded_motion_mag_l1_weight 120.0 \
  --decoded_motion_mag_threshold 0.004 \
  --decoded_from_first_l1_weight 180.0 \
  --decoded_static_l1_weight 25.0 \
  --decoded_static_threshold 0.025 \
  --decoded_raw_motion_l1_weight 760.0 \
  --decoded_raw_temporal_l1_weight 260.0 \
  --decoded_raw_motion_mag_l1_weight 860.0 \
  --decoded_raw_from_first_l1_weight 900.0 \
  --decoded_raw_static_l1_weight 120.0 \
  --decoded_raw_losses_roi \
  --counterfactual_action_mode alternate \
  --counterfactual_every 1 \
  --counterfactual_latent_rank_weight 0.0 \
  --counterfactual_latent_separation_weight 0.0 \
  --counterfactual_teacher_delta_weight 35.0 \
  --counterfactual_teacher_temporal_weight 15.0 \
  --counterfactual_teacher_delta_threshold 0.006 \
  --decoded_motion_mask_threshold 0.018 \
  --decoded_motion_mask_spatial_dilate 5 \
  --decoded_motion_mask_temporal_dilate 1 \
  --decoded_motion_mask_floor 0.02 \
  --max_train_windows 0 \
  --max_val_windows 1024 \
  --eval_batches 16 \
  --eval_every_steps 250 \
  --ckpt_every_steps 250 \
  --keep_last_checkpoints 2 \
  --milestone_every_steps 4000 \
  --print_every 25 \
  --seed 108 \
  --ddp_static_graph \
  --hidden 256 \
  --n_blocks 6 \
  --output_mode direct_temporal_delta_bgprotect \
  --mask_motion_threshold 0.006 \
  --mask_motion_softness 0.02 \
  --latent_motion_mask_topk 0.0 \
  --latent_motion_mask_floor 0.0 \
  --mask_bias_init -1.2 \
  --mask_temperature 0.85 \
  --mask_min 0.0 \
  --mask_max 1.0 \
  --adapter_use_motion \
  --adapter_use_temporal_memory \
  --adapter_temporal_memory_heads 4 \
  --adapter_temporal_memory_layers 2 \
  --adapter_temporal_memory_mlp_mult 2.0 \
  --adapter_temporal_memory_gate_init 0.35 \
  --adapter_use_rgb_features \
  --adapter_rgb_feature_dim 96 \
  --adapter_rgb_feature_gain 0.35 \
  --motion_region_threshold 0.08 \
  --motion_region_softness 0.04 \
  --motion_region_power 0.8 \
  --motion_region_dilate 2 \
  --motion_region_temporal_dilate 1 \
  --motion_region_topk 0.35 \
  --motion_region_floor 0.0 \
  --motion_region_prior_weight 1.0 \
  --motion_region_bg_ceiling 0.1 \
  --motion_region_mask_mode prior \
  --direct_delta_static_center_weight 0.3 \
  --direct_delta_temporal_center_weight 0.15 \
  --direct_delta_spatial_highpass_weight 0.05 \
  --direct_delta_spatial_highpass_kernel 5 \
  --direct_delta_static_floor 0.65 \
  --mask_target_source dynamic_or_region \
  --base_latents_source context \
  --latent_temporal_delta_scale 1.0 \
  --wm_pixel \
  --no_adapter_use_rough
