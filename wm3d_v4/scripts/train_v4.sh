#!/usr/bin/env bash
cd /home/user01/Minko/newwm/wm3d_v4
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH=/home/user01/Minko/newwm/wm3d_v3:/home/user01/Minko/newwm/wm3d_v4
exec torchrun --standalone --nproc_per_node=4 -m wm3d_v4.training.train_v4 \
  --cfg "${1:-configs/v4_oxe.yaml}" \
  --v3_ckpt "${2:-/home/user01/Minko/newwm/results/wm3d_v3/ckpt/best.pt}" \
  "${@:3}"
