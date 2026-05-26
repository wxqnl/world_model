#!/usr/bin/env bash
cd /home/user01/Minko/newwm/wm3d_v3
export CUDA_VISIBLE_DEVICES=2,3
exec torchrun --standalone --nproc_per_node=2 -m wm3d_v3.training.train_vla_c --cfg "${1:-configs/v3_vla_c.yaml}" "${@:2}"
