#!/usr/bin/env bash
set -euo pipefail

cd /data/Minko/world_model/wm3d_v5

RUN=/0604-10T-test/wm3d_v5/results/wm3d_v5_p64_1b_stage0_native3d_wan_vam_actionexpert_fsdp_jointpt_stage134_16gpu_20260625_1130
mkdir -p "$RUN/logs"

exec bash scripts/start_stage134_wan_vam_actionexpert_2node.sh
