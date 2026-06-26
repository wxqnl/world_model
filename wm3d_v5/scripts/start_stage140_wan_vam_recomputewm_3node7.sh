#!/usr/bin/env bash
set -euo pipefail

cd /data/Minko/world_model/wm3d_v5

RUN=/0604-10T-test/wm3d_v5/results/wm3d_v5_p64_1b_stage0_native3d_wan_vam_recomputewm_policyfix_fsdp_jointpt_stage140_21gpu_no41gpu0_20260626_0320
mkdir -p "$RUN/logs"

ssh -f -n -o StrictHostKeyChecking=no 172.27.0.7 "cd /data/Minko/world_model/wm3d_v5 && nohup bash scripts/launch_stage140_wan_vam_recomputewm_3node7_node44.sh > '$RUN/logs/launch_node44.out' 2>&1 < /dev/null"
ssh -f -n -o StrictHostKeyChecking=no 172.27.0.4 "cd /data/Minko/world_model/wm3d_v5 && nohup bash scripts/launch_stage140_wan_vam_recomputewm_3node7_node41.sh > '$RUN/logs/launch_node41.out' 2>&1 < /dev/null"
nohup bash scripts/launch_stage140_wan_vam_recomputewm_3node7_node43.sh > "$RUN/logs/launch_node43.out" 2>&1 &

echo "$!" > "$RUN/logs/launcher_node43.pid"
echo "stage140 21gpu launched on nodes 43/44/41; run=$RUN"
