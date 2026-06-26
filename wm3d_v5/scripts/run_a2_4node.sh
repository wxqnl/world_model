#!/usr/bin/env bash
# WM3D-v5 LIBERO A2 (trunk-through) — 4-node / 32-GPU launch orchestrator.
# Run ON node43 (172.27.0.6 = rank0 / master). rank0 loads init_ckpt + writes ckpts.
set -uo pipefail
ROOT=/data/Minko/world_model/wm3d_v5
TORCHRUN=/data/Minko/.venvs/wm3d/bin/torchrun
CFG=configs/v5_p64_1b_libero_action_policy_sft_a2_4node_v1.yaml
MASTER=172.27.0.6
PORT=29511
LOGD=/data/Minko/logs/a2_4node
# node_rank -> ip   (rank0 must be this node = 172.27.0.6)
RANK_IPS=( "0:172.27.0.6" "1:172.27.0.4" "2:172.27.0.5" "3:172.27.0.7" )

ENVS="PYTHONPATH=$ROOT HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
WM3D_DDP_BACKEND=nccl USE_LIBUV=0 NCCL_NVLS_ENABLE=0 NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=bond0.1411 GLOO_SOCKET_IFNAME=bond0.1411 NCCL_DEBUG=WARN \
OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7"

RDZV_ID="a2run_$(date +%s)"
TR_ARGS="--nnodes=4 --nproc_per_node=8 --rdzv_backend=c10d --rdzv_endpoint=$MASTER:$PORT --rdzv_id=$RDZV_ID \
-m wm3d_v3.training.train_libero_action_policy --cfg $CFG --print_every 25"

do_launch() {
  for entry in "${RANK_IPS[@]}"; do
    r="${entry%%:*}"; ip="${entry##*:}"
    if [ "$r" = "0" ]; then
      pkill -9 -f bin/torchrun 2>/dev/null; pkill -9 -f wm3d_v3.training.train_libero 2>/dev/null; mkdir -p "$LOGD"
    else
      ssh -o StrictHostKeyChecking=no root@"$ip" "pkill -9 -f bin/torchrun 2>/dev/null; pkill -9 -f wm3d_v3.training.train_libero 2>/dev/null; mkdir -p $LOGD" </dev/null >/dev/null 2>&1 || true
    fi
  done
  sleep 5
  local sshpids=()
  for entry in "${RANK_IPS[@]}"; do
    r="${entry%%:*}"; ip="${entry##*:}"; LOG="$LOGD/train_rank${r}.log"
    if [ "$r" = "0" ]; then
      ( cd "$ROOT" && env $ENVS setsid "$TORCHRUN" $TR_ARGS </dev/null >"$LOG" 2>&1 & )
    else
      ssh -o StrictHostKeyChecking=no root@"$ip" "( cd $ROOT && env $ENVS setsid $TORCHRUN $TR_ARGS </dev/null >$LOG 2>&1 & ) ; exit 0" </dev/null >/dev/null 2>&1 &
      sshpids+=("$!")
    fi
    echo "launched rank$r ($ip) -> $LOG"
  done
  for p in "${sshpids[@]:-}"; do [ -n "${p:-}" ] && wait "$p" 2>/dev/null || true; done
  echo "ALL_LAUNCHED master=$MASTER:$PORT cfg=$CFG"
}

do_status() {
  for entry in "${RANK_IPS[@]}"; do
    r="${entry%%:*}"; ip="${entry##*:}"
    echo "===== rank$r ($ip) ====="
    if [ "$r" = "0" ]; then
      pgrep -af "train_libero_action_policy" | head -1 || echo "  (no proc)"
      tail -n 4 "$LOGD/train_rank0.log" 2>/dev/null | sed 's/^/  /'
    else
      ssh -o StrictHostKeyChecking=no root@"$ip" "pgrep -af train_libero_action_policy | head -1 || echo '  (no proc)'; tail -n 4 $LOGD/train_rank${r}.log 2>/dev/null" 2>/dev/null | sed 's/^/  /'
    fi
  done
}

do_stop() {
  for entry in "${RANK_IPS[@]}"; do
    r="${entry%%:*}"; ip="${entry##*:}"
    if [ "$r" = "0" ]; then pkill -9 -f bin/torchrun 2>/dev/null; pkill -9 -f wm3d_v3.training.train_libero 2>/dev/null || true
    else ssh -o StrictHostKeyChecking=no root@"$ip" "pkill -9 -f bin/torchrun 2>/dev/null; pkill -9 -f wm3d_v3.training.train_libero 2>/dev/null" </dev/null >/dev/null 2>&1 || true; fi
  done
  echo "STOPPED all ranks"
}

case "${1:-launch}" in
  launch) do_launch ;;
  status) do_status ;;
  stop)   do_stop ;;
  *) echo "usage: $0 [launch|status|stop]" >&2; exit 2 ;;
esac
