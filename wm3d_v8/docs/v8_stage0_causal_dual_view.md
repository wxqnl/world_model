# V8 Stage0 数据流水线

## 1. 不变量

每个样本包含 16 帧历史和 8 帧未来监督。缓存必须执行两个相互独立的 VGGT forward：

1. context forward 只读取历史 T16，产物进入模型输入；
2. target forward 读取 T16+K8，但只保留最后 K8 作为监督。

所有几何量都在第一帧观测相机坐标系中表达。target token 不得写回 context。模型训练时显式预测 RGB、depth、point 和 pose。

```text
schema                    wm3d_v8_stage0_causal_dual_view_v1
representation            wm3d_v8_vggt_observed_context_target_split_v1
context_future_leakage    false
target_usage              supervision_only
geometry_coordinate_frame first_observed_camera
T / P / D / K             16 / 64 / 2048 / 8
```

五源混合固定为：

| source | 每 100 optimizer steps |
|---|---:|
| OXE DROID | 35 |
| OXE Bridge | 15 |
| RoboCasa Atomic | 10 |
| RoboCasa Composite | 20 |
| RoboCasa MG | 20 |

## 2. 输入清单

开始缓存前先固定以下文件及 SHA256：

- DROID/Bridge source manifest；
- 两源 canonical action manifest、action audit gate、train-only stats；
- RoboCasa Atomic/Composite/MG factual manifest 与 action audit；
- 每个 RoboCasa 分区的 RGB sidecar index；
- PCA384 token codec 与 downstream gate；
- 本地 VGGT 权重快照。

所有正式输出必须写入新目录。生产器采用原子发布和 no-clobber：相同内容可重复核验，不同内容拒绝覆盖。

## 3. OXE causal cache

DROID、Bridge 的 train/val 分别执行。正式数据通过 `--num-shards` 与 `--shard-id` 覆盖完整选择集；`--max-windows` 只用于 canary。

```bash
export WM3D_VGGT_SOURCE_ROOT=/path/to/pinned/vggt
export HF_HOME=/path/to/hf-cache
export HF_HUB_OFFLINE=1

python scripts/cache_wm3d_v8_stage0_causal_dual_view_oxe.py   --manifest "$SOURCE_MANIFEST"   --cache-root "$SEALED_OXE_CACHE_ROOT"   --output-root "$V8_OXE_OUTPUT_ROOT"   --index "$V8_OXE_INDEX"   --codec "$PCA384_CODEC"   --codec-downstream-report "$CODEC_GATE"   --source oxe_droid_action   --split train   --num-shards "$NUM_SHARDS"   --shard-id "$SHARD_ID"   --device cuda:0
```

Bridge 使用 `--source oxe_bridge_action`。每个 shard 会生成 index 与 producer report；finalize 前必须验证所有 shard 的 selection/config SHA 一致。

## 4. RoboCasa causal cache

Atomic、Composite、MG 分开执行；`--v7-source` 必须与输入 manifest 分区一致。脚本名中的 `v7_compact` 是封存输入 ABI，不代表运行旧 V7 pipeline。

```bash
export QWEN3_VL_EMBEDDING_PATH=/path/to/pinned/qwen-embedding
export WM3D_VGGT_SOURCE_ROOT=/path/to/pinned/vggt
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python scripts/cache_robocasa365_v7_compact.py   --root "$ROBOCASA_PARTITION_ROOT"   --manifest "$FACTUAL_MANIFEST"   --action-audit "$FACTUAL_ACTION_AUDIT"   --rgb-sidecar-index "$RGB_SIDECAR_INDEX"   --codec "$PCA384_CODEC"   --codec-downstream-report "$CODEC_GATE"   --output-root "$V8_PARTITION_OUTPUT_ROOT"   --index "$V8_PARTITION_INDEX"   --causal-dual-view   --v7-source atomic   --num-shards "$NUM_SHARDS"   --shard-id "$SHARD_ID"   --device cuda:0
```

## 5. world16 finalize

所有 shard 完成后，用中央 finalizer 检查 shard closure、重复 identity、producer config/selection SHA 和全局计数，再发布合并 index：

```bash
WM3D_V8_STAGE0_WORLD16_FINALIZE=EXECUTE_WM3D_V8_STAGE0_WORLD16_FINALIZE_V1 python scripts/finalize_wm3d_v8_stage0_causal_dual_view_world16.py   --manifest-dir "$PRODUCER_REPORT_DIR"   --output-dir "$FINAL_INDEX_DIR"   --report "$FINALIZE_REPORT"   --mode execute
```

正式执行前先用 `--mode dry-run`。finalize 不重算 archive，也不覆盖不一致文件。

## 6. 20 Hz action-only sidecar

RoboCasa causal archive 仍保留既有视觉/几何数据；这里只新增真实 20 Hz action、valid mask、dt 和 action history。先 dry-run 全量复合审计，再去掉 `--dry-run` 发布。

```bash
python scripts/build_wm3d_v8_dual_rate_action_sidecars.py   --input-index "$COMBINED_ROBOCASA_INDEX"   --adapter-audit atomic="$ATOMIC_ACTION_AUDIT"   --adapter-audit composite="$COMPOSITE_ACTION_AUDIT"   --adapter-audit mg="$MG_ACTION_AUDIT"   --adapter-audit-sha256 atomic="$ATOMIC_ACTION_AUDIT_SHA256"   --adapter-audit-sha256 composite="$COMPOSITE_ACTION_AUDIT_SHA256"   --adapter-audit-sha256 mg="$MG_ACTION_AUDIT_SHA256"   --output-root "$ACTION20_ROOT"   --output-index "$ACTION20_INDEX"   --output-stats "$ACTION20_STATS"   --dry-run
```

OXE 没有被验证的 20 Hz 子步，因此不会伪造 fine label；它只使用真实 5 Hz coarse action 与 mask。

## 7. 生成 sealed runtime config

seal 会合并三个 RoboCasa index，并把 causal index、action sidecar index/stats 及其 SHA 全部写入 runtime overlay：

```bash
python scripts/seal_wm3d_v8_stage0_causal_dual_view_canary.py   --base-config configs/wm3d_v8_stage0_causal_dual_view_unified_action_canary_v2.yaml   --oxe-droid-train-index "$DROID_TRAIN_INDEX"   --oxe-droid-val-index "$DROID_VAL_INDEX"   --oxe-bridge-train-index "$BRIDGE_TRAIN_INDEX"   --oxe-bridge-val-index "$BRIDGE_VAL_INDEX"   --robocasa-atomic-index "$ATOMIC_INDEX"   --robocasa-composite-index "$COMPOSITE_INDEX"   --robocasa-mg-index "$MG_INDEX"   --combined-robocasa-index "$COMBINED_ROBOCASA_INDEX"   --action-sidecar-index "$ACTION20_INDEX"   --action-sidecar-stats "$ACTION20_STATS"   --runtime-config "$SEALED_RUNTIME_CONFIG"   --report "$SEAL_REPORT"   --max-steps 100   --initial-stop-step 20   --out-root "$CANARY_OUTPUT_ROOT"   --run-lineage "$CANARY_RUN_LINEAGE"
```

输出报告必须满足 `passed=true`、`launch_ready=true`、`errors=[]`、`blockers=[]`。两台机器还要分别对同一 runtime config 运行 full preflight，并比较 resolved config SHA。

## 8. Canary 与正式训练

Canary 固定两段：

```text
fresh 0→20
strict exact resume 20→100
```

两段使用同一 max_steps=100 调度、同一 output root 和同一 run lineage。step20/100 都必须是完整编号 checkpoint，包含 model、optimizer、scheduler、sampler、RNG 和 V8 action ABI。

审查顺序：

1. `review_wm3d_v8_stage0_causal_dual_view_canary.py` 固化训练与恢复证据；
2. `gate_wm3d_v8_stage0_causal_dual_view_canary.py` 发布 gate；
3. gate SHA 写入 formal sealed overlay；
4. 两机 full preflight；
5. 使用相同 resolved config 启动 formal world16。

任何阶段都不使用 `latest.pt` 作为 authority，也不自动跨 milestone 晋级。
