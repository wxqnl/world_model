# WM3D-V7 Stage1-P 原生 3D 规划阶段

这个目录只实现 V7 的规划阶段，不替换 Stage0，也不改变 Stage0 的 action serving owner。

## 不变量

- Stage0 的 direct pose head 与 delta-composed gripper head保持冻结，仍是可执行 action 的来源。
- flow head只生成 6D pose 候选；所有 flow 候选的 gripper 仍来自 direct event head。
- WM3D core保持 `T16 / P64 / D2048 / K8`，通过 4 个原生 K8 chunk 自回归到 H32。
- T16 是当前固定 RoboCasa runtime 中真实回放得到的 16 帧历史，不用重复 root 图像代替；最后一帧与所有候选的 root state/RGB 必须逐字节一致。
- 每个候选都预测显式 native token、depth、point、pose；planner 只读取这些未来和 task embedding，不读取候选 action。
- planner 的梯度在 imagined evidence 处截断。world model 只能通过真实同根分支的 token/depth/point/pose 重建损失更新。
- 数据标签来自固定 RoboCasa runtime 中的真实同根执行；不接受伪 outcome、复制 factual geometry 或未来 observation。

候选顺序固定为：

1. `factual_teacher`（只用于 dynamics 监督，不能被选择）
2. `direct`
3. `flow_0..3`
4. `grip_open / grip_close`
5. `arm_hold / pose_reverse / pose_half`

## 前置条件

必须先等 Stage0 自然完成并封存：

```text
/data/Minko/world_model/wm3d_v7_actionrepair1b_20260806/results/
wm3d_v7_1b_native_actionpolicy_joint_formal100k_3node24_v3/ckpt/step_00100000.pt
```

把该文件的 SHA256 写入三个 Stage1 配置中的 `source_checkpoint_sha256`。在此之前，预检必然失败，Stage1 不会启动。

## 数据生成

以下命令均从项目根目录运行，`PY=/data/Minko/.venvs/wm3d/bin/python`。多 GPU 时每个进程使用唯一的 `SHARD` 和独立 index 文件；最后用 seal 脚本合并，禁止直接 `cat`。

### 1. 缓存严格因果的真实 T16 root context

只回放到 `t0`，按 20 Hz 原生步长每 4 步采一帧；`t0` 之后的 simulator step 不会执行。少于 16 帧历史的 root 会被明确排除。

```bash
CUDA_VISIBLE_DEVICES=${GPU} QWEN3_VL_EMBEDDING_PATH=${QWEN_EMBED_PATH} ${PY} \
  scripts/cache_v7_stage1_planner_root_contexts.py \
  --legacy-branch-index /data/Minko/world_model/wm3d_v7/manifests/robocasa_same_root_branch_compact_expand_h32_k5_grip_r3_120ep_8gpu_v4/index.jsonl \
  --raw-manifest /data/Minko/world_model/wm3d_v7/manifests/robocasa_same_root_cf_expand_h32_k5_grip_r3_120ep_8gpu_v4.jsonl \
  --source-audit /data/Minko/world_model/wm3d_v7/manifests/audits/robocasa_atomic_sim_sources.json \
  --action-audit /data/Minko/world_model/wm3d_v7/manifests/audits/robocasa365_atomic_factual_action_v2.json \
  --codec /data/Minko/world_model/wm3d_v7/manifests/token_codec/pca384_int8_strict_v2.pt \
  --codec-downstream-report /data/Minko/world_model/wm3d_v7/manifests/token_codec/pca384_int8_strict_v2_downstream_droid7.json \
  --output-root /data/Minko/world_model/wm3d_v7/manifests/robocasa_stage1_planner_root_context_v1 \
  --output-index /data/Minko/world_model/wm3d_v7/manifests/robocasa_stage1_planner_root_context_v1/index.shard-${SHARD}.jsonl \
  --num-shards ${SHARDS} --shard-index ${SHARD} --device cuda:0
```

### 2. 从封存 Stage0 生成 action 候选

```bash
CUDA_VISIBLE_DEVICES=${GPU} ${PY} scripts/harvest_wm3d_v7_stage1_planner_candidates.py \
  --model-config configs/wm3d_v7_1b_native_actionpolicy_joint_formal100k_3node24_v3.yaml \
  --checkpoint results/wm3d_v7_1b_native_actionpolicy_joint_formal100k_3node24_v3/ckpt/step_00100000.pt \
  --root-context-index /data/Minko/world_model/wm3d_v7/manifests/robocasa_stage1_planner_root_context_v1/index.shard-${SHARD}.jsonl \
  --action-stats /data/Minko/world_model/wm3d_v7/manifests/robocasa365_stage0_full_v1/action_stats_train.npz \
  --output-root /data/Minko/world_model/wm3d_v7/manifests/robocasa_stage1_planner_candidates_v1 \
  --output-index /data/Minko/world_model/wm3d_v7/manifests/robocasa_stage1_planner_candidates_v1/index.shard-${SHARD}.jsonl \
  --device cuda:0
```

### 3. 在固定 RoboCasa runtime 中执行所有候选

每个 candidate shard单独执行；不能把旧 K8 outcome 当成 H32 标签。

```bash
${PY} scripts/generate_robocasa_stage1_planner_branches.py \
  --candidate-index /data/Minko/world_model/wm3d_v7/manifests/robocasa_stage1_planner_candidates_v1/index.shard-${SHARD}.jsonl \
  --action-audit /data/Minko/world_model/wm3d_v7/manifests/audits/robocasa365_atomic_factual_action_v2.json \
  --output-root /data/Minko/world_model/wm3d_v7/manifests/robocasa_stage1_planner_runtime_v1 \
  --output-index /data/Minko/world_model/wm3d_v7/manifests/robocasa_stage1_planner_runtime_v1/index.shard-${SHARD}.jsonl
```

### 4. 对每个真实分支缓存原生 3D 证据

一个分支的 root + H32 必须在同一次 33-frame VGGT 调用中，保持同一 gauge。

```bash
CUDA_VISIBLE_DEVICES=${GPU} ${PY} scripts/cache_v7_stage1_planner_branches.py \
  --runtime-index /data/Minko/world_model/wm3d_v7/manifests/robocasa_stage1_planner_runtime_v1/index.shard-${SHARD}.jsonl \
  --codec /data/Minko/world_model/wm3d_v7/manifests/token_codec/pca384_int8_strict_v2.pt \
  --codec-downstream-report /data/Minko/world_model/wm3d_v7/manifests/token_codec/pca384_int8_strict_v2_downstream_droid7.json \
  --output-root /data/Minko/world_model/wm3d_v7/manifests/robocasa_stage1_planner_h32_v2 \
  --output-index /data/Minko/world_model/wm3d_v7/manifests/robocasa_stage1_planner_h32_v2/index.shard-${SHARD}.jsonl \
  --batch-frames 33 --device cuda:0
```

### 5. 合并与封存

```bash
${PY} scripts/seal_wm3d_v7_stage1_planner_indices.py \
  --kind cache \
  $(printf -- '--input %q ' /data/Minko/world_model/wm3d_v7/manifests/robocasa_stage1_planner_h32_v2/index.shard-*.jsonl) \
  --output /data/Minko/world_model/wm3d_v7/manifests/robocasa_stage1_planner_h32_v2/index.jsonl
sha256sum /data/Minko/world_model/wm3d_v7/manifests/robocasa_stage1_planner_h32_v2/index.jsonl
```

将最终 index SHA256 写入三个配置中的 `planner_data.index_sha256`。

## 三阶段训练

每阶段都先运行 `--mode data`，再在每台节点运行 `--mode train`。节点映射保持与 Stage0 正式 IB 配置一致：node44 rank0、node41 rank1、node43 rank2。

1. `configs/wm3d_v7_stage1_planner_dynamics10k.yaml`：只校准原生 3D action dynamics。
2. `configs/wm3d_v7_stage1_planner_planner10k.yaml`：冻结 world，只训练 planner；需先填入 dynamics endpoint SHA。
3. `configs/wm3d_v7_stage1_planner_joint5k.yaml`：低学习率联合校准；需先填入 planner endpoint SHA。

预检：

```bash
${PY} scripts/preflight_wm3d_v7_stage1_planner.py \
  --cfg configs/wm3d_v7_stage1_planner_dynamics10k.yaml --mode data
```

显式启动单个阶段，不会自动串行进入下一阶段：

```bash
WM3D_V7_STAGE1_PLANNER_CONFIRM=EXECUTE_WM3D_V7_STAGE1_PLANNER_PHASE \
  scripts/launch_wm3d_v7_stage1_planner.sh \
  configs/wm3d_v7_stage1_planner_dynamics10k.yaml ${NODE_RANK}
```

断点恢复必须同时给出完整编号 checkpoint 和 SHA：

```bash
WM3D_V7_STAGE1_RESUME=/abs/path/step_00005000.pt \
WM3D_V7_STAGE1_RESUME_SHA256=<sha256> \
WM3D_V7_STAGE1_PLANNER_CONFIRM=EXECUTE_WM3D_V7_STAGE1_PLANNER_PHASE \
  scripts/launch_wm3d_v7_stage1_planner.sh <config> ${NODE_RANK}
```

## 评测与晋级

离线评测同时计算 H8/H16/H32 action-effect、显式几何误差、true-future 与 imagined-future 的选择能力：

```bash
${PY} scripts/eval_wm3d_v7_stage1_planner.py \
  --cfg <phase-config> --overlay <numbered-overlay.pt> \
  --overlay-sha256 <sha256> --split test --output <offline-report.json>
```

joint 阶段还必须运行配对、真实 RoboCasa 闭环对照：

```bash
${PY} scripts/eval_robocasa_wm3d_v7_stage1_planner_closed_loop.py \
  --cfg configs/wm3d_v7_stage1_planner_joint5k.yaml \
  --overlay <joint-step_00005000.pt> --overlay-sha256 <sha256> \
  --split test --max-roots 100 --output <closed-loop-report.json>
```

最后运行 gate。固定门槛包括 H8 effect lower95 > 0.10、H32 lower95 > 0、true-future AUC >= 0.80、imagined uplift保留率 >= 70%、mixed-root success@1 提升 >= 15pp、candidate oracle提升 >= 10pp，以及 joint 闭环相对 Stage0 提升 >= 5pp。gate 通过也不会自动启动下一阶段。
