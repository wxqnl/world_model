# V8 Stage0 因果双视图原生 3D 设计

## 1. 目标

修复 V8 Stage0 旧 VGGT 整窗缓存把未来 `K` 帧混入 observed context 表示的问题，同时保持以下能力不变：

- WM3D 原生 3D 主干与 `T16 / P64 / D2048 / K8` 接口；
- 显式 RGB、depth、point、camera/pose 与 future-token 监督；
- 五源 factual action conditioning；
- direct pose policy、delta-composed gripper、pose-only flow auxiliary；
- native action/no-teacher 与 native future anchor；
- 训练和 serving 都不把未来 observation 作为输入。

本设计创建新的 Stage0 lineage。旧 Stage0 checkpoint、旧 causal diagnostic checkpoint、旧 cache 与旧结果均只读，不作为 exact resume 或 warm start。

## 2. 已确认的问题

旧 OXE window cache 与 RoboCasa compact cache 会把 `T+K` 或更长 temporal segment 一次送入 VGGT。VGGT temporal aggregator 是双向的，因此 observed token 会依赖未来帧。direct action head 虽然只读取 WM3D `core_pred`，但其 context 已经间接携带未来 observation。

现有逐帧 causal 原型虽然消除了泄漏，却失去 VGGT 的跨帧几何 gauge；它还绑定了弱化的 dynamics 配方，没有训练完整 direct/flow policy，因此不能直接合入 V8。

## 3. 因果双视图

每个训练窗口执行两个用途严格分离的 VGGT forward：

1. Observed-context forward：只输入 `[start, start + T)`，产出 `context_pooled`，只供 `s_in/s_wrist`。
2. Future-target forward：输入 `[start, start + T + K)`，只保留最后 `K` 帧的 pooled、depth、point、pose 与 confidence；前 `T` 帧输出必须丢弃。

两次 forward 以同一个首个 observed frame 为 camera gauge anchor。缓存声明：

```text
schema                     = wm3d_v8_stage0_causal_dual_view_v1
representation_contract   = wm3d_v8_vggt_observed_context_target_split_v1
context_future_leakage     = false
context_frame_range        = [start, start + T)
target_frame_range         = [start + T, start + T + K)
target_usage               = supervision_only
geometry_coordinate_frame  = first_observed_camera
```

这仍是原生 3D：模型预测和损失直接作用于显式 point/depth/pose，不用 latent-3D 代替真实 3D 输出。

## 4. Cache 布局

### 4.1 OXE

每个窗口 NPZ 至少包含 `context_pooled [T,P,D]`、`future_pooled [K,P,D]`、future depth/depth_conf、point/point_conf、pose/pose_conf，以及 clip/start/T/K 和因果合同元数据。

canonical action、action stats、RGB 与 task embedding 继续使用原有封存资产；新窗口 cache 不改变 action 语义。

### 4.2 RoboCasa compact

仍按 clip 保存一个原子 NPZ，避免把约一百万个窗口展开成一百万个小文件。文件内增加 window 维字段：

```text
window_starts
anchor_context_codes / anchor_context_scale
wrist_context_codes  / wrist_context_scale
anchor_future_codes  / anchor_future_scale
future_depth/depth_conf/point/point_conf/pose/pose_conf
```

窗口索引必须精确绑定 `clip_hash + start + T + K`。dataset 禁止从旧 clip-level `anchor_codes` 切片冒充新 context。

## 5. Stage0 objective 不变量

新配置继承当前 V8 正式 action-policy 配方，并由静态测试锁定：

```text
model.enable_action_policy                 = true
model.policy_enable_flow_head              = true
model.policy_flow_use_as_policy            = false
model.policy_grip_owner                    = delta_composed
train.joint_native_action_pretraining      = true
train.direct_policy_weight                 = 1.0
train.policy_flow_weight                   = 0.25
train.native_action_no_teacher_weight      = 0.15
train.native_action_no_teacher_start_step  = 0
train.native_action_no_teacher_every       = 1
train.native_future_no_teacher_weight      = 0.20
train.factual_action_conditioning.start_step = 0
```

五源 mix、canonical signed-close action、pose normalization、direct/flow 分工与 RGB/depth/point loss 不得被 causal cache 覆盖。

## 6. Fail-closed 边界

以下任一情况必须在 preflight 或 dataset 首次读取时失败：

- 缺少新 schema 或 representation contract；
- context/target frame range 与 `start/T/K` 不一致；
- `context_future_leakage` 不是显式 `false`；
- `target_usage` 不是 `supervision_only`；
- clip/start/gauge 身份不一致；
- 使用旧整窗 pooled 字段冒充 context；
- action-policy 关键权重与正式配方不同；
- 从旧 checkpoint warm start 或 cross-lineage resume。

## 7. 实施隔离

- 开发分支：`codex/v8-stage0-causal-dualview`；
- 使用新 cache schema、新配置、新输出目录和新 run lineage；
- 不覆盖现有正式配置、cache、checkpoint 或 result；
- 全部验收通过后才合入正式 `v8`。

## 8. 验收门槛

### 8.1 单元与合同测试

- fake future-mixing encoder：只改变 K 帧时，`context_pooled` 必须逐元素不变，而 future target 必须变化；
- legacy cache、错 clip/start/T/K、错 gauge、target 回流均被拒绝；
- resolved config 的 action objective 与当前正式配方一致；
- signed gripper 不被提前改写成错误的 close01 dynamics input；
- RGB/depth/point/pose shape 与 finite contract 通过；
- V8 原有 70 个测试继续通过。

### 8.2 五源数据 canary

- 每个正式 source 至少生成并读取一个真实窗口；
- cache provenance、schema、VGGT revision、codec 与 action stats 可哈希；
- dataset 输出保持 `T16/P64/D2048/K8`；
- observed context 的 future-perturbation invariance 复验通过。

### 8.3 短训练 canary

- 从头初始化新 lineage，硬停且不自动晋级；
- direct policy、pose flow、native no-teacher action、native future anchor 均 finite 且有非零梯度；
- factual action 对 native core prediction 有可测 sensitivity；
- RGB/depth/point/pose loss finite；
- 无 OOM、CUDA/NCCL、nonfinite、Traceback、I/O 或数据错误；
- 完整编号 checkpoint 可读、稳定、ZIP 完整。

## 9. 合并标准

只有测试先失败后通过的记录、全量发布测试、五源 cache receipt、短训练 canary report/编号 checkpoint、以及无 Stage1/旧 V7 变更的 diff 审查全部齐全，才合入正式 `v8`。

合并只代表 V8 正式代码具备正确的新 Stage0；不会自动启动长训。
