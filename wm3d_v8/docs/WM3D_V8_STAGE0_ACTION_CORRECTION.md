# WM3D V8 Stage0 动作预训练修正说明

## 1. 修正目标

本修正只处理 WM3D **V8 Stage0** 的动作数据合同、动作监督和下游继承合同。它不改变 V8 的核心定义：

- 世界状态主干仍以 5 Hz 运行；
- 仍然显式预测 native RGB、depth、point 和 camera pose；
- `T=16`、`P=64`、`K=8`、`D=2048` 的原生 3D 世界建模形状保持不变；
- policy 必须读取 action-free 的 WM3D `core_pred`，不得绕过 3D core 退化成图像到动作的 VLA；
- 不修改 V7，不修改 Stage1，不接入 Wan，不扩大到新的模型主干。

本修正保留 5 Hz 3D core，并把两种不同频率的物理量拆开：

1. **world lane**：5 Hz，预测动作造成的显式 3D 世界结果；
2. **policy lane**：20 Hz，输出可直接执行的 action chunk。

## 2. 已确认的问题

### 2.1 5 Hz 动作同时承担 dynamics 输入和 policy 标签

RoboCasa compact cache 保存了原始 20 Hz 控制，但现有 loader 只读取下采样后的 5 Hz `actions`。下采样方式为：

- 平移四步求和；
- 旋转四步按 SO(3) 顺序复合；
- gripper 取最后一步状态。

这对 5 Hz 世界动力学是正确的，却不是 20 Hz controller command。现有 policy 以相同的 `K=8` 预测这组 5 Hz 聚合动作，因此实际含义是 1.6 秒的八个 physical-effect interval，而不是可执行的 0.4 秒、八步 20 Hz action chunk。

影响：

- 接触、抓取和夹爪切换的细粒度轨迹被聚合；
- 下游 LIBERO 若按 20 Hz 执行，会把 5 Hz effect 当成 controller command；
- policy 被迫学习“聚合结果的逆问题”，迁移下限明显受损。

### 2.2 正式 V8 配置存在多个 action owner

当前 V8 正式配置继承了旧 action-repair 链，实际同时启用：

- direct pose head；
- delta-event gripper head，并以 composed gripper 作为 serving owner；
- pose-only flow head；
- native `ActionProjHead` 的 no-teacher action loss。

因此“训练什么”和“部署执行什么”不是同一个张量、同一个 head、同一个语义。尤其 gripper 在预训练中由 event head 拥有，LIBERO/serving 又容易回到 absolute gripper，导致 checkpoint 即使能加载，也不是语义连续的继承。

### 2.3 现有 checkpoint 合同不足以证明可迁移性

`wm3d_action_policy_v1` 只记录 horizon 和输入维度，没有固定：

- world/policy 控制频率；
- pose 坐标系、单位和旋转表示；
- gripper 极性与 absolute/delta 语义；
- action history 的时间语义；
- serving owner；
- policy 读取的是 `core_pred` 还是旁路特征；
- 归一化统计的身份。

这允许“形状相同、物理意义不同”的 checkpoint 被误加载。

### 2.4 OXE 没有可验证的 20 Hz 标签

当前 Bridge/DROID canonical cache 是已经审计过的 5 Hz interval action。它不能通过重复、线性插值或除以四来伪造四个 20 Hz controller command。伪造会使训练 loss 看似下降，却没有增加真实动作监督。

## 3. 修正后的唯一动作合同

### 3.1 统一 serving 输出

V8 Stage0 的唯一可执行动作输出为：

```text
[B, C=8, 7]
= 8 × (base-frame delta pose 6D + absolute gripper close01)
@ 20 Hz
```

具体语义：

- translation：robot-base frame，米，逐 controller step 的 delta；
- rotation：robot-base frame，弧度，SO(3) rotation vector；
- gripper：absolute `close01`，0=open，1=closed；
- chunk：8 步，共 0.4 秒；
- 训练、checkpoint、LIBERO 微调和 serving 使用同一个 base policy head；head 输出 pose normalization 与 gripper logit，执行边界通过同一 source-bound affine stats 解码为物理 `[B,8,7]`。

以下路径在新的 V8 正式配置中必须关闭：

- delta-event gripper serving owner；
- flow action head/loss；
- native `ActionProjHead` 的 policy/action no-teacher loss。

旧通用实现可为历史 checkpoint 保留，但不得被新的 V8 配置实例化或作为 serving owner。

### 3.2 5 Hz dynamics 条件

每个 5 Hz world interval 最多保存四个真实 controller substep：

```text
4 × action7 + valid4 + dt4 = 36 dimensions
```

因此 native state/action stream 的 `action_cond_dim` 为 36。`valid` 和 `dt` 是合同的一部分，避免把缺失的 controller step 当成零动作。

- RoboCasa：使用 compact archive 中真实 20 Hz raw action，经已固定的 source adapter canonicalize；四步复合必须逐 interval 精确还原现有 5 Hz action。
- OXE：每个 5 Hz interval 只放一个已经审计的 coarse action，`valid=1`、`dt=0.2`；其余 slot 无效，不做重复或插值。

### 3.3 双层 policy 监督

RoboCasa 同时提供：

- 真实 20 Hz fine loss：8 步 pose Huber + absolute gripper BCE；
- 两个 5 Hz group 的可微复合一致性 loss。

OXE 只提供：

- 两个真实 5 Hz coarse target；
- 将 policy 输出的每四个 20 Hz action 通过同一 SO(3) 规则复合后，与 coarse target 对齐。

OXE 的 fine mask 必须全为 false。任何 repeat/interpolate/split 生成的 fine target 都属于合同失败。

### 3.4 action history

policy history 使用 0.8 秒的统一时间窗口：

```text
[16, 9] = 16 × (action7 + dt + valid)
```

- RoboCasa：16 个真实 20 Hz 历史动作均有效，`dt=0.05`；
- OXE：只保留四个真实 5 Hz coarse interval token，`dt=0.2`，其余位置无效。

history 中 gripper 同样使用 close01。时间和有效性不再由固定数组位置暗示。

## 4. 数据处理边界

### 4.1 只新增 action-only sidecar

RoboCasa 已有 V8 causal dual-view cache 内含 `raw_actions`，无需重算 VGGT、RGB、depth 或 point cache。新增 builder 只生成 action sidecar：

- 读取现有 compact archive；
- 读取并 SHA256 固定每个 partition 的 factual action audit/adapter；
- canonicalize 真实 20 Hz action；
- 逐 5 Hz interval 复合并核对 archive 中的 `actions`；
- no-clobber、原子写入 sidecar、index 和 train-only stats；
- sidecar/index/stats 均记录来源、adapter 和 digest。

任何复合误差、缺项、重复 clip、split/source 不一致都必须 fail closed。

### 4.2 OXE 不重新制造数据

OXE 继续使用现有 canonical 5 Hz cache、source-bound stats 和时间偏移证据。loader 只把它转换成带 `valid/dt` 的 coarse 监督合同。

## 5. checkpoint 与 LIBERO 继承

新的 checkpoint 必须保存
`schema=wm3d_v8_stage0_action_policy_contract_v2`，至少固定：

- `world_state_hz=5`、`policy_hz=20`、`policy_horizon=8`、
  `policy_chunk_seconds=0.4`；
- action7 的 pose frame/unit/rotation 和 gripper semantics；
- 概念 owner 为 `base_policy_unified_action7`，checkpoint 内的精确字段为
  `serving_owner=action_policy.base_policy.[pose_norm,gripper_logit]`；
- history schema `[16,9]`；
- `policy_context_source=core_pred`；
- `policy_core_action_cond=none`；
- normalizer schema/manifest digest；训练 loader 必须在读取时重新核对
  sealed SHA，不能只依赖更早的 preflight。

Stage0→LIBERO 只能通过严格 transition audit：

- checkpoint 必须包含上述合同；
- `action_policy.*` 和被声明继承的 native core 参数必须逐 key、逐 shape 完全匹配；
- 不允许 missing/unexpected/skipped/expanded key；
- 不允许从 `latest.pt` 推断；只接受显式编号 checkpoint；
- LIBERO 只可更换 source-bound normalization statistics，不能更换动作语义、head owner 或频率。

## 6. 代码修改范围

本次允许修改：

1. V8 dual-rate action contract 与 action-only sidecar builder；
2. V8 数据 loader 的双频字段；
3. V8 unified action loss 与 36D dynamics conditioner；
4. action policy checkpoint v2 合同；
5. V8 正式配置和 preflight；
6. Stage0→LIBERO 严格 transition audit；
7. 与上述行为直接相关的定向测试。

本次明确不修改：

- V7 仓库或 V7 正式配置；
- native 3D core 的 5 Hz、T/P/K/D 和显式输出；
- Stage1 planning objective；
- Wan、Qwen、VGGT 主干；
- 训练数据 mix、DDP/FSDP 或 checkpoint 保留策略。

## 7. 验收清单

代码完成必须同时满足：

- [x] RoboCasa 真实 20 Hz 四步复合逐 interval 还原 5 Hz action；
- [x] OXE fine mask 全 false，未生成伪 20 Hz 标签；
- [x] dynamics condition 固定 `[B,K,36]`，valid/dt 有效；
- [x] policy 输出固定 `[B,8,7]`，absolute close01 gripper；
- [x] 新 V8 配置不实例化 delta/flow owner，native action policy loss 为 0；
- [x] fine/coarse loss 在有标签与无标签 batch 上都 finite，梯度能到 action policy 和 native core；
- [x] checkpoint 含完整 v2 合同；
- [x] Stage0→LIBERO 错一项合同、key 或 shape 都 fail closed；
- [x] 底层 compact/data ABI 回归测试保持通过；
- [x] 使用至少一个真实 RoboCasa archive 做只读复合审计；
- [x] 空白 Agent 按本文逐项独立审查并给出结论。

## 8. 实现位置与使用顺序

### 8.1 代码对应关系

| 修正点 | 实现文件 |
|---|---|
| 双频 action ABI、SO(3) 复合、36D conditioner、fine/coarse target | `wm3d_v3/data/v8_action_contract.py` |
| RoboCasa 真实 20 Hz action-only sidecar | `scripts/build_wm3d_v8_dual_rate_action_sidecars.py` |
| RoboCasa/OXE loader 接入 | `wm3d_v3/data/v7_compact_dataset.py`、`wm3d_v3/data/window_dataset.py` |
| 唯一 action owner、fine/coarse loss、checkpoint v2 | `wm3d_v3/training/train.py` |
| 正式数据、目标与 owner 的 fail-closed 检查 | `scripts/preflight_wm3d_v8_stage0_causal_dual_view.py` |
| Stage0→LIBERO 严格继承 | `wm3d_v3/training/v8_action_policy_transition.py`、`scripts/audit_wm3d_v8_stage0_libero_transition.py` |
| 新版配置 | `configs/wm3d_v8_stage0_causal_dual_view_unified_action_*_v2.yaml` |

### 8.2 生成 RoboCasa action-only sidecar

下列命令只读取既有 V8 causal archive，不重算视觉/几何 cache。正式运行时把输入、输出改为对应 canary 或 formal sealed 目录；三个 adapter SHA 不得改写或省略。

```bash
python scripts/build_wm3d_v8_dual_rate_action_sidecars.py \
  --input-index "${COMPACT_INDEX}" \
  --adapter-audit atomic=/data/Minko/world_model/wm3d_v7/manifests/audits/robocasa365_atomic_factual_action_v2.json \
  --adapter-audit composite=/data/Minko/world_model/wm3d_v7/manifests/audits/robocasa365_composite_factual_action_v2.json \
  --adapter-audit mg=/data/Minko/world_model/wm3d_v7/manifests/audits/robocasa365_mg_factual_action_v1.json \
  --adapter-audit-sha256 atomic=f0e3c99f5f5792d996473bf47035d989ebe17ff2665a3c5c07c73ea736a3c27f \
  --adapter-audit-sha256 composite=432948a82fd3bc03f0b7c3b80a82d33517970bc013d088d5ea9956f4009320e9 \
  --adapter-audit-sha256 mg=54a62393fc107ec8ed7d0e53e734bd09f5141e0f871a5f1cf89552a964e94dc1 \
  --output-root "${ACTION20_ROOT}" \
  --output-index "${ACTION20_INDEX}" \
  --output-stats "${ACTION20_STATS}"
```

先加 `--dry-run` 可完成全 archive 的只读复合审计。正式输出采用 no-clobber 原子发布；相同内容可幂等复用，不同内容会拒绝覆盖。

### 8.3 封存配置与 preflight

builder 完成后，将它打印的 index/stats SHA256 和已封存 compact index SHA 写入 runtime overlay，替换 v2 配置中的 `PENDING_*`。`PENDING_*` 存在时只能做结构检查，禁止启动训练。

```bash
python scripts/preflight_wm3d_v8_stage0_causal_dual_view.py \
  --config configs/wm3d_v8_stage0_causal_dual_view_unified_action_canary_v2.yaml \
  --mode static

python scripts/preflight_wm3d_v8_stage0_causal_dual_view.py \
  --config "${SEALED_RUNTIME_CONFIG}" \
  --mode full \
  --json-out "${PREFLIGHT_REPORT}"
```

只有 full report 同时满足 `passed=true`、`launch_ready=true`、`errors=[]`、`blockers=[]` 才允许进入训练入口。

### 8.4 Stage0→LIBERO 继承审计

只接受名称和 payload step 一致的 `step_XXXXXXXX.pt`：

```bash
python scripts/audit_wm3d_v8_stage0_libero_transition.py \
  --checkpoint "${NUMBERED_STAGE0_CHECKPOINT}" \
  --expected-config "${SEALED_STAGE0_CONFIG}" \
  --report "${TRANSITION_REPORT}"
```

下游 trainer 必须调用 `load_v8_stage0_for_libero_strict`，完整继承同构 native 3D core 和 `action_policy.*`；任何缺键、额外键、shape 变化或 action ABI 变化都会在加载 CUDA 训练前失败。

执行时必须调用 `decode_v8_executable_action_chunk`：pose 用下游显式提供的统计量解归一化，gripper 极性必须由环境边界显式选择，禁止按数据集名字猜测。

## 9. 完成证据（2026-08-10）

- 清理后发布树测试：`102 passed`。一条 warning 来自 PyTorch Transformer
  `norm_first` 的 nested-tensor 提示，不涉及本修正。
- 真实数据审计：使用 canary compact index 只读检查 9 个 RoboCasa archive，
  覆盖 atomic、composite、MG，train split 共 532 条真实 20 Hz action；
  源 archive SHA 和 20 Hz→5 Hz SO(3) 复合均通过。
- 运行时 normalizer：RoboCasa coarse、DROID、Bridge 三份统计文件均按
  sealed SHA 重新校验。loader 会在读取前拒绝 symlink、非法 SHA 或内容变化。
- Stage0→LIBERO：CLI 会从 sealed 配置实例化 CPU 目标模型，并实际调用
  `load_v8_stage0_for_libero_strict`；缺键、额外键、shape 或 ABI 变化均拒绝。
- 静态 preflight：canary、formal、world16 三份 v2 配置均为
  `passed=true`、`errors=[]`。模板仍含 `PENDING_*`，因此
  `launch_ready=false`；生成并封存正式 sidecar/runtime overlay 之前不得启动训练。
- 独立空白 Agent 最终审查：`P0=0`、`P1=0`、`P2=0`，merge gate 为 PASS。
