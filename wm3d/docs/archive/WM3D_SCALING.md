# WM3D 统一训练与扩展设计

本文定义 WM3D 从单机验证到 5B 多机训练所使用的同一条生产链路。5B 不是单独项目，也不拥有专用的数据格式、训练器或模型实现；1B 与 5B 只允许在模型配置、数据清单和运行时配置上不同。

## 1. 目标与边界

WM3D 的扩展工作必须同时满足以下条件：

1. 保留 WM3D 已验证的原生 3D 世界模型语义，不能退回 V7 的错误 action 路径。
2. 同一个模型类能够实例化 1B 与 5B；不能出现 `train_5b.py`、`dataset_5b.py` 一类分叉实现。
3. 同一个数据 ABI 能表达单臂、双臂、底盘、腰部、头部、夹爪和触觉等可变 embodiment。
4. 同一个训练入口支持 DDP 与 FSDP2；同一个 checkpoint 合同支持精确恢复。
5. 新数据通过 adapter 和 manifest 接入，训练器不得按数据集名称编写分支。
6. 下载、转换、cache、封存、训练与评测均可分片、断点续作、并行执行，并具有可审计收据。

V7 的 5B 工程实现只作为参考。V7 中已经成熟的时空分解注意力、FSDP2、分片 checkpoint、数据清单和一键工作流可以复用其工程思想；V7 的 action 插值、固定 action 语义和旧 Stage0 owner 不能进入 WM3D。

## 2. WM3D 不可回退合同

以下语义在 1B、5B、预训练、下游微调和 serving 中保持一致：

| 项目 | WM3D 合同 |
|---|---|
| 世界状态时间轴 | 原生 3D 状态使用真实时间戳和连续时间编码；不存在全局固定 Hz，5Hz 只属于旧实验的兼容 profile |
| Policy action | 唯一 owner，按显式 query timestamp 输出 action chunk；“高频”由数据源/控制器定义，不等于固定 20Hz |
| Panda 兼容 ABI | 当前 LIBERO adapter 可输出 `[B, 8, 7]`；其 20Hz 仅复现实验协议，不是统一模型的 action 频率 |
| 世界动力学 action | 使用实际执行过的 source-native 子步和时间戳作为条件，不使用未来 policy 预测替代事实 action |
| 当前机器人状态 | policy 必须直接接收与 action chunk 首帧同一时刻的真实 proprio 和 embodiment token |
| 未来预测 | 显式预测 native token、RGB、depth、point、camera/pose；不能只训练一个 VLA action decoder |
| 防泄漏 | policy 不能读取未来真值 action、未来真值视觉或 teacher-forced future state |
| Stage1 | 从 Stage0 的 action-blind native state 做规划；不能用动作标签生成候选 world state |
| 归一化 | 每源统计量与 index、payload SHA 绑定，运行时 fail closed |

## 3. 统一配置分层

一次训练由四份正交配置组成，最终物化为一份不可变 runtime YAML：

```text
model profile       决定 T/P/K/D、宽度、深度、注意力和 decoder
data profile        决定 source、比例、manifest、adapter、embodiment 和资产 SHA
runtime profile     决定 DDP/FSDP2、节点、卡数、batch、精度和 checkpoint
objective profile   决定 Stage0/Stage1 loss 及其权重
```

配置物化器必须记录：每个输入配置的路径与 SHA256、最终配置 SHA256、代码 commit、环境 lock SHA、数据 closure SHA。正式训练只能读取物化后的 runtime YAML。

### 3.1 模型 profile

推荐保留两份正式 profile：

| 参数 | native-1b | native-5b |
|---|---:|---:|
| T | 16 | 24 |
| P | 64 | 144 |
| K | 8 | 16 |
| token D | 2048 | 2048 |
| state hidden | 1600 | 2560 |
| state layers | 18 | 32 |
| state heads | 16 | 20 |
| action hidden | 1280 | 2048 |
| action layers | 14 | 24 |
| action heads | 16 | 16 |
| state/action bridge | 配置项 | 10 |
| factual dynamics refinement | 1 block × 2 次共享执行 | 1 block × 2 次共享执行 |
| max action groups | 8 | 8 |
| max action dim/group | 16 | 16 |
| max action substeps/interval | 容量上限 128 | 容量上限 128；batch 使用真实较短 S |
| max policy queries | 容量上限 256 | 容量上限 256；batch 使用真实较短 C |

这里的 `token D=2048` 是 VGGT/codec 外部接口，`hidden` 才是主干容量。5B profile 保留已经规划的 T24、P144、K16 和 2560/2048 双主干，不通过缩短序列、减少 decoder 或冻结大块参数伪造 5B 训练。WM3D 为隔离 action-free policy state 与 factual-action world state新增 dynamics refinement；高频动作则在汇聚前对每个 `(真实时间戳, 动作值)` 做联合非线性编码，避免交换两个子步后表示不变。当前 dual-path RGB profile 的精确参数量为：5B `5,323,627,059`，1B `1,327,691,187`。两者都必须在参数报告中精确封印，不能为了保留旧的整数标签而削弱 RGB 输出。

实测参数组成如下；数字由 `NativeWorldModel.parameter_counts()` 从正式 profile
实例化后计算，不是按层数手工估算：

| 模块 | native-1b | native-5b | 配置理由 |
|---|---:|---:|---|
| state trunk | 722,620,800 | 3,250,831,360 | 世界状态建模是主容量，5B 重点扩宽到 2560、加深到 32 层 |
| action trunk | 270,699,520 | 1,195,474,944 | 保留独立 action 时序建模，但小于 state trunk，避免退化成 VLA 主干 |
| state/action bridges | 99,550,080 | 424,719,360 | 10 个深层交互点让 policy 读取预测的原生 3D 状态 |
| factual dynamics refinement | 50,390,400 | 127,810,560 | 只让已执行 action 修正 future world，不把未来真值 action 泄漏给 policy |
| multi-view fuser | 9,884,160 | 16,783,360 | 同时间戳多相机融合；缺失视角只用 mask |
| appearance dynamics | 11,769,344 | 42,185,472 | 保留逐视角 P256 高频信息，并由 3D future state 条件化预测未来 appearance latent |
| RGB head | 126,467,203 | 182,456,067 | 联合逐视角 appearance latent 与 3D 条件解码；1B/5B 分别为 1280/1536 hidden、每层级 2 个 residual block，并监督全部 K 帧 |
| geometry head | 1,656,864 | 3,959,840 | depth、point、camera/pose 的显式几何输出 |
| grouped action head | 21,776 | 34,832 | 轻量共享 head；能力来自 3D/action trunk，而不是另建大型 VLA decoder |
| 其余 embedding/norm/projection | 34,631,040 | 79,371,264 | task、连续时间、group、semantic、embodiment、current-state 等合同参数 |
| **总计** | **1,327,691,187** | **5,323,627,059** | 与两份 dual-path profile 的 `expected_parameter_count` 精确一致 |

`dynamics_layers=1` 保持参数量与 checkpoint lane 不变；`factual_dynamics_repeats=2`
让同一个 factual-only block 共享权重执行两次，只加深真实动作对 world state 的修正，
不改 action-free state 或 policy。step500 固定 10 样本扫描中，两次执行把 mean token gain
从 `0.000173` 提高到 `0.000739`，8/10 样本为正；三次执行没有进一步改善，因此正式
profile 固定为两次。该改动仍必须通过从零 canary 验证 RGB 与训练稳定性。
训练期的 token counterfactual 使用同一 action-free trunk、同一 factual dynamics 和两组
可导的 factual/zero-future-action 输出做相对排序；zero 分支不能 detach，否则排序项会
退化为重复 factual 重建。RGB 在 teacher ratio 大于零时可直接看到未来 appearance，因而
不在该阶段施加 RGB counterfactual 排序；RGB 动作因果性统一在 `teacher=0` 验证中测量。
若后续消融表明动作条件动力学不足，应通过 model profile 同时调整 state/dynamics
预算，并重新封存精确参数数；不能在训练脚本中按“5B”名字偷偷加层。

### 3.2 数据 profile

数据 profile 只引用封存后的 source manifest。每个 source 条目至少包含：

```yaml
name: agibot_beta
adapter: agibot
manifest: /path/to/manifest.jsonl
manifest_sha256: ...
embodiment: agibot_g1_bimanual
sampling_weight: 0.25
split: train
assets:
  native_cache_index: /path/to/index.jsonl
  native_cache_index_sha256: ...
  # action/proprio 是同一 robot shard 中的物理量，按同一 episode identity 封存；
  # 训练时的 grouped normalization 另以 SHA-bound artifact 注入 runtime。
```

训练器只能消费统一 batch，不感知 `agibot`、`droid`、`bridge` 或 `robocasa` 名称。数据差异由 adapter 在 cache 阶段消解。

## 4. 可变 embodiment 数据 ABI

### 4.1 Action group

每个机器人由若干 action group 组成，例如：

```text
Panda:        [arm, gripper]
双臂机器人:   [left_arm, left_gripper, right_arm, right_gripper]
全身机器人:   [base, waist, head, left_arm, left_gripper, right_arm, right_gripper]
```

统一张量使用 padding + mask，而不是固定 7D。`world_state_times` 与
`action_times` 都来自真实时间戳；二者没有固定整数频率关系：

```text
fine_action_values   [B, F, G, S, A]
fine_action_mask     [B, F, G, S, A]
fine_action_dt       [B, F, G, S]
coarse_action_values [B, F, G, A]
coarse_action_mask   [B, F, G, A]
group_ids            [B, G]
group_mask           [B, G]
action_semantic_ids  [B, G, A]
world_state_times    [B, F + 1]
policy_query_dt      [B, G, C]
policy_query_mask    [B, G, C]
```

其中 `F` 是相邻 native world-state 时间戳形成的 interval，`S` 是该 interval
内实际存在的 command 子步。状态和 action 都可以是任意经审计的原生 cadence，
也可以是非均匀采样；示例中的 5/10/15/20/30Hz 都不是核心默认值。两条时间轴都进入连续时间编码，
padding 位置 mask 为 false。模型输出的 query 时间也由 serving/data profile 显式给出，
不能从 `world_state_hz` 推导。

`max_action_substeps` 和 `max_policy_queries` 只是 fail-closed 容量上限。cache 按本批
真实的 `S/C` 保存，模型前向也只构造实际 query 数量；提高容量不会强迫所有数据按
最高频率补空位或承担最高频率的 attention 成本。policy query 使用共享 seed、连续
时间嵌入、group/embodiment 和 current state 定义身份；容量上限不会新增一组离散
位置参数，也不会暗中恢复固定 20Hz 的 slot 语义。

`T/K` 仅规定每个窗口包含多少个真实状态样本，不规定这些样本覆盖多少秒。cache
只能从原始时间轴选择严格递增的真实样本并保留其 timestamp；允许按时间范围选择
观测子序列，但不允许插值出新的世界状态或把选中的 timestamp 改写成等间隔。

### 4.2 禁止伪造高频监督

粗频 action 不能通过线性插值变成更高频标签：

- 有原生时间戳的高频 command 写入 `fine_action_values`。
- 只有 world-state interval effect 的数据写入 `coarse_action_values`，其 fine mask 必须全 false。
- policy 高频 loss 只在 fine mask 为 true 的维度计算。
- coarse 数据仍可训练 action-conditioned dynamics、interval effect 和 coarse composition loss。
- 不得 padding 最后一帧、插值中间帧或复制 coarse action 来增加 fine label 数量。

### 4.3 当前状态

当前状态同样采用 group-aware ABI：

```text
current_state_values [B, G, Q]
current_state_mask   [B, G, Q]
state_semantic_ids   [B, G, Q]
embodiment_id        [B]
```

Panda 当前 10D `xyz + rotation6d + gripper_close01` 是该 ABI 的一个 profile，而不是训练器中的固定形状。原始状态必须锚定 action chunk 首个 command 的时间戳；缺失、越界或无法确定坐标系时必须拒绝样本。

## 5. 统一模型

`NativeWorldModel` 是 1B 与 5B 的共同实现：

1. 多视角 native token 进入 factorized state trunk。
2. 每帧做空间注意力；每个 patch 位置做因果时间注意力，避免对 5760 token 做全局 dense attention。
3. action trunk 对真实时间轴和 embodiment group 做分解注意力，复杂度从
   `(steps×groups)²` 降为 `groups×steps² + steps×groups²`；两个轴都用实际
   timestamp 做因果 mask，双臂/全身不会退化成互不通信的独立 policy。
4. 已执行 action group 经过 group、semantic、时间间隔和 mask 编码，只条件化对应 future dynamics。
5. policy action query 从 action-blind predicted native state、语言、历史 action、当前 proprio 和 embodiment 解码。
6. 单一 grouped action head 按显式 query 时间输出所有 group；双臂、全身与单臂走同一 head。Panda serving adapter 只是按当前 benchmark 协议投影为 `[B,8,7]`，不拥有第二条训练路径。
7. decoder 始终保留 native token、RGB、depth、point 和 camera/pose 输出。

模型必须提供 `iter_fsdp_units()`，把 state block、action block、bridge 和大型 decoder 作为 FSDP2 分片单位。activation checkpoint 按 block 配置，不得在训练脚本里按模型规模硬编码。

## 6. 统一分布式与 checkpoint

训练入口只接受 `distributed.strategy`：

```yaml
distributed:
  strategy: fsdp2       # ddp | fsdp2
  shard_mesh: [16, 8]   # 可选 HSDP；16 个 replica，每个 replica 8 卡分片
  param_dtype: bf16
  reduce_dtype: fp32
  activation_checkpoint: true
```

- 1B 小规模验证可使用 DDP。
- 1B 或 5B 都可使用 FSDP2；策略不由模型名字决定。
- 5B 默认 64×H200 正式配置使用 FSDP2/HSDP、BF16、逐层 activation checkpoint；
  128 卡 profile 保留为资源充足时的可选扩展。
- optimizer 与 model 通过 Distributed Checkpoint 分片保存，不在 rank0 聚合完整 state dict。
- checkpoint 使用临时目录、分片写入、全局 barrier、manifest、`COMMITTED.json` 原子提交。
- exact resume 恢复 model、optimizer、scheduler、step、RNG、sampler epoch/cursor、数据 closure 和配置 SHA。
- world size 变化只有在 topology-safe reshard canary 通过后才允许。

## 7. 高吞吐数据链路

完整数据链路分成五层：

```text
lock -> download -> convert/materialize -> schema audit -> adapter strict audit
     -> inventory -> data profile -> task bank -> cache -> seal -> window index
```

### 7.1 下载

- 每个数据源提供独立 downloader adapter；支持校验和、断点续传、并发分片和已有文件跳过。
- 下载层只产出 raw inventory，不直接生成训练样本。
- Hugging Face、对象存储和本地 tar 使用相同 inventory schema。

### 7.2 转换

- 转换任务按 episode/shard 建立静态 task manifest。
- worker 只写自己的 content-addressed 临时结果，成功后使用 no-clobber 原子发布。
- CPU 解码、GPU VGGT/native encoding 和压缩写盘分成流水线，避免 GPU 等待单线程视频解码。
- 单机、多机和作业数组使用同一 task manifest，只改变 worker 数量。

### 7.3 Cache

- 昂贵 episode cache key 只绑定 source manifest row/payload SHA、adapter SHA、视觉 encoder SHA、task encoder/bank SHA、规范视角与 episode representation SHA。
- `T/K`、模型深宽、objective 和 1B/5B profile 不得进入 episode cache key；它们只进入便宜的 window index/runtime closure。`P/D` 只有在它们改变共享 episode representation/encoder contract 时才会自然改变 representation SHA，不能再以模型 profile 名义重复写入 cache identity。
- 已完成且 SHA 一致的 artifact 直接复用；不一致时 fail closed，不能覆盖。
- cache worker 定期写进度收据，可从未完成 task 继续。
- view token 使用 int8 per-vector + scale；depth/point 使用 fp16；RGB 使用 JPEG pack。action/proprio 与 native 表征写入同一 content-addressed episode artifact，共用同一 episode identity 和 receipt；禁止另建会与 native cache 漂移的 sidecar。修改 window 长度不要求重算昂贵的 VGGT 几何。

### 7.4 封存与快速启动

全量 payload 哈希只在资产首次封存时执行一次。后续训练启动使用：

1. 已封存 closure/index SHA；
2. 抽样 payload schema/identity 检查；
3. 训练读取时对实际采样 artifact 做严格校验；
4. 周期性后台 scrub。

这样既不降低数据闭包，又不会在每次启动前重复扫描数 TB 数据。

## 8. 一键入口

最终入口统一为：

```bash
./run_wm3d.sh env
./run_wm3d.sh lock-resolve  <source-template> <lock.yaml> <lock-receipt.json>
./run_wm3d.sh download      <lock.yaml> <raw-root> <token-file>
./run_wm3d.sh schema-audit  ...
./run_wm3d.sh adapter-audit ...       # 第一处必须由人确认语义
./run_wm3d.sh inventory     ...
./run_wm3d.sh data-profile  ...
./run_wm3d.sh task-bank     ...       # 必须先于 cache-plan
./run_wm3d.sh cache-plan    ...
./run_wm3d.sh cache-worker  ...
./run_wm3d.sh cache-seal    ...
./run_wm3d.sh window        ...
./run_wm3d.sh normalization ...       # 按 model/window 生成，不重算 episode cache
./run_wm3d.sh runtime       ...
./run_wm3d.sh preflight     ...
./run_wm3d.sh train         ...
./run_wm3d.sh eval          ...
```

`smoke` 命令必须能从一个真实小分片完成下载、转换、cache、两卡 FSDP2 训练、exact resume 和 eval。正式集群只扩大 profile，不更换命令和代码路径。

多相机数据使用 data profile 中的 canonical view slots。每个 source adapter 显式把
原始相机名映射到这些槽位；没有的槽位只写 `view_mask=false`，不能用另一路相机填充。
视频帧按封存的 episode 行序/真实 frame index 与 observation clock 绑定，容器 PTS 作为
独立审计证据保存；不能假设 MP4 的名义 fps 与机器人记录频率相同，也不能最近邻重采样。

## 9. 发布门槛

WM3D 可以交付 5B 集群前，至少完成以下验收：

1. clean release tree 的统一依赖闭包、静态检查和全量测试通过；发布树不保留旧 trainer、fixed-rate sidecar 或 single-file eval 旁路。
2. 统一模型用 1B 与 5B profile 实例化；参数组成报告与预算一致。
3. 双臂真实/封存样本经过 adapter 后 group、mask、时间戳和 gripper 语义正确。
4. 粗频数据不能产生任何 fine policy label，突变测试必须失败。
5. 1B 两卡 FSDP2 完成真实 optimizer step、分片 checkpoint 和独立进程 exact resume；5B 完成真实 optimizer step、committed DCP 和独立进程重载 eval。
6. DDP 与 FSDP2 的分布式 owner、梯度归约、DCP exact-resume 回归通过；目标集群另做同批次短程数值对齐。
7. 默认 64 卡拓扑、通信 canary、checkpoint 带宽和 dataloader 吞吐在目标集群达到预算。
8. Stage0 的 RGB/depth/point/native/action/proprio 梯度均 finite 且非零。
9. Stage1 能消费同一 Stage0 checkpoint，并保持 action-blind rollout 合同。
10. 从空服务器按中文 README 执行 `smoke-real`，无需修改源码或手工补路径，且总 receipt 绑定代码 commit 与全部输入/输出 SHA。

在这些证据完成前，配置可以标记为 `candidate`，不能标记为 `release-ready`。

## 10. Factual action 条件约束

action-free native prior 是 policy 隔离分支，不是“零 action 的同模型反事实”；两者的
误差量级相差过大，不能拿来证明 world dynamics 使用了 factual action。Stage0 必须在
同一模型、同一 dynamics、同一 mask/时间语义下，把 future factual action value 置零并
做一次 stop-gradient 对照前向。token 用 masked MSE、RGB 用 masked L1 比较 factual 与
zero-action 目标误差，factual 未达到封存 margin 时施加 ranking penalty。模型同时把
factual action 的逐 horizon 摘要作为 parameter-free residual 送入 world state 和
appearance query；这两个 residual 都位于 policy 分支之后，所以 future action 仍不可能
写回 policy lane；factual-only refinement 使用同一 block 两次共享执行。日志必须记录
zero-action token/RGB error、target gain、advantage 和
factual-zero response RMS。第二次无梯度前向属于 objective contract 的显式计算成本；
任何改动其权重、margin 或 residual scale 的运行都必须重新走 canary，不能冒充旧
checkpoint 的同合同 exact resume。
