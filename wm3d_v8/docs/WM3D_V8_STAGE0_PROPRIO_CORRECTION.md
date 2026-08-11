# WM3D V8 Stage0 当前机器人状态修正说明

## 1. 问题与结论

V8 Stage0 已经修正了旧版动作监督的主要错误：5 Hz native 3D dynamics 与
20 Hz policy lane 分离，policy 唯一输出为 `[B,8,7]`，gripper 为
absolute `close01`。但现有正式 v2 配置仍设置：

```yaml
model:
  policy_lowdim_dim: 0
```

因此 action policy 没有直接看到当前 EEF pose、当前 gripper state 或机器人
embodiment；它只能从 action-free native 3D core、任务语义和历史动作间接推断。
这不会破坏 world-model correctness，但会限制闭环 action policy 的迁移上限。

本修正属于 **V8 Stage0 v3**，不修改 V7，也不改变 WM3D 的核心定义：

- native 3D core 继续以 5 Hz 显式预测 RGB/depth/point/camera pose；
- policy 必须从 action-free `core_pred` 解码，不能绕过 3D core 退化为 VLA；
- policy lane 继续输出 20 Hz、8-step、统一 7D action chunk；
- 不重算 RGB/VGGT/geometry cache，只新增 content-addressed proprio sidecar；
- 不修改 Stage1、Wan、Qwen/VGGT 主干、数据 mix 或训练并行策略。

## 2. 唯一 proprio ABI

每个 policy 样本必须提供：

```text
policy_proprio_raw: [10]
policy_proprio:     [10]  # source-bound train-only affine normalization
embodiment_id:      scalar int64
```

10D 物理布局固定为：

```text
[eef_x, eef_y, eef_z,
 R00, R10, R20, R01, R11, R21,
 gripper_close01]
```

- EEF position：robot-base frame，米；
- rotation：base-from-EEF rotation matrix 的前两列，顺序必须与上面完全一致；
- gripper：absolute `close01`，0=open，1=closed；
- 不包含 future observation、future state、action target 或由 future 推回的状态。

固定 embodiment vocabulary：

| id | 名称 | 数据 |
|---:|---|---|
| 0 | `franka_droid` | DROID |
| 1 | `widowx_bridge` | Bridge |
| 2 | `panda_robocasa_libero` | RoboCasa 与 LIBERO |

词表本身必须 JSON canonicalize 后 SHA256 固定，并写入 checkpoint contract。
未知 embodiment、越界 id、错误 shape、NaN/Inf 一律 fail closed。

## 3. 时间锚点

当前状态必须与 policy 的第一个 action target 使用同一物理时刻：

- RoboCasa compact：`world_action_start = context_end - 1`；
- OXE：使用 canonical action resolver 给出的
  `action_frame_indices[0]`，不得用固定 `start+T-1` 代替；
- LIBERO serving：使用产生当前 action chunk 的同一环境 observation。

不允许 nearest-frame、前后补齐、重复末帧、零向量 fallback 或 silent
pad/truncate。

## 4. 各来源转换

### RoboCasa

从原始 parquet 的 `observation.state[16]` 读取：

- `[7:10]`：EEF position；
- `[10:14]`：EEF quaternion，`xyzw`；
- `[14:16]`：Panda finger qpos。

两指保存的是有符号的相向关节坐标，物理 aperture 必须固定为
`abs(qpos_left-qpos_right)`；这也与既有 factual-action audit 的定义一致。
`abs(left)+abs(right)` 只在两指严格异号时等价，在同号采集帧上会产生错误宽度。
固定名义合同为 `0.00=closed, 0.08=open`，超出名义开口的状态显式饱和为
open；严格观测 envelope 为 `[0.00,0.12]`，超过该范围、字段缺失或
episode/frame 身份不一致必须失败。

正式原始数据全量审计覆盖 402 个 parquet、145,352,665 帧：4,587,206 帧
两指同号，aperture 最大值为 0.118746579 m，423 帧超过 0.10 m，没有任何
一帧超过 0.12 m。因此这里不是放宽为无界 clip，而是以完整数据和物理开口
共同封存可接受观测上界；归一化仍使用名义 0.08 m。

### DROID

读取 canonical action index 已固定的 `state_pose_path`：

- pose 为 `xyz + fixed-extrinsic xyz Euler`；
- grip 已是 `close01`；
- sidecar SHA 必须与 canonical index 一致。

### Bridge

从原始 RLDS pickle 读取 `xyz + fixed-extrinsic xyz Euler + open01`，
将 `open01` 唯一转换为 `close01 = 1-open01`。名义区间仍为 `[0,1]`，
严格观测 envelope 为 `[-0.05,1.12]`，区间内的采集过冲显式饱和到名义
区间，更大越界必须失败。原始 tar/member 与 manifest 身份必须一致。

正式 Bridge 全量审计覆盖 7,267 clips、262,794 帧：只有两个重复来源 clip
各有两帧超过 1.05，四帧最大值为 1.115462542；相邻序列连续回落，属于
观测过冲而非字段错位。全量最小值为 0.049614936，没有低端越界。

### LIBERO

serving 与预训练共用同一个 Panda adapter：

- EEF position + `xyzw` quaternion + 两指 qpos；
- 同一个 rotation6D 顺序和 `close01` 转换；
- 只允许替换 source-bound normalization stats，不允许更换 ABI。

## 5. Sidecar 与归一化

每个来源独立生成 proprio sidecar、JSONL index 和 train-only stats：

- payload 固定 schema、identity、split、source、embodiment、frame indices、
  raw proprio、上游 state bytes SHA；
- index 固定每个 payload 的 SHA、frame count 和上游身份；
- stats 固定 index SHA、layout、vocab SHA、mean/std；
- runtime 在 dataset 构造时重新核对 index/stats SHA，在首次读取每个 payload
  时再次核对 payload SHA；
- train/val 共用 train-only stats；低方差维度使用显式 std floor 规则；
- sidecar builder 使用同目录临时文件、文件 `fsync`、原子 hard-link 发布与目录
  `fsync`；并发发布只能有一个创建者，重复 identity 内容相同则复用，内容冲突失败。

现有 causal dual-view cache 和 action20 sidecar不变，不需要重新生成。

## 6. 模型、checkpoint 与下游合同

新的正式模型设置：

```yaml
model:
  policy_lowdim_dim: 10
  policy_require_lowdim_state: true
  policy_embodiment_vocab_size: 3
  policy_require_embodiment: true
```

proprio 与 embodiment embedding 作为 action policy 的独立当前状态 token 注入；
它们不能进入 native 3D target，也不能替代 `core_pred`。缺字段时模型必须抛错，
不得产生零 token。

checkpoint action contract 升级为 v3，并额外固定：

- proprio schema/layout/dim/anchor；
- embodiment vocabulary 与 SHA；
- 三类 source index/stats 的路径和 SHA；
- `required=true`；
- 原 v2 的 5/20 Hz、action7、history36D、normalizer 和 serving owner 全部保留。

Stage0→LIBERO transition audit 必须逐项核对 v3 contract，并严格加载新增的
`action_policy.lowdim_proj.*` 与 `action_policy.embodiment_embed.*` 参数。

## 7. 必须通过的验收

- 五个正式 source 都提供 finite `[10]` 与合法 embodiment id；
- state 时间锚点精确等于第一个 action target；
- source/index/stats/payload 任一 SHA 错误均 fail closed；
- 缺 state、未知 embodiment、错误 shape、静默补零均 fail closed；
- 统一 head 对 proprio token、embodiment token 与 action-free native 3D
  `core_pred` 都有 finite、非零梯度；
- v2 checkpoint 读取兼容性保留，但 v2 不得冒充带 proprio 的 v3；
- 真实 0→20、exact resume 20→100 canary 与 authority gate 通过后，才允许
  启动 v3 formal world16；旧 v2 formal 配置不得继续使用。

## 8. 实现与验证记录

本次修改只落在 V8 release worktree，旧 V7、正在运行的 causal-dual-view
cache producer 和已封存 cache 均未改动。新增/修改点为：

1. `v8_proprio_contract.py`：统一 10D 物理 ABI、三类 adapter、词表、严格
   content-addressed runtime store；
2. `build_wm3d_v8_proprio_sidecars.py`：从真实 RoboCasa parquet、DROID
   canonical state、Bridge raw RLDS pickle 构建 sidecar；RoboCasa 按 parquet
   分组逐文件读取，避免正式规模时把全部状态表常驻内存；
3. RoboCasa/OXE dataset：在第一个 policy action target 的精确帧读取当前状态，
   输出 normalized/raw proprio、embodiment 与可审计锚点；
4. action policy：增加 proprio token 和 embodiment token，并在 required 模式
   禁止零向量 fallback；native 3D 路径与 36D causal conditioner 不变；
5. train/preflight/checkpoint transition：封存 v3 ABI、三源 index/stats SHA，
   并检查 batch 中 proprio frame 与首个 action frame 完全相等；
6. 新建 canary/formal/world16 v3 配置；所有 sidecar 与 gate SHA 在正式生成后
   才可替换 `PENDING_*`，因此未封存资产时配置必然 fail closed。
7. checkpoint transition 显式绑定 config schema 与 action contract schema：v2
   只能配 v2，v3 只能配 v3；交叉组合即使各自哈希自洽也必须拒绝。
8. V8 完整 proprio 合同只由 V8 专属 metadata 触发；旧 V7/v2 batch 单独携带
   通用 `lowdim_state` 时仍走原兼容路径，不得被误判为不完整的 v3 batch。

代码回归在 2026-08-12 使用项目训练环境完成：`120 passed`，并包含并发
no-clobber 竞态、source manifest 重复 identity、legacy lowdim 兼容、v2/v3
checkpoint contract 交叉组合，以及 training/structure preflight 对
schema↔proprio 模式双向冒充、signed Panda aperture 与封存观测 envelope
的确定性回归；envelope 端点与其 float32 `nextafter` 外侧分别测试，禁止
隐式浮点容差放宽。对真实数据
执行了 sidecar dry-run 和实际 no-clobber 双次发布验证：

- RoboCasa：9 个真实 causal-dual-view archive，train/val/test 均覆盖；
- DROID：9 个真实 canonical clip，共 583 个 current-state frame；
- Bridge：9 个真实 raw-tar clip，总计 381 个 current-state frame；其中
  train-only stats 使用 357 帧；
- 三类实际发布均可由 `V8ProprioStore` 按首/末帧重新读取，二次发布 SHA
  完全一致且未覆盖已有内容。

## 9. 独立空白审查结论

2026-08-11 的独立只读审查最终分级为 `P0/P1/P2 = 0/0/0`，merge gate
PASS。审查者未修改文件、未操作 cache/训练进程，也未使用 node41/42。

该结论只表示代码与合同可以合并，不表示资产门禁已经完成：v3 配置中的
`PENDING_*` 必须在正式 cache 并集、三源 proprio/action sidecar 与 SHA 封存后
替换；随后仍须依次通过 static/full preflight、真实 0→20、exact resume
20→100 canary 和 authority gate。临时 `/tmp` 验证资产不能进入正式配置，旧
v2 formal 不能继续启动。

v3 runtime 必须由 seal 工具一次性接收 RoboCasa、DROID、Bridge 三源各自的
proprio index/stats（六个参数 all-or-none）。seal 会把真实路径与 SHA 写入
顶层 RoboCasa 和 `v8_proprio_by_source.{droid,bridge}`，并在发布 runtime
config 前执行 full preflight；缺任意一个资产、SHA 不匹配或仍有
`PENDING_*` 都不得启动。

LIBERO 在本 Stage0 修正中的交付边界是共享 Panda physical adapter、固定
embodiment id=2、v3 checkpoint ABI 和严格 Stage0→LIBERO loader。LIBERO
环境 observation 接线、下游 source-bound stats 与 rollout serving 属于后续
下游集成，不在本次 Stage0 formal 启动路径中冒充已完成。
