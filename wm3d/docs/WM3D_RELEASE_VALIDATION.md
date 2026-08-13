# WM3D 发布验收

本文记录统一 WM3D 实现的代码门禁、真实训练证据和交付边界。最终发布以 clean
commit 上生成的 `smoke-real` 总 receipt 为准；开发 worktree 的结果只用于定位问题，
不能替代发布验收。

## 1. V7 问题与 WM3D 修正

| V7 问题 | WM3D 实现 | 验收方式 |
|---|---|---|
| direct action head 没有稳定获得真实监督，预训练和下游存在不同 owner | `NativeWorldModel.action_head` 是唯一 serving owner，连续动作在归一化坐标训练、物理坐标输出 | action owner 梯度非零；训练/eval 都由同一个 `_forward` 和 objective 计算 |
| world dynamics 输入与 policy 输出混用，未来真值动作可能污染 policy state | action-free native prior 与 factual-action dynamics refinement 分离 | policy 只读 action-free state；action shuffle/泄漏回归测试 |
| world state 5 Hz 与 policy 20 Hz 被写成全局常量 | world/action/state 都使用源数据的真实时间戳；T、K、query 数和 horizon 由 profile/window 决定 | 非均匀 cadence、mask、interval ownership 回归 |
| 单臂 7D/10D 数据结构限制模型 | grouped action/current-state ABI，group、semantic、embodiment 和 mask 显式入模 | 真实双臂 ALOHA cache、双臂梯度与模型前向 |
| policy 缺少与首个 action query 对齐的当前机器人状态 | current-state encoder 直接进入 action query，时间戳必须精确对齐 | `current_state_proprio` owner 梯度非零；缺失/错位样本 fail closed |
| 不同 source/group 的物理量尺度混在一起 | grouped normalization v2 分开封存 `fine_command`、`coarse_effect`、`current_state` | artifact/runtime SHA 闭包和 lane 交叉 mask 回归 |
| DDP 不能承载 5B，checkpoint/恢复不支持拓扑审计 | DDP/FSDP2 共用训练入口；DCP 分片 checkpoint；exact resume 与受限 topology reshard | 双卡真实 optimizer step、换新进程 exact resume、5B meta-sharded 物化 |
| Stage1 使用旧 D384 branch codec，和 native 3D token 不一致 | Stage1 直接加载 committed Stage0 DCP，候选 evidence 必须由相同 profile 的 frozen native encoder 生成 | P/D/K/H/runtime/checkpoint/generator receipt 全闭包；旧 D384 明确拒绝 |

## 2. 静态与单元验收

统一 review 树执行：

```bash
PYTHON_BIN=.venv/bin/python ./run_wm3d.sh check
```

移除旧 V7 trainer、fixed-rate sidecar、single-file checkpoint eval 和它们的专属历史
测试后，最终发布候选必须直接执行上述 `./run_wm3d.sh check`，并且不能使用 `-k`、路径筛选、
排除列表或允许失败项。测试数量会随代码变化，不作为发布合同；以发布 commit 上无排除的
完整命令退出码为准。发布 commit 还要由 `smoke-real` 在空目录重跑并写入总 receipt。

这些测试覆盖：

- source lock、下载闭包、schema/adapter audit、inventory、data profile；
- episode cache、window selection、grouped normalization；
- 1B/5B 参数封印、双臂 grouped ABI、变 cadence；
- FSDP2、activation checkpoint、gradient ownership、DCP exact resume；
- offline eval；
- Stage1 action-blind rollout、label/action shuffle 和 planner gradient ownership。

## 3. 真实 Stage0 证据

### 3.1 数据

开发验收使用公开数据 `lerobot/aloha_sim_insertion_human`，冻结 revision：

```text
cc571a3c661df81b566dbfde3d5c1e85fcdf7884
```

只选 episode 0（train）与 episode 30（val）。两条机械臂分别映射为独立 action/state
group；原始 observation/action cadence 为 50 Hz。任务文本使用冻结 Qwen3-VL encoder，
视觉状态使用冻结 native VGGT encoder。episode cache 只生成一次，1B window 与
normalization 是独立的小型派生 artifact。

这个小样本用于检查真实数据链路和可学习性，不代表正式数据分布，也不能给出 LIBERO
成功率结论。

### 3.2 FSDP2 与 activation checkpoint

`native_1b` 的精确参数量为 `1,194,740,883`。node43 的 GPU 0–1 完成：

```text
BF16 mixed precision
FSDP2 meta-sharded initialization
structural non-reentrant activation checkpoint
forward → backward → AdamW optimizer.step
```

两 rank 的全部 loss 均 finite，`gradient_nonfinite=0`，峰值显存约 9.80 GB/卡。
11 个 owner 互斥覆盖全部 trainable 参数；除没有输入的 optional auxiliary owner 外，
其余 owner 的梯度均非零，包括 current-state、action trunk、action head、native state
trunk、factual dynamics、RGB 和 geometry decoder。

activation checkpoint 的发布修正是结构性的：先在构造期安装
`CheckpointWrapper`，再把 wrapper 作为 `fully_shard` 单元。这样 backward recompute 与
原 forward 都经过同一个 FSDP mixed-precision 边界，避免 BF16 saved metadata 与 FP32
recomputed metadata 不一致。

### 3.3 exact resume 与 offline eval

真实 ALOHA 链路完成：

```text
fresh 0 → 1
committed step_00000001
新 torchrun 进程 exact resume 1 → 2
committed step_00000002
固定 val window offline eval
```

DCP 独立回归还逐 rank 比较了保存前的下一步与换新进程恢复后的下一步，RNG 和 loss
逐字节一致。eval receipt 绑定 runtime SHA、data closure SHA、window index SHA、
`COMMITTED.json`、manifest content SHA 和代码 commit。

### 3.4 100-step 可学习性

完整 1B 模型使用同一真实双臂数据，从 step 20 的完整 DCP 换新进程恢复到 step 100。
训练曲线：

| 指标 | step 5 | step 20 | step 100 |
|---|---:|---:|---:|
| total | 10.5274 | 7.5395 | 4.4392 |
| action fine | 0.8186 | 0.5670 | 0.2474 |
| token MSE | 6.7067 | 5.2305 | 3.7818 |
| RGB Charbonnier | 0.3862 | 0.2365 | 0.0435 |
| depth log | 0.6844 | 0.3594 | 0.0273 |
| point | 0.5119 | 0.2382 | 0.0201 |
| camera pose | 0.2959 | 0.0252 | 0.00054 |

固定 val window 的 step20/step100 对比：

| 指标 | step 20 | step 100 | 变化 |
|---|---:|---:|---:|
| total | 8.5297 | 6.4822 | -24.0% |
| action fine | 1.3452 | 1.1657 | -13.3% |
| token MSE | 5.0656 | 3.9500 | -22.0% |
| RGB Charbonnier | 0.2311 | 0.0438 | -81.1% |
| depth log | 0.1369 | 0.0522 | -61.9% |
| point | 0.1300 | 0.0202 | -84.5% |
| camera pose | 0.0190 | 0.00128 | -93.3% |

这组结果证明 unified action/current-state/native 路径能够从真实样本学习，并且固定验证
集同步改善。它不证明正式规模的泛化能力，也不替代 LIBERO 闭环评测。

## 4. 5B 承载证据

`native_5b` 的精确参数量为 `5,108,342,963`。node43 双卡 FSDP2
meta-sharded 物化结果：

| rank | 本地参数存储 | 全局占比 | 峰值显存 |
|---|---:|---:|---:|
| 0 | 2,557,313,020 | 50.0615% | 10.318 GB |
| 1 | 2,551,029,943 | 49.9385% | 10.318 GB |

两 rank 的参数都已从 meta materialize，没有任何 rank 持有完整 5B replica。正式
P144/T24/K16 window 也由同一 ALOHA episode cache 派生；独立 normalization 和
runtime 通过双卡 full preflight。随后完整 5B profile 在真实双臂样本上完成一个
forward、backward、AdamW step 和 committed DCP：10 个必需 gradient owner 全部非零且
finite，action/native/RGB/depth/point/pose loss 全部 finite。checkpoint 两个分片合计约
61.3 GB，没有 rank0 full-state 聚合。新的双卡进程随后从这个 committed DCP 重新加载
5B 模型并完成固定 val window 的统一离线评估；全部指标 finite。该 eval receipt 的
SHA256 为 `3d3584c8c63ed245cf10068789f9e6986e5a0d940c803e9a0c91f4efbd7e8244`，
`COMMITTED.json` 的 SHA256 为
`aba8cb7a501d9491008432558a88624c0e4a349014a9421081bf2b05126388b8`。

正式 128×H200 配置使用同一模型、dataset、trainer 和 DCP manager，只更换
model/runtime profile。集群交付前仍需在目标环境执行 NCCL、吞吐、checkpoint 带宽和
128-rank preflight；本地双卡结果不能代替目标集群通信验收。

目标集群配置分为同拓扑的 1K canary、100K validation 和 600K formal。每个进程边界都
必须先消费 30 分钟内生成的 resource receipt；receipt 绑定 128 rank 的 hostname、GPU
UUID、H200/HBM/ECC/空闲状态、节点内 NVLink、IB、ulimit、`/dev/shm`、磁盘和真实
all-reduce。复制旧 receipt 或更换 rank/GPU 会被当前身份复核拒绝。

## 5. 真实 Stage1 双 source、四 root 证据

历史 Stage1 v7 开发批次使用两个独立 RoboCasa source：`OpenBlenderLid` 提供 train/val/test 各一个 root，`CoffeeServeMug` 提供第二个 train root；selection 为 train 2、val 1、test 1，没有复制、有放回采样或跨 source 伪装。它证明过真实 simulator 主线的可行性，但其 audit、branch 和 receipt 都是旧 schema，不包含当前要求的 branch v3、generator receipt v2 和持久 `rollout_audit_sha256` 闭包。旧 SHA 因此仅是开发记录，已从当前发布 authority 清单移除；不得用于当前恢复、评测或发布提升。

val/test 的四项 action-blind、label sensitivity 与梯度 ownership 门禁全部通过，但这个 correctness canary 只训练了两个 optimizer step。val/test 的 `selected_success` 都是 0，success AUC 分别为 0.25 和约 0.0333。结论只能是 Stage1 真实 pipeline 已闭合，不能声称规划质量提升或策略已经可用。

## 6. 发布判定

以下项目全部满足后，WM3D 才标记为可交付：

1. clean commit 上的 `./run_wm3d.sh check` 无排除通过；
2. 空目录执行 `smoke-real`，真实完成下载、cache、0→1、exact resume 1→2 和 eval；
3. 总 receipt 绑定代码 commit 与全部输入/输出 SHA；
4. 5B 参数封印、真实 optimizer step、committed DCP 与独立进程重载 eval 通过；
5. 真实 Stage1 simulator branch closure 包含非空 train/val/test，并完成 planner train/eval；
6. 独立空白 Agent 按本文逐项审查，P0/P1/P2 为 0/0/0；
7. README、配置和实际命令一致，发布树中没有旧 trainer 旁路。

目标集群仍需完成两项环境相关验收：公开全集 revision/许可下载和 128-rank 通信/吞吐
canary。仓库会在缺少这些证据时停止，不会把模板或小样本 receipt 当成正式全集结果。
