# WM3D 5B：从公开数据到 10K 验证训练

这份手册写给第一次接触 WM3D、但负责算力和集群运行的同事。目标不是解释所有模型
细节，而是让操作者在另一套服务器上完成数据下载、cache、5B 验证训练、断点恢复和
结果验收，并能明确判断每一步是否正确。

## 1. 先知道自己在运行什么

WM3D Stage0 是动作条件的 3D 世界模型。每个训练样本同时包含：

- 多视角 RGB 和真实观测时间戳；
- VGGT 生成的 native 3D token、depth、point、camera pose；
- 机器人 current state；
- 真实执行过的 grouped action；
- 任务文本 embedding。

5B profile 是同一模型实现的较大配置，不是另一套代码。当前配置有
`5,108,342,963` 个参数，使用 `T=24` 个历史状态、`K=16` 个未来状态、每个状态
`P=144` 个 native token，token 维度为 `2048`。推荐拓扑是 16 台 8×H200：节点内
8 卡 FSDP2 分片，节点间 16 路 data parallel，总 world size 128。

本手册覆盖 Stage0 5B。Stage1 是冻结 Stage0 后的 simulator candidate planner，单独见
[Stage1 手册](WM3D_STAGE1_UNIFIED.md)。

## 2. 三条不可省略的规则

1. 所有节点必须看到相同的 Git checkout 和相同绝对路径的共享存储。
2. 下载 revision、source manifest、adapter、cache、runtime、checkpoint 都由 SHA/receipt
   绑定。不要覆盖旧文件；换配置就换新的 `WORK_ROOT` 或 `RUN_ROOT`。
3. adapter 的单位、坐标系、夹爪极性、action group 和 current-state 语义只能由项目
   负责人确认。算力同事可以完成其余全部流程，但不能根据数组维度猜 adapter。

## 3. 数据清单与下载位置

扩展数据模板是 `configs/data/public_robot_6106h.template.yaml`。它包含：

| 数据 | Hugging Face repo | 计划时长 | 用途 |
|---|---|---:|---|
| DROID | [`lerobot/droid_1.0.1`](https://huggingface.co/datasets/lerobot/droid_1.0.1) | 350 h | 单臂真实操作 |
| Bridge V2 | [`ember-lab-berkeley/bridge_v2`](https://huggingface.co/datasets/ember-lab-berkeley/bridge_v2) | 100 h | 单臂桌面操作 |
| RoboCasa Atomic | [`ember-lab-berkeley/robocasa365-pretrain-atomic`](https://huggingface.co/datasets/ember-lab-berkeley/robocasa365-pretrain-atomic) | 21 h | 仿真原子技能 |
| RoboCasa Composite | [`ember-lab-berkeley/robocasa365-pretrain-composite`](https://huggingface.co/datasets/ember-lab-berkeley/robocasa365-pretrain-composite) | 383 h | 仿真组合任务 |
| RoboCasa MG | [`ember-lab-berkeley/robocasa365-pretrain-mg`](https://huggingface.co/datasets/ember-lab-berkeley/robocasa365-pretrain-mg) | 1,615 h | 大规模仿真操作 |
| AgiBotWorld 2026 | [`agibot-world/AgiBotWorld2026`](https://huggingface.co/datasets/agibot-world/AgiBotWorld2026) | 661 h | 双臂、全身、多交互类型 |
| AgiBotWorld Beta | [`agibot-world/AgiBotWorld-Beta`](https://huggingface.co/datasets/agibot-world/AgiBotWorld-Beta) | 2,976.4 h | 大规模双臂数据 |

总预算约 6,106.4 小时。AgiBot 数据可能要求先在 Hugging Face 网页接受许可。实际训练
使用的是物化后的 `DATA_PROFILE`，不是表格里的预算数字；报告会打印真实 episode/window
数量。

如果要复现已经交付过的数据家庭和权重，使用
`configs/data/public_robot_5649h_legacy_compatible.template.yaml`。不要把 5,649 h 和
6,106 h 两种配比混进同一个 run。

128×H200 profile 要求 data 和 output 所在文件系统各至少有 10 TB 空闲空间。5B DCP 的
实测量级约 61 GB/个；10K profile 会保存 100、500 和每 1,000 step 的 checkpoint，
应额外预留约 0.8 TB。cache 空间取决于真实 episode/frame 数，必须在 `cache-plan` 后结合
源数据大小重新测算，不能只按 nominal hours 猜。

## 4. 环境和站点配置

### 4.1 克隆和检查代码

```bash
git clone --branch v8 --single-branch https://github.com/wxqnl/world_model.git
cd world_model/wm3d
git status --short
```

`git status --short` 必须没有输出。正式 runtime 会记录当前 commit，训练和评测会拒绝
dirty tree。

### 4.2 建立 site 文件

下面的文件放在共享存储，不能提交到 Git：

```bash
./run_wm3d.sh 5b init /shared/wm3d/control/h200_5b.env
vim /shared/wm3d/control/h200_5b.env
```

至少修改：`WORK_ROOT`、`HF_TOKEN_FILE`、三个 encoder/model snapshot 路径、
`MASTER_ADDR`。默认配置是 16×8 H200 和 10K 验证训练。

```bash
chmod 600 /shared/secrets/huggingface_token
./run_wm3d.sh 5b env /shared/wm3d/control/h200_5b.env
./run_wm3d.sh 5b doctor /shared/wm3d/control/h200_5b.env
./run_wm3d.sh 5b plan /shared/wm3d/control/h200_5b.env
```

`doctor` 应打印：

- `model=native_5b expected_parameters=5,108,342,963`；
- `world_size=128 total_steps=10000`；
- Python `pip check` 通过；
- 每张卡是 H200、约 140 GB 显存、ECC 为 0；
- 数据 profile 若尚未交付，会明确显示 `WAITING`。

## 5. 数据准备

### 5.1 锁 revision 并下载

先在一台登录节点或数据节点运行：

```bash
SITE=/shared/wm3d/control/h200_5b.env
./run_wm3d.sh 5b lock "$SITE"
./run_wm3d.sh 5b download "$SITE"
```

`lock` 把每个 Hugging Face source 的 revision 解析成 40 位 commit。`download` 支持断点
重入；已验证的文件会跳过，不会默默覆盖不同内容。下载结束后检查：

```bash
source "$SITE"
test -s "$SOURCE_LOCK"
find "$RAW_ROOT" -name download_receipt.json -type f | sort
df -h "$RAW_ROOT"
```

如果下载因为 gated dataset 失败，先确认该账号接受了许可、token 有 read 权限，再重跑同一
命令。不要把 revision 改成 `main` 或 `latest`。

### 5.2 schema、adapter 和 inventory

这一步只需对每个冻结 source revision 做一次，但必须由懂机器人字段的人签字。标准顺序是：

```text
schema-audit -> adapter-audit -> inventory -> data-profile
```

低层命令及 AgiBot converter 见 [从零手册](WM3D_FROM_ZERO.md)。操作者应向项目负责人交付
schema candidate；项目负责人确认以下内容后返回一个 sealed `DATA_PROFILE`：

- 每个相机文件映射到 `head / left_wrist / right_wrist` 中哪个槽位；
- action/state 每一列的含义、单位和坐标系；
- gripper 0/1 极性；
- fine command 与 coarse effect 哪一条 lane 有监督；
- 原始时间戳和 episode 边界。

把返回的文件放到 site 文件中的 `DATA_PROFILE` 路径。以下命令必须成功：

```bash
./run_wm3d.sh 5b doctor "$SITE"
```

如果你只做小数据验证，项目负责人可以在 `inventory` 阶段用
`--episode-index-file` 提供确定性 episode 白名单。不能从完整 manifest 临时 `head -n`
制造子集，因为那会破坏 source receipt 和 split 闭包。

### 5.3 task bank 和 episode cache

先用一张 GPU 构建任务文本 embedding，再生成每个 episode 的确定性任务清单：

```bash
./run_wm3d.sh 5b task-bank "$SITE"
./run_wm3d.sh 5b cache-plan "$SITE"
./run_wm3d.sh 5b status "$SITE"
```

随后每张 GPU 启一个长生命周期 cache worker。手工多节点示例：

```bash
# 每台节点分别设置 NODE_RANK=0..15，然后运行：
source "$SITE"
for LOCAL_GPU in $(seq 0 $((GPUS_PER_NODE - 1))); do
  GLOBAL_WORKER=$((NODE_RANK * GPUS_PER_NODE + LOCAL_GPU))
  ./run_wm3d.sh 5b cache-worker "$SITE" \
    "$GLOBAL_WORKER" "$CACHE_WORKER_COUNT" "$LOCAL_GPU" \
    >"$CACHE_ROOT/worker_${GLOBAL_WORKER}.log" 2>&1 &
done
wait
```

Slurm 示例：

```bash
export SITE=/shared/wm3d/control/h200_5b.env
srun --nodes=16 --ntasks=128 --ntasks-per-node=8 --gpus-per-task=1 \
  --export=ALL,SITE bash -lc '
  GLOBAL_WORKER=$SLURM_PROCID
  LOCAL_GPU=$SLURM_LOCALID
  # Slurm 已把每个 task 限定到一张 GPU；保留它的 CUDA_VISIBLE_DEVICES。
  ./run_wm3d.sh 5b cache-worker "$SITE" "$GLOBAL_WORKER" 128 inherited
'
```

worker 是安全重入的。某台机器重启后，使用相同 worker index/count 重跑即可；已存在且
SHA 一致的 episode 会显示 `already_complete`。任何 worker 的 `failed` 必须为 0。

### 5.4 seal、window、normalization 和 runtime

全部 worker 正常退出后，在一台节点运行：

```bash
./run_wm3d.sh 5b cache-seal "$SITE"
./run_wm3d.sh 5b window "$SITE"
./run_wm3d.sh 5b normalization "$SITE"
./run_wm3d.sh 5b runtime "$SITE"
./run_wm3d.sh 5b status "$SITE"
```

正确结果应满足：

- cache task、episode、window 都不是 0；
- train 和 val 都有 window；
- episode/window seal 显示 `PASS`；
- runtime 显示 `native_5b / 5,108,342,963 params / world 128`；
- 没有 `NaN`、`SHA mismatch`、`missing source` 或 `adapter mismatch`。

5B 与 1B 可共用同一 episode cache，但 window index 绑定模型的 `T/K/horizon`，所以 5B
必须生成自己的 window index，不能直接拿 1B 的 window seal。

## 6. 10K 验证性训练

10K 指 10,000 个 optimizer steps，不是随意截取 10,000 行数据。全局 batch 是 128，
因此完整验证约消费 128 万个确定性采样位置。建议按 100 → 500 → 10,000 三段运行：前
两段快速证明 forward/backward、gradient ownership、DCP 和独立进程 exact resume；最后
一段看稳定性和吞吐。

### 6.1 调度器需要做什么

所有节点运行同一命令，只让 `NODE_RANK` 不同。Slurm 示例：

```bash
export SITE=/shared/wm3d/control/h200_5b.env
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)

srun --nodes=16 --ntasks=16 --ntasks-per-node=1 \
  --export=ALL,SITE,MASTER_ADDR bash -lc '
  export NODE_RANK=$SLURM_PROCID
  ./run_wm3d.sh 5b preflight "$SITE"
'
```

preflight 必须在每次 fresh/resume/eval 启动前重新运行，因为资源 receipt 只有 30 分钟
有效。它会验证 GPU 型号、显存、ECC、空闲状态、NVLink、IB、memlock、共享内存、磁盘
空间和真实 all-reduce。

### 6.2 fresh 0 → 100

```bash
srun --nodes=16 --ntasks=16 --ntasks-per-node=1 \
  --export=ALL,SITE,MASTER_ADDR bash -lc '
  export NODE_RANK=$SLURM_PROCID
  ./run_wm3d.sh 5b train "$SITE" 100
'
```

等所有 16 个 launcher 完全退出。确认 checkpoint 是完整编号目录：

```bash
source "$SITE"
test -f "$RUN_ROOT/checkpoints/step_00000100/COMMITTED.json"
./run_wm3d.sh 5b status "$SITE"
```

### 6.3 新进程 exact resume 100 → 500

先重新 preflight，再启动新进程：

```bash
CKPT100=/shared/wm3d/runs/5b_validation10k/checkpoints/step_00000100
srun --nodes=16 --ntasks=16 --ntasks-per-node=1 \
  --export=ALL,SITE,MASTER_ADDR bash -lc '
  export NODE_RANK=$SLURM_PROCID
  ./run_wm3d.sh 5b preflight "$SITE"
'
srun --nodes=16 --ntasks=16 --ntasks-per-node=1 \
  --export=ALL,SITE,MASTER_ADDR bash -lc '
  export NODE_RANK=$SLURM_PROCID
  ./run_wm3d.sh 5b resume "$SITE" '"$CKPT100"' 500
'
```

日志和 checkpoint metadata 必须显示从 step 100 恢复，新的 committed checkpoint 是
`step_00000500`。不要使用 `latest` 符号链接。

### 6.4 500-step 离线评测

```bash
CKPT500=/shared/wm3d/runs/5b_validation10k/checkpoints/step_00000500
srun --nodes=16 --ntasks=16 --ntasks-per-node=1 \
  --export=ALL,SITE,MASTER_ADDR bash -lc '
  export NODE_RANK=$SLURM_PROCID
  ./run_wm3d.sh 5b preflight "$SITE"
'
srun --nodes=16 --ntasks=16 --ntasks-per-node=1 \
  --export=ALL,SITE,MASTER_ADDR bash -lc '
  export NODE_RANK=$SLURM_PROCID
  ./run_wm3d.sh 5b eval "$SITE" '"$CKPT500"' \
    /shared/wm3d/runs/5b_validation10k/eval_step_00000500.json
'

./run_wm3d.sh 5b verify "$SITE" 500 "$CKPT500" \
  /shared/wm3d/runs/5b_validation10k/eval_step_00000500.json
```

500-step 验收通过后，再以相同方式从 500 恢复到 10,000：

```bash
srun --nodes=16 --ntasks=16 --ntasks-per-node=1 \
  --export=ALL,SITE,MASTER_ADDR bash -lc '
  export NODE_RANK=$SLURM_PROCID
  ./run_wm3d.sh 5b preflight "$SITE"
'
srun --nodes=16 --ntasks=16 --ntasks-per-node=1 \
  --export=ALL,SITE,MASTER_ADDR bash -lc '
  export NODE_RANK=$SLURM_PROCID
  ./run_wm3d.sh 5b resume "$SITE" '"$CKPT500"' 10000
'
CKPT10K=/shared/wm3d/runs/5b_validation10k/checkpoints/step_00010000
srun --nodes=16 --ntasks=16 --ntasks-per-node=1 \
  --export=ALL,SITE,MASTER_ADDR bash -lc '
  export NODE_RANK=$SLURM_PROCID
  ./run_wm3d.sh 5b preflight "$SITE"
'
srun --nodes=16 --ntasks=16 --ntasks-per-node=1 \
  --export=ALL,SITE,MASTER_ADDR bash -lc '
  export NODE_RANK=$SLURM_PROCID
  ./run_wm3d.sh 5b eval "$SITE" '"$CKPT10K"'
'
./run_wm3d.sh 5b verify "$SITE" 10000 "$CKPT10K" \
  /shared/wm3d/runs/5b_validation10k/eval_step_00010000.json
```

torchrun 参数由 `5b` 包装器从 site 文件统一生成。以上所有 `srun` 都是一节点一 launcher；
每个 launcher 再启动本机 8 个 GPU process。

## 7. 怎么看训练是否正确

### 7.1 一条命令看状态

```bash
./run_wm3d.sh 5b status "$SITE"
```

训练完成后使用严格验收：

```bash
./run_wm3d.sh 5b verify "$SITE" 10000 "$CKPT10K" "$EVAL_OUTPUT"
```

输出示例：

```text
WM3D 5B pipeline: PASS
  data: public_robot_6106h_expanded (9 sources)
  cache: 420,000 episodes, seal=PASS
  windows: 8,100,000, seal=PASS
  model: native_5b / 5,108,342,963 params / world 128
  train: step 10000, total=..., grad_norm=...
  throughput: ... samples/s
  gradients: PASS (10 required owners)
  checkpoint: PASS step=10000 size=... GiB
  eval: PASS coverage_lanes=...
```

`PASS` 的含义是：数据闭包、梯度、数值、checkpoint、resume 和 eval coverage 正确。它不
等于模型能力已经达到目标。

严格 `verify` 会重新读取并计算 DCP payload SHA。5B checkpoint 约几十 GB，这一步可能
持续数分钟；不要因为它比 `status` 慢而跳过最终验收。

### 7.2 必看文件

| 文件 | 正确时应该看到什么 |
|---|---|
| `$RUN_ROOT/train_metrics.jsonl` | step 单调递增；`total` 和各 loss 有限；无 NaN/Inf |
| `$RUN_ROOT/gradient_ownership.json` | `passed=true`；所有 required owner 非零且 nonfinite=0 |
| `checkpoints/step_XXXXXXXX/COMMITTED.json` | 文件存在；报告显示 checkpoint PASS |
| eval receipt | `all_metrics_finite=true`；每个 expected coverage lane > 0 |
| launch qualification | 绑定本次 runtime、GPU UUID、world size 和 source checkpoint |

建议实时观察：

```bash
tail -f "$RUN_ROOT/train_metrics.jsonl"
watch -n 2 nvidia-smi
```

资源利用率应满足：8 张节点内 GPU 都有训练进程，训练稳定阶段不是长期 0% util；NVLink
clique 和 IB all-reduce 在 preflight 通过；若 GPU 经常等待，先检查 raw/cache 是否在本地
高吞吐存储、DataLoader worker 是否被限速，以及共享存储延迟，而不是先改模型。

### 7.3 损失怎么看

需要同时检查，而不是只看 `total`：

- `token_mse`：native future token；
- `rgb`、`depth`、`point`、`pose`：显式视觉和 3D 输出；
- `action_fine` / `action_coarse`：profile 声明的 action lane；
- `grad_norm`：必须有限；
- validation metrics：不能全部因为 mask 为空而变成 0，eval coverage 会阻止这种假通过。

短跑的 loss 不保证单调下降。500/10K validation 的正确判断是数值有限、监督 coverage
非零、必要梯度 owner 全部通过、吞吐稳定、resume 精确、validation 没有系统性爆炸。模型
效果要用更长训练和下游任务另行判断。

## 8. 通过 10K 后怎么扩到正式训练

复制 site 文件到新路径，至少修改 `RUNTIME_PROFILE`、`RUN_NAME`、`RUN_LINEAGE`、
`RUN_ROOT`、`RUNTIME_YAML` 和 `EVAL_OUTPUT`。不要覆盖 10K runtime/run。

- 100K validation：`configs/runtime/h200_128_fsdp2_validation100k.yaml`
- 600K formal：`configs/runtime/h200_128_fsdp2.yaml`

每个新 run 都重新执行 `runtime -> preflight -> train`。episode cache 可以复用；只要模型
profile 不变，5B window index 和 normalization 也可复用。正式 600K 前应保存 10K/100K
验收报告、吞吐、峰值显存和 checkpoint 空间测量。

## 9. 常见失败

- `data_profile=WAITING`：项目负责人还没交付 adapter/inventory closure；不能训练。
- `source revision ...`：source lock 未物化为 40 位 SHA，重新执行 `lock`。
- `preflight receipt stale`：每次 launch 前重新跑 preflight。
- `GPU busy` / ECC 非零 / NVLink 或 IB 不达标：换空闲节点或修基础设施，不要绕过门禁。
- `--stop-after-step must be a sealed checkpoint step`：只能停在 runtime 规定的 checkpoint
  step；10K profile 支持 100、500、1000、2000……10000。
- `COMMITTED.json` 不存在：checkpoint 未完成，禁止 resume/eval。
- cache worker `failed > 0`：修首个 episode 错误并用相同 worker partition 重跑。
- `zero coverage`：val split 或 adapter supervision lane 错误，不能把全 0 loss 当成功。
- OOM：先确认使用 8-way FSDP2、bf16 和 activation checkpoint；不要靠减小 token/action
  ABI 掩盖 profile 漂移。

底层数据审计和 converter 命令见 [从零手册](WM3D_FROM_ZERO.md)，模型规模与 FSDP2 原理见
[统一训练与扩展](WM3D_SCALING.md)，发布证据边界见
[发布验收](WM3D_RELEASE_VALIDATION.md)。
