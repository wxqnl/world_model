# WM3D 5B 训练流程

WM3D 5B 使用 `configs/model/native_5b.yaml`，参数量约 51 亿。默认训练规模为
8 个节点、每节点 8 张 H200，共 64 张 GPU。模型在每个节点内做 8-way FSDP2
分片，8 个节点组成 data-parallel replicas。每张卡的 micro batch 为 4，不做梯度累积，
global batch 为 256。

整个流程按数据下载、数据整理、数据访问准备、1K 集群验证和正式训练
依次进行。数据访问可以使用完整 episode cache，也可以使用有容量上限的按需缓存。命令会
自动记录和校验中间产物，使用者只需要指定数据目录、数据许可、模型目录和集群地址。

## 1. 训练预设

| preset | optimizer steps | 用途 |
|---|---:|---|
| `canary1k` | 1,000 | 验证 64 卡通信、训练、保存、恢复和评测 |
| `validation100k` | 100,000 | 可选的中程训练，不是正式训练前置条件 |
| `formal600k` | 600,000 | 正式训练 |

三套预设使用同一个 5B 模型、data profile 和数据访问方式。它们各自生成独立的 runtime
和 checkpoint，不能把不同 preset 的 checkpoint 混在一起恢复。

默认 global batch 为 256，因此 1K、100K 和 600K 分别对应约 25.6 万、2,560 万和
1.536 亿个全局采样位置。

## 2. 服务器与存储

默认集群配置：

- 8 个 8×H200 SXM 节点，节点内 NVLink；
- 400 Gb/s InfiniBand；
- Linux x86_64、Python 3.10、CUDA 12.8；
- 所有节点都能访问代码、原始数据、cache 和训练输出；底层可以是共享存储，也可以是各节点
  独立存储并由站点配置映射；
- 数据盘应能持续供给训练和 cache 构建所需的读写吞吐；
- 按第 4.1 节的总量规划磁盘，下载前用 `df -h` 确认剩余容量。

克隆项目：

```bash
git clone --branch v8 --single-branch https://github.com/wxqnl/world_model.git
cd world_model/wm3d
```

创建 1K 验证站点配置和 Python 环境：

```bash
SITE=/data/wm3d/control/5b_canary1k.env
./run_wm3d.sh 5b init canary1k "$SITE"
vim "$SITE"
./run_wm3d.sh 5b env "$SITE"
./run_wm3d.sh 5b data-template "$SITE"
./run_wm3d.sh 5b doctor "$SITE"
./run_wm3d.sh 5b plan "$SITE"
```

站点文件至少需要修改：

```bash
WORK_ROOT=/data/wm3d
HF_TOKEN_FILE=/data/secrets/huggingface_token
ACCEPT_DATA_LICENSES=YES
INCLUDE_AGIBOT_BETA=NO
MASTER_ADDR=TRAIN_NODE_0
WM3D_VGGT_SOURCE_ROOT=/data/models/vggt
WM3D_VGGT_MODEL_SNAPSHOT=/data/models/facebook-VGGT-1B
QWEN3_VL_EMBEDDING_PATH=/data/models/Qwen3-VL-Embedding-2B
```

默认值已经设置为 `NNODES=8`、`GPUS_PER_NODE=8`，不需要再改 world size。

## 3. 数据准备

### 3.1 默认数据组合

默认 `DATA_FAMILY=public_robot_oxe`。项目保留 DROID、Bridge、RoboCasa365 和
AgiBotWorld2026，并加入完整的
[LeRobot Open X-Embodiment collection](https://huggingface.co/collections/lerobot/open-x-embodiment-68de658d8b544a43be4c6687)。
AgiBotWorld Beta 默认关闭。

| 数据源 | 公开仓库 | 默认状态 | 规划时长 | `$RAW_ROOT/` 下的目录 |
|---|---|---|---:|---|
| DROID | [`lerobot/droid_1.0.1`](https://huggingface.co/datasets/lerobot/droid_1.0.1) | 使用 | 约 350 h | `droid` |
| Bridge V2 | [`ember-lab-berkeley/bridge_v2`](https://huggingface.co/datasets/ember-lab-berkeley/bridge_v2) | 使用 | 约 100 h | `bridge` |
| RoboCasa365 Atomic | [`ember-lab-berkeley/robocasa365-pretrain-atomic`](https://huggingface.co/datasets/ember-lab-berkeley/robocasa365-pretrain-atomic) | 使用 | 约 21 h | `atomic` |
| RoboCasa365 Composite | [`ember-lab-berkeley/robocasa365-pretrain-composite`](https://huggingface.co/datasets/ember-lab-berkeley/robocasa365-pretrain-composite) | 使用 | 约 383 h | `composite` |
| RoboCasa365 MG | [`ember-lab-berkeley/robocasa365-pretrain-mg`](https://huggingface.co/datasets/ember-lab-berkeley/robocasa365-pretrain-mg) | 使用 | 约 1,615 h | `mg` |
| AgiBotWorld2026 真机数据 | [`agibot-world/AgiBotWorld2026`](https://huggingface.co/datasets/agibot-world/AgiBotWorld2026) | 使用 | 约 661 h | `agibot_world_2026` |
| OXE | [LeRobot OXE collection](https://huggingface.co/collections/lerobot/open-x-embodiment-68de658d8b544a43be4c6687) | 使用，DROID 去重 | 当前约 97.3 h | `oxe/<dataset>` |
| AgiBotWorld Beta | [`agibot-world/AgiBotWorld-Beta`](https://huggingface.co/datasets/agibot-world/AgiBotWorld-Beta) | 可选，默认关闭 | 约 2,976.4 h | `agibot_beta` |

当前官方清单包含 56 套 OXE 数据，其中 DROID 已作为主数据使用，因此默认新增 55 个 OXE
source。默认组合共 63 个训练 source，规划时长约 3,227 小时。官方 collection 发生变化时，
source 数量以 `data-template` 生成结果为准。

AgiBotWorld2026 只使用 Imitation Learning、Rich Interaction 和 Reinforcement Learning
三类真机数据，不使用其 Simulation 部分。表中的时长用于容量规划；最终训练量以整理后的
episode、frame 和 window 数量为准。

### 3.2 生成默认数据模板

site 文件默认配置为：

```bash
DATA_FAMILY=public_robot_oxe
INCLUDE_AGIBOT_BETA=NO
```

运行：

```bash
./run_wm3d.sh 5b data-template "$SITE"
```

该命令读取官方 OXE collection 和每个数据集的 `meta/info.json`，生成 source template、data
template 和 adapter 候选。DROID 自动去重。现有主数据权重保持不变；每个新增 OXE 数据集
作为普通 source 加入，权重为 1。OXE 不再拥有固定的整体采样比例。

如需加入 AgiBotWorld Beta，在第一次生成模板前设置：

```bash
INCLUDE_AGIBOT_BETA=YES
```

启用后，生成器同时加入 Beta 和官方 Alpha converter。已经生成的模板不会被不同配置覆盖；
更改该选项时应使用新的 site 和 `CONTROL_ROOT`。

当前默认组合的规模和 cache 预算如下：

| 组成 | 数据变化 | 数据规模 | 预计 episode cache |
|---|---|---:|---:|
| 保留的主数据 | DROID、Bridge、RoboCasa、AgiBotWorld2026 | 约 3,130 h | 约 105.5 TB |
| OXE | 新增除 DROID 外的 55 个 source | 约 97.3 h 原始记录；约 264.6 万个 cache 帧 | 约 2.5 TB |
| **默认组合合计** | **63 个训练 source，不含 AgiBotWorld Beta** | **约 3,227 h 原始记录** | **约 108 TB（约 98 TiB）** |

cache 估算按正式的三视角 `native_p144` 表征、每个视角 144 个 token、最高约 10 Hz
保留观测计算，平均每个被保留的观测约占 0.936 MB。OXE 中许多数据是 20 Hz 或 50 Hz，
因此会按真实时间戳降到最高约 10 Hz；动作和状态仍保留原始时间戳，不做固定频率插值。

`108 TB` 是最终 episode cache 本身。把原始下载、下载缓存、临时文件、日志和 checkpoint
一并计算后，默认组合的完整 cache 方案建议准备 **130–150 TB 总磁盘**。启用 AgiBotWorld
Beta 后，episode cache 预计增加到约 209 TB，总磁盘也需要相应增加。

生成模板后照常执行 `lock`、`download`、schema audit、adapter audit 和 inventory。某个数据集
的动作或状态维度超过 WM3D 容量时，生成步骤会停止并报告数据集名称。

### 3.3 下载

先准备 Hugging Face token：

```bash
install -d -m 700 /data/secrets
umask 077
read -rsp "Hugging Face token: " HF_TOKEN
printf '%s\n' "$HF_TOKEN" > /data/secrets/huggingface_token
unset HF_TOKEN
chmod 600 /data/secrets/huggingface_token
```

下载命令：

```bash
./run_wm3d.sh 5b lock "$SITE"
./run_wm3d.sh 5b download "$SITE"
```

`lock` 自动固定所有数据版本，`download` 支持断点续传。确认下载是否完成：

```bash
source "$SITE"
find "$RAW_ROOT/receipts" -name '*.json' -type f | wc -l
du -sh "$RAW_ROOT"/*
df -h "$RAW_ROOT"
```

### 3.4 数据整理

RoboCasa 和 AgiBotWorld 2026 先按 [WM3D 从零数据流程](WM3D_FROM_ZERO.md) 中的
archive、schema、adapter、inventory 命令整理。OXE 数据已经是 LeRobot 格式；生成命令会为
每套数据保留其官方相机键、action/state 维度和原始时间戳，并生成 opaque controller adapter
候选。负责人仍需按同一数据流程完成 schema、adapter audit、inventory 和 data-profile，不能
把候选文件直接当成已审计 adapter。当前默认配置应包含 63 个训练 source；启用 Beta 后为
64 个。source 数量随官方 OXE collection 调整，每个 source 都必须有非空 episode 数量。

完成后运行：

```bash
./run_wm3d.sh 5b doctor "$SITE"
./run_wm3d.sh 5b plan "$SITE"
```

输出应显示 `native_5b`、64 个 rank，以及与所选配置一致的数据 source 数量，并且不再
出现 `data_profile=WAITING`。

## 4. 高吞吐 episode cache

cache 是数据准备中最耗时的阶段。默认配置按 64 张 GPU 全量展开：每张 GPU 一个长驻 worker，
每个 worker 使用 4 个视频解码线程、`batch_frames=16` 的 VGGT 前向和 2 个写盘线程。
worker 会流水执行“准备下一个 episode、GPU 编码当前 episode、写盘上一个 episode”，避免
解码和落盘期间 GPU 空转。

先构建任务：

```bash
./run_wm3d.sh 5b task-bank "$SITE"
./run_wm3d.sh 5b cache-plan "$SITE"
```

用 Slurm 在 64 张卡上启动 cache：

```bash
export SITE=/data/wm3d/control/5b_canary1k.env
srun --nodes=8 --ntasks=64 --ntasks-per-node=8 --gpus-per-task=1 \
  --cpus-per-task=8 --cpu-bind=cores \
  --export=ALL,SITE bash -lc '
  ./run_wm3d.sh 5b cache-worker "$SITE" \
    "$SLURM_PROCID" 64 inherited
'
```

站点文件中的默认性能参数为：

```bash
CACHE_WORKER_COUNT=64
CACHE_BATCH_FRAMES=16
CACHE_DECODE_WORKERS=4
CACHE_WRITER_THREADS=2
```

H200 显存足够时保持这组默认值。只有出现单卡 OOM 时，才把 `CACHE_BATCH_FRAMES` 降为 12
或 8；不要先减少 worker 数量。写入吃不满时，先检查存储带宽、挂载方式和小文件性能，再
考虑增加写线程，避免在慢盘上盲目堆线程。

运行期间查看：

```bash
watch -n 2 nvidia-smi
iostat -xz 2
./run_wm3d.sh 5b status "$SITE"
```

正常表现是 64 张 GPU 都有一个 cache 进程，warm-up 后 GPU 周期性处于高利用率，worker
持续输出 `prepare_seconds`、`encode_seconds`、`write_seconds`、`tasks_per_second`，且
`failed` 始终为 0。若 `prepare_seconds` 明显高于编码时间，检查 CPU 绑定和视频盘读取；若
`write_seconds` 持续堆积，检查 cache 文件系统带宽。

worker 可重入。任务中断后用完全相同的 worker count 重跑，已经完成的 episode 会自动跳过。
全部 worker 退出后执行：

```bash
./run_wm3d.sh 5b cache-seal "$SITE"
./run_wm3d.sh 5b window "$SITE"
./run_wm3d.sh 5b normalization "$SITE"
./run_wm3d.sh 5b runtime "$SITE"
./run_wm3d.sh 5b status "$SITE"
```

### 4.1 几十 TB 磁盘：按需缓存训练

磁盘无法容纳完整视觉 cache 时，使用 `streaming_raw`。这个模式不会在每一步重复处理
原始视频。它先生成全量的轻量 metadata 和窗口索引，训练第一次访问某个 episode 时才解码
视频并运行冻结的 VGGT，然后把生成的标准 episode cache 放进有容量上限的 LRU。后续窗口直接读取
这份缓存；达到容量上限后，LRU 只淘汰最久未使用的 episode。

只有几十 TB 磁盘时，使用默认数据组合并启用 `streaming_raw`。DROID、Bridge、RoboCasa、
AgiBotWorld 2026 和 OXE 都参与训练，按需缓存只改变数据访问方式。

两种数据访问方式使用相同的 data profile、采样权重、模型输入和训练目标。下表是 64×H200、
默认 OXE 组合的容量与排期预算。`300K 等样本预算`与旧 global batch 128 下的 600K
拥有相同的 7,680 万个全局采样位置；当前正式预设仍训练 600K steps，共 1.536 亿个采样位置。

| 数据访问方式 | 建议总磁盘 | 相对训练吞吐 | 300K 等样本预算 | 600K 正式训练 | 选择条件 |
|---|---:|---:|---:|---:|---|
| `episode_cache` | 约 130–150 TB | 1.0 | 约 15–23 天 | 约 30–45 天 | 磁盘充足，优先训练吞吐 |
| `streaming_raw` | 约 35–45 TB | 约 0.75–0.9 | 约 20–28 天 | 约 40–55 天 | 只有几十 TB，接受一定速度损失 |

吞吐按完整 cache 为 `1.0`。LRU 命中后的读取速度约为完整 cache 的 `0.95–1.0`；长期平均还要
计入 episode 第一次解码和 VGGT 编码，因此约为 `0.75–0.9`。换算成训练耗时，按需缓存通常是
完整 cache 的 `1.1–1.35` 倍。短 episode 较多、原始视频盘较慢或 LRU 频繁淘汰时，差距会扩大。

micro batch 从 1 调到 4、梯度累积从 2 调到 1 后，每个 rank 每个 optimizer step 的有效
样本数从 2 增加到 4，global batch 从 128 增加到 256。这使固定样本预算所需的 optimizer
steps 减半，但不是 8 倍墙钟加速。当前 `formal600k` 没有减少 steps，而是把训练样本预算
翻倍，因此 600K 排期不能直接除以 2 或 8。

这些时间是排期估算，不代替 64 卡 canary。64 卡实际跑完前 100–500 steps 后，用日志中的
单步中位时间更新正式排期：

```text
预计天数 = 剩余 steps × 单步中位秒数 ÷ 86400 × 1.08
```

按需缓存的排期参考：

| 训练预设 | 预计用时 |
|---|---:|
| 1K canary | 2–3 小时 |
| 100K | 6–9 天 |
| 600K 正式训练 | 40–55 天，通常按约 45 天安排，资源窗口预留 55 天 |

`streaming_raw` 去掉的是约 108 TB 的完整视觉 cache，原始数据仍需保留。当前上游规模下，
默认 OXE 组合的原始数据预计约 15–20 TB。总磁盘按下面的项目规划：

| 项目 | 预计空间 | 使用阶段 |
|---|---:|---|
| 原始数据和物化后的训练数据 | 15–20 TB | 全程保留 |
| 下载、转换临时文件 | 峰值额外 5–10 TB | 数据确认后清理 |
| metadata、task bank、日志和临时输出 | 预留 1–2 TB | 全程 |
| 10 个完整 5B checkpoint | 约 1 TB | 训练期间 |
| 有容量上限的 episode LRU | 建议预留 2–4 TB | 训练期间，可按可用空间调整 |
| **建议总磁盘** | **约 35–45 TB** | 覆盖准备阶段峰值并保留运行余量 |

因此实际规划按 **40 TB 左右** 准备最稳妥，不要按清理临时文件后的理论最低值采购。LRU
建议从总量 2–4 TB 起步；磁盘更紧时可以缩小，代价是 episode 淘汰增加、吞吐下降。LRU
可以放在共享存储或节点本地存储，优先选择现场速度最快且容量稳定的路径。程序会按主机名和
global rank 自动隔离缓存目录。

在 site 文件中设置：

```bash
WM3D_DATA_MODE=streaming_raw
STREAMING_METADATA_ROOT=/data/wm3d/streaming_metadata/native_p144
STREAMING_LRU_ROOT=/data/wm3d/streaming_lru
STREAMING_LRU_GIB_PER_RANK=64
STREAMING_METADATA_WORKERS=32
STREAMING_ENCODE_BATCH_FRAMES=16
STREAMING_DECODE_WORKERS=4
```

以上路径和容量是起始建议，不是存储架构要求。`STREAMING_METADATA_ROOT` 保存时间戳、动作、
状态、任务向量和窗口索引，不保存 VGGT 视觉 token、depth、point 或 RGB pack。
`STREAMING_LRU_GIB_PER_RANK` 可按总磁盘空间调整；`STREAMING_METADATA_WORKERS=32` 用于并行
扫描 episode，存储和 CPU 还有余量时可以提高到 64。

下载、adapter audit、inventory 和 data profile 与完整 cache 模式相同。完成这些步骤后执行：

```bash
./run_wm3d.sh 5b task-bank "$SITE"
./run_wm3d.sh 5b cache-plan "$SITE"
./run_wm3d.sh 5b streaming-prepare "$SITE"
./run_wm3d.sh 5b runtime "$SITE"
./run_wm3d.sh 5b status "$SITE"
```

`cache-plan` 在这里仅生成 episode 任务清单，不会生成视觉 cache。这个模式不运行
`cache-worker`、`cache-seal`、`window` 和 `normalization`；`streaming-prepare` 一次性生成
轻量 metadata、窗口和归一化统计。采样器按 episode 连续取窗口，减少冷缓存切换。

之后的 64 卡启动方式与完整 cache 模式相同。先使用第 5 节定义的 `run_5b` 函数，然后按
独立进程执行：

```bash
run_5b preflight
run_5b train 100

run_5b preflight
run_5b resume 100 500

run_5b preflight
run_5b eval 500
./run_wm3d.sh 5b verify "$SITE" 500
```

第一次训练进程会产生冷缓存。恢复进程继续使用已有 LRU 时应当看到缓存命中。更换节点或
缓存路径不会影响 checkpoint 正确性，但未命中的 episode 需要重新生成。

训练日志中会增加 `streaming_raw` 字段：

- `generated_episodes`：本进程从原始数据生成的 episode 数；
- `cache_hits`：命中 LRU 的次数；
- `evicted_episodes`：因容量上限被删除的 episode 数；
- `prepare_seconds`、`encode_seconds`：累计解码准备和 VGGT 编码时间；
- `resident_bytes`、`resident_episodes`：当前 LRU 占用。

正常运行应满足：

- 冷启动时 `generated_episodes` 增加，`prepare_seconds` 和 `encode_seconds` 为有限值；
- 进入同一 episode 的后续窗口后，`cache_hits` 持续增加，准备和编码时间不再重复增加；
- `resident_bytes` 不超过每 rank 的容量上限；
- `evicted_episodes` 不应每个 step 都快速增加，否则应扩大 LRU 或检查采样是否保持 episode 连续；
- checkpoint、独立进程恢复和离线评测与完整 cache 模式使用相同的通过标准。

性能调优顺序是缓存盘吞吐、LRU 容量、CPU/视频读取带宽、解码线程和 VGGT batch。默认每 GPU
使用 4 个解码线程、`batch_frames=16`。`prepare_seconds` 长期高于 `encode_seconds` 时，先增加
CPU 和原始视频盘带宽；VGGT OOM 时把 `STREAMING_ENCODE_BATCH_FRAMES` 调到 12 或 8。
冷缓存阶段 GPU 利用率会随视频准备产生波动，LRU 命中后的训练阶段应恢复稳定。

## 5. 1K 集群验证

1K canary 使用与正式训练相同的 64 卡拓扑，验证通信、5B 参数分片、前向、反向、梯度、
checkpoint、独立进程恢复和离线评测。

```bash
SITE=/data/wm3d/control/5b_canary1k.env
./run_wm3d.sh 5b init canary1k "$SITE"
vim "$SITE"
./run_wm3d.sh 5b doctor "$SITE"
./run_wm3d.sh 5b runtime "$SITE"
```

多节点启动函数：

```bash
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)

run_5b () {
  srun --nodes=8 --ntasks=8 --ntasks-per-node=1 \
    --export=ALL,SITE,MASTER_ADDR bash -lc \
    "./run_wm3d.sh 5b $1 \"$SITE\" ${2:-} ${3:-}"
}
```

按三个独立进程完成 canary：

```bash
run_5b preflight
run_5b train 100

run_5b preflight
run_5b resume 100 500

run_5b preflight
run_5b resume 500 1000

run_5b preflight
run_5b eval 1000
./run_wm3d.sh 5b verify "$SITE" 1000
```

`verify` 通过后可以开始 600K 正式训练。不要跳过独立进程 resume，因为它能发现只在恢复时
出现的模型、optimizer、sampler 或分布式状态问题。

## 6. 正式训练

1K 通过后创建 600K site。需要观察中程曲线时可以另建 100K site，但它不是正式训练的
前置条件。两者复用 data profile 和已选择的数据访问方式；
`episode_cache` 复用完整 cache，`streaming_raw` 复用 metadata 和有容量上限的 LRU。每个 preset
仍从自己的 step 0 开始训练。

```bash
./run_wm3d.sh 5b init validation100k /data/wm3d/control/5b_validation100k.env
./run_wm3d.sh 5b init formal600k /data/wm3d/control/5b_formal600k.env
```

100K：

```bash
SITE=/data/wm3d/control/5b_validation100k.env
./run_wm3d.sh 5b runtime "$SITE"
run_5b preflight
run_5b train 1000
run_5b preflight
run_5b resume 1000 100000
run_5b preflight
run_5b eval 100000
./run_wm3d.sh 5b verify "$SITE" 100000
```

600K：

```bash
SITE=/data/wm3d/control/5b_formal600k.env
./run_wm3d.sh 5b runtime "$SITE"
run_5b preflight
run_5b train 1000
run_5b preflight
run_5b resume 1000 5000
run_5b preflight
run_5b resume 5000 20000
run_5b preflight
run_5b resume 20000 600000
run_5b preflight
run_5b eval 600000
./run_wm3d.sh 5b verify "$SITE" 600000
```

## 7. 常见问题

| 现象 | 处理 |
|---|---|
| `data_profile=WAITING` | 数据尚未完成 schema、inventory 和 profile 物化 |
| 下载 401/403 | 先在上游页面接受许可，再检查 token 文件 |
| cache 中 GPU 经常为 0% | 检查是否启动了 64 个 worker、CPU 绑定、原始视频盘吞吐和 batch size |
| cache 写盘积压 | 检查存储带宽、挂载方式和小文件性能 |
| cache OOM | 先将 `CACHE_BATCH_FRAMES` 从 16 降到 12 或 8 |
| streaming 冷缓存反复生成 | 提高 LRU 容量；恢复时尽量复用原缓存路径和 rank 布局 |
| streaming 中 GPU 长时间空闲 | 检查原始视频盘、CPU 绑定和 `STREAMING_DECODE_WORKERS`；再观察 `prepare_seconds` |
| 训练 OOM | 先检查 8-way FSDP2、BF16、activation checkpoint 和 64 卡 topology；确认无误后将 micro batch 从 4 降为 2，并同步把 global batch 改为 128 |
| GPU busy、ECC、NVLink、IB 失败 | 更换节点或修复资源后重新运行 preflight |
| 恢复失败 | 只从完整编号 checkpoint 恢复，并重新运行 preflight |
| eval coverage 为 0 | 检查验证集和对应 action/视觉 supervision 是否真实存在 |

模型与分布式原理见 [WM3D 统一训练与扩展](WM3D_SCALING.md)，数据底层命令见
[WM3D 从零数据流程](WM3D_FROM_ZERO.md)，Stage1 见
[WM3D Stage1](WM3D_STAGE1_UNIFIED.md)。
