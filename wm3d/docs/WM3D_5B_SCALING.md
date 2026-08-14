# WM3D 5B 训练流程

WM3D 5B 使用 `configs/model/native_5b.yaml`，参数量约 51 亿。默认训练规模为
8 个节点、每节点 8 张 H200，共 64 张 GPU。模型在每个节点内做 8-way FSDP2
分片，8 个节点组成 data-parallel replicas。每张卡的 micro batch 为 1，梯度累积
2 次，global batch 为 128。

整个流程按数据下载、数据整理、episode cache、1K 集群验证、10K 验证性训练和正式训练
依次进行。命令会自动记录和校验中间产物，使用者只需要指定共享存储、数据许可、模型目录
和集群地址。中间产物由程序自动管理，不需要人工计算或填写额外标识。

## 1. 训练预设

| preset | optimizer steps | 用途 |
|---|---:|---|
| `canary1k` | 1,000 | 验证 64 卡通信、训练、保存、恢复和评测 |
| `validation10k` | 10,000 | 验证数据质量、吞吐和训练稳定性 |
| `validation100k` | 100,000 | 中程训练 |
| `formal600k` | 600,000 | 正式训练 |

四套预设使用同一个 5B 模型和同一份 episode cache。它们各自生成独立的 runtime 和
checkpoint，不能把不同 preset 的 checkpoint 混在一起恢复。

## 2. 服务器与共享存储

默认集群配置：

- 8 个 8×H200 SXM 节点，节点内 NVLink；
- 400 Gb/s InfiniBand；
- Linux x86_64、Python 3.10、CUDA 12.8；
- 代码、原始数据、cache 和训练输出在所有节点上使用相同绝对路径；
- cache 放在并行文件系统或本地 NVMe 汇聚存储，不要放在低吞吐 NAS；
- 原始数据和 cache 分开计量磁盘空间，下载前先用 `df -h` 确认容量。

克隆项目：

```bash
git clone --branch v8 --single-branch https://github.com/wxqnl/world_model.git
cd world_model/wm3d
```

创建 10K 站点配置和 Python 环境：

```bash
SITE=/shared/wm3d/control/5b_validation10k.env
./run_wm3d.sh 5b init validation10k "$SITE"
vim "$SITE"
./run_wm3d.sh 5b env "$SITE"
./run_wm3d.sh 5b doctor "$SITE"
./run_wm3d.sh 5b plan "$SITE"
```

站点文件至少需要修改：

```bash
WORK_ROOT=/shared/wm3d
HF_TOKEN_FILE=/shared/secrets/huggingface_token
ACCEPT_DATA_LICENSES=YES
MASTER_ADDR=TRAIN_NODE_0
WM3D_VGGT_SOURCE_ROOT=/shared/models/vggt
WM3D_VGGT_MODEL_SNAPSHOT=/shared/models/facebook-VGGT-1B
QWEN3_VL_EMBEDDING_PATH=/shared/models/Qwen3-VL-Embedding-2B
```

默认值已经设置为 `NNODES=8`、`GPUS_PER_NODE=8`，不需要再改 world size。

## 3. 数据准备

### 3.1 默认数据组合

默认数据模板为 `configs/data/public_robot_6106h.template.yaml`。数据整理完成后，实际训练
使用 site 文件中 `DATA_PROFILE` 指向的物化配置。公开仓库和默认下载目录如下：

| 数据源 | 公开仓库 | 规划时长 | `$RAW_ROOT/` 下的目录 |
|---|---|---:|---|
| DROID | [`lerobot/droid_1.0.1`](https://huggingface.co/datasets/lerobot/droid_1.0.1) | 约 350 h | `droid` |
| Bridge V2 | [`ember-lab-berkeley/bridge_v2`](https://huggingface.co/datasets/ember-lab-berkeley/bridge_v2) | 约 100 h | `bridge` |
| RoboCasa365 Atomic | [`ember-lab-berkeley/robocasa365-pretrain-atomic`](https://huggingface.co/datasets/ember-lab-berkeley/robocasa365-pretrain-atomic) | 约 21 h | `atomic` |
| RoboCasa365 Composite | [`ember-lab-berkeley/robocasa365-pretrain-composite`](https://huggingface.co/datasets/ember-lab-berkeley/robocasa365-pretrain-composite) | 约 383 h | `composite` |
| RoboCasa365 MG | [`ember-lab-berkeley/robocasa365-pretrain-mg`](https://huggingface.co/datasets/ember-lab-berkeley/robocasa365-pretrain-mg) | 约 1,615 h | `mg` |
| AgiBotWorld2026 真机数据 | [`agibot-world/AgiBotWorld2026`](https://huggingface.co/datasets/agibot-world/AgiBotWorld2026) | 约 661 h | `agibot_world_2026` |
| AgiBotWorld Beta | [`agibot-world/AgiBotWorld-Beta`](https://huggingface.co/datasets/agibot-world/AgiBotWorld-Beta) | 约 2,976.4 h | `agibot_beta` |

默认组合的规划总量约为 6,106.4 小时。AgiBotWorld2026 只使用 Imitation Learning、
Rich Interaction 和 Reinforcement Learning 三类真机数据，不使用其 Simulation 部分。
AgiBotWorld Alpha 只提供 Beta 的官方格式转换器，不进入训练统计。表中的时长用于容量规划；
最终训练量以整理后的 episode、frame 和 window 数量为准。

### 3.2 用 OXE 替换 AgiBotWorld Beta

如果磁盘或预处理算力不足，可以暂时去掉 AgiBotWorld Beta，改用
完整的 [LeRobot Open X-Embodiment collection](https://huggingface.co/collections/lerobot/open-x-embodiment-68de658d8b544a43be4c6687)。
它不是 OXE-only 训练：DROID、Bridge、RoboCasa 和 AgiBotWorld 2026 仍然保留，只移除
AgiBotWorld Beta。DROID 本身已在主数据中，因此不会重复下载或重复计权；collection 中其余
全部数据集组成一个等权 OXE 池，整体接替 Beta 原来的 30% 采样份额，其它主数据的相对权重
保持不变。撰写本文时官方 collection 含 56 套数据，其中 DROID 已存在，因此会新增 55 个
source；实际数量以运行生成命令时的官方清单为准。

当前官方清单下，替代组合的规模和 cache 预算如下：

| 组成 | 数据变化 | 数据规模 | 预计 episode cache |
|---|---|---:|---:|
| 保留的主数据 | DROID、Bridge、RoboCasa、AgiBotWorld2026，均保持不变 | 约 3,130 h | 约 105.5 TB |
| OXE 池 | 新增除 DROID 外的 55 个 OXE source | 约 97.3 h 原始记录；约 264.6 万个 cache 帧 | 约 2.5 TB |
| **替代组合合计** | **63 个训练 source，不含 AgiBotWorld Beta** | **约 3,227 h 原始记录** | **约 108 TB（约 98 TiB）** |

cache 估算按正式的三视角 `native_p144` 表征、每个视角 144 个 token、最高约 10 Hz
保留观测计算，平均每个被保留的观测约占 0.936 MB。OXE 中许多数据是 20 Hz 或 50 Hz，
因此会按真实时间戳降到最高约 10 Hz；动作和状态仍保留原始时间戳，不做固定频率插值。

`108 TB` 只表示最终 episode cache，不包含原始下载、Hugging Face 下载缓存、临时文件和
训练 checkpoint。实际部署建议为替代组合单独准备 **130–150 TB** 的 cache 空间。若只有
200 TB 可用空间，单独存放 cache 足够；如果原始数据和训练输出也在同一文件系统，应分别
统计并预留空间。作为对比，保留 AgiBotWorld Beta 的 6,106.4 小时默认组合，按相同格式估算
约需 206 TB cache。

选择替代方案时，在 site 文件中改这五行，然后生成替代模板：

```bash
DATA_FAMILY=public_robot_oxe_no_beta
SOURCE_TEMPLATE=${CONTROL_ROOT}/public_sources_oxe_no_beta.template.yaml
SOURCE_LOCK=${CONTROL_ROOT}/public_sources_oxe_no_beta.lock.yaml
DATA_TEMPLATE=${CONTROL_ROOT}/public_robot_oxe_no_beta.template.yaml
DATA_PROFILE=${CONTROL_ROOT}/public_robot_oxe_no_beta.yaml
```

```bash
./run_wm3d.sh 5b oxe-replacement "$SITE"
```

该命令读取官方 collection 当前清单和每个数据集的 `meta/info.json`，生成下载模板、data
template 和独立 adapter 候选。随后照常执行 `lock`、`download`、schema audit、adapter audit
和 inventory。这样上游 collection 新增或删除数据集时不会靠手工列表悄悄遗漏；如果某个数据
的动作或状态维度超过 WM3D 容量，生成步骤会直接停止并报告数据集名称。

### 3.3 下载

先准备 Hugging Face token：

```bash
install -d -m 700 /shared/secrets
umask 077
read -rsp "Hugging Face token: " HF_TOKEN
printf '%s\n' "$HF_TOKEN" > /shared/secrets/huggingface_token
unset HF_TOKEN
chmod 600 /shared/secrets/huggingface_token
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
把候选文件直接当成已审计 adapter。默认完整配置应包含 9 个训练 source；替代配置的 source
数量由官方 OXE collection 当前清单决定。每个 source 都必须有非空 episode 数量。

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
export SITE=/shared/wm3d/control/5b_validation10k.env
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
或 8；不要先减少 worker 数量。共享存储写入吃不满时，先检查挂载和 striping，再考虑增加
写线程，避免在慢盘上盲目堆线程。

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

### 4.1 磁盘不足时直接从原始数据训练

如果无法为 OXE 替代组合准备约 108 TB 的完整视觉 cache，可以把 site 文件切换为：

```bash
WM3D_DATA_MODE=streaming_raw
STREAMING_METADATA_ROOT=/shared/wm3d/streaming_metadata/native_p144
STREAMING_LRU_ROOT=/local_nvme/wm3d_streaming_lru
STREAMING_LRU_GIB_PER_RANK=64
STREAMING_METADATA_WORKERS=32
STREAMING_ENCODE_BATCH_FRAMES=16
STREAMING_DECODE_WORKERS=4
```

`STREAMING_METADATA_ROOT` 放共享存储，保存时间戳、动作、状态、任务向量和窗口索引；它不保存
VGGT 视觉 token、depth、point 或 RGB pack。`STREAMING_LRU_ROOT` 必须放每个节点的本地 NVMe，
每个 rank 最多使用 64 GiB，因此默认上限为每节点 512 GiB、8 节点约 4 TiB。实际占用不会超过
正在使用的 episode 数量；空间更紧时可降到 32 GiB/rank，但 episode 切换会更频繁。

下载、adapter audit、inventory、data profile 和 task bank 与完整 cache 模式相同。生成任务后执行：

```bash
./run_wm3d.sh 5b task-bank "$SITE"
./run_wm3d.sh 5b cache-plan "$SITE"
./run_wm3d.sh 5b streaming-prepare "$SITE"
./run_wm3d.sh 5b runtime "$SITE"
```

此模式不再运行 `cache-worker`、`cache-seal`、`window` 和 `normalization`；
`streaming-prepare` 一次性生成这些步骤所需的轻量 metadata、窗口和归一化统计。训练首次访问
一个 episode 时，从原始视频解码并用冻结 VGGT 生成与普通 cache 完全相同的量化张量；同一
episode 的后续窗口直接命中本地 LRU。采样器按 episode 连续取窗口，避免每个 batch 都重复
解码。训练、checkpoint、恢复和评测命令不变。

训练日志中会增加 `streaming_raw` 字段：

- `generated_episodes`：本进程从原始数据生成的 episode 数；
- `cache_hits`：命中本地 LRU 的次数；
- `evicted_episodes`：因容量上限被删除的 episode 数；
- `prepare_seconds`、`encode_seconds`：累计解码准备和 VGGT 编码时间；
- `resident_bytes`、`resident_episodes`：当前 LRU 占用。

性能优先级依次是：本地 NVMe、足够大的 episode LRU、每 GPU 4 个解码线程和
`batch_frames=16`。如果 `prepare_seconds` 高于 `encode_seconds`，增加 CPU 与视频读取带宽；
如果 VGGT OOM，先把 `STREAMING_ENCODE_BATCH_FRAMES` 调到 12 或 8。raw 模式会比预计算
cache 慢，但不会因缺少 100 TB 级视觉 cache 而无法开训。

## 5. 1K 集群验证

1K canary 使用与正式训练相同的 64 卡拓扑，验证通信、5B 参数分片、前向、反向、梯度、
checkpoint、独立进程恢复和离线评测。

```bash
SITE=/shared/wm3d/control/5b_canary1k.env
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

`verify` 通过后再开始 10K。不要跳过独立进程 resume，因为它能发现只在恢复时出现的模型、
optimizer、sampler 或分布式状态问题。

## 6. 10K 验证性训练

10K 使用约 128 万个全局采样位置，主要验证数据分布、吞吐、loss、梯度和长期资源稳定性。

```bash
SITE=/shared/wm3d/control/5b_validation10k.env
./run_wm3d.sh 5b runtime "$SITE"

run_5b preflight
run_5b train 100

run_5b preflight
run_5b resume 100 500

run_5b preflight
run_5b resume 500 10000

run_5b preflight
run_5b eval 10000
./run_wm3d.sh 5b verify "$SITE" 10000
```

训练期间查看：

```bash
tail -f /shared/wm3d/runs/5b_validation10k/train_metrics.jsonl
watch -n 2 nvidia-smi
./run_wm3d.sh 5b status "$SITE"
```

需要重点确认：

- 64 个 rank 全部在线，8 张 GPU/节点都参与计算；
- step 连续增加，`total`、`token_mse`、`rgb`、`depth`、`point`、`pose` 均为有限数；
- `action_fine` 或数据声明的 action lane 有有效监督；
- required gradient owners 非零，nonfinite 数量为 0；
- 吞吐稳定，没有周期性长时间 GPU 空闲；
- step 100、500 和 10,000 的 checkpoint 都完整；
- offline eval 的 supervision coverage 非零。

## 7. 正式训练

10K 通过后再创建 100K 或 600K site。它们复用 data profile、episode cache、window 和
normalization，但从自己的 step 0 开始训练。

```bash
./run_wm3d.sh 5b init validation100k /shared/wm3d/control/5b_validation100k.env
./run_wm3d.sh 5b init formal600k /shared/wm3d/control/5b_formal600k.env
```

100K：

```bash
SITE=/shared/wm3d/control/5b_validation100k.env
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
SITE=/shared/wm3d/control/5b_formal600k.env
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

## 8. 常见问题

| 现象 | 处理 |
|---|---|
| `data_profile=WAITING` | 数据尚未完成 schema、inventory 和 profile 物化 |
| 下载 401/403 | 先在上游页面接受许可，再检查 token 文件 |
| cache 中 GPU 经常为 0% | 检查是否启动了 64 个 worker、CPU 绑定、原始视频盘吞吐和 batch size |
| cache 写盘积压 | 检查并行文件系统带宽、striping 和小文件性能 |
| cache OOM | 先将 `CACHE_BATCH_FRAMES` 从 16 降到 12 或 8 |
| 训练 OOM | 检查 8-way FSDP2、BF16、activation checkpoint 和 64 卡 topology |
| GPU busy、ECC、NVLink、IB 失败 | 更换节点或修复资源后重新运行 preflight |
| 恢复失败 | 只从完整编号 checkpoint 恢复，并重新运行 preflight |
| eval coverage 为 0 | 检查验证集和对应 action/视觉 supervision 是否真实存在 |

模型与分布式原理见 [WM3D 统一训练与扩展](WM3D_SCALING.md)，数据底层命令见
[WM3D 从零数据流程](WM3D_FROM_ZERO.md)，Stage1 见
[WM3D Stage1](WM3D_STAGE1_UNIFIED.md)。
