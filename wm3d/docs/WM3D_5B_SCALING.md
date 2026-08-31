# WM3D V8 5B 训练交付手册

本文只描述当前正式方案。训练操作员按顺序执行，不修改模型结构、数据模式、loss、batch、
动作语义或 runtime profile。

## 1. 固定训练合同

- GitHub 分支：`v8`
- 模型：`configs/model/native_5b_v8_core.yaml`
- 参数量：`5,440,933,496`
- 视觉 encoder：`configs/encoder/vggt_native_p144.yaml`
- objective：`configs/objective/stage0_v8_core.yaml`
- 数据访问：`direct_raw`
- 集群：8 个节点，每节点 8 张 H200，共 64 张 GPU
- 分布式：节点内 8-way FSDP2，节点间 data parallel
- micro batch：每卡 4
- global batch：256
- canary：1,000 steps
- 正式训练：600,000 steps，必须 fresh step 0 启动

V8 的 future physical action 在 state encoder 前进入独立 factual pass，并再次进入两层独立
factual decoder。P144 factual future state 负责运动和低频 RGB；原始 V7
`ContextResidualPixelDecoder` 负责 RGB 主输出；受限高频 refiner 只补充晚期细节。
policy/action-free trunk 不读取 future candidate。当前合同不使用 absolute future P256、
P256 自回归、teacher forcing、copy-last、flow 或 RAFT。

以下文件由 site 自动指定，不要手工换成旧配置：

```text
configs/model/native_5b_v8_core.yaml
configs/encoder/vggt_native_p144.yaml
configs/encoder/task_qwen3_vl_embedding_2b.yaml
configs/objective/stage0_v8_core.yaml
configs/runtime/h200_64_fsdp2_canary1k.yaml
configs/runtime/h200_64_fsdp2.yaml
```

## 2. 启动前准备

要求登录节点能访问 GitHub、Python package index 和 Hugging Face。计算节点可以离线，但必须
共享 `/data/wm3d`、模型目录、原始数据和训练输出。集群必须具备 400 Gb/s InfiniBand，节点内
8 张 H200 必须组成正常的 NVLink clique。

数据负责人需要提前把已审计的数据控制包完整放到：

```text
/data/wm3d/control
```

其中必须包含 source lock、上游 file-list receipt、正式 data profile、adapter audit receipt、
source inventory 和 manifest。训练操作员不修改 adapter，也不自行解释 action/state 字段。
只有原始数据、没有正式 data profile 时不能启动训练。

固定模型资产路径：

```text
/data/models/vggt/a288dd0f14786c93483e45524328726ab7b1b4ce
/data/models/huggingface/facebook-VGGT-1B/860abec7937da0a4c03c41d3c269c366e82abdf9
/data/models/huggingface/Qwen3-VL-Embedding-2B/9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda
```

## 3. 拉取代码并创建环境

在共享登录节点执行：

```bash
cd /data
git clone --branch v8 --single-branch https://github.com/wxqnl/world_model.git
cd /data/world_model/wm3d

SITE=/data/wm3d/control/5b_canary1k.env
./run_wm3d.sh 5b init canary1k "$SITE" direct_raw
sed -i 's/^ACCEPT_DATA_LICENSES=NO$/ACCEPT_DATA_LICENSES=YES/' "$SITE"
sed -i 's/REQUIRED_MASTER_ADDR/127.0.0.1/' "$SITE"
```

site 已固定为以下设置，不要打开文件改成其他路径或配置：

```bash
WORK_ROOT=/data/wm3d
HF_TOKEN_FILE=/data/secrets/huggingface_token
ACCEPT_DATA_LICENSES=YES
INCLUDE_AGIBOT_2026=YES
INCLUDE_AGIBOT_BETA=NO
WM3D_VGGT_SOURCE_ROOT=/data/models/vggt/a288dd0f14786c93483e45524328726ab7b1b4ce
WM3D_VGGT_MODEL_SNAPSHOT=/data/models/huggingface/facebook-VGGT-1B/860abec7937da0a4c03c41d3c269c366e82abdf9
QWEN3_VL_EMBEDDING_PATH=/data/models/huggingface/Qwen3-VL-Embedding-2B/9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda
```

site 中的 `127.0.0.1` 只供登录节点执行本地检查。第 6、7 节启动训练时会用 Slurm 分配到的
第一台训练节点覆盖它。

安全保存 Hugging Face token：

```bash
install -d -m 700 /data/secrets
umask 077
read -rsp "Hugging Face token: " HF_TOKEN
printf '%s\n' "$HF_TOKEN" > /data/secrets/huggingface_token
unset HF_TOKEN
chmod 600 /data/secrets/huggingface_token
```

创建并封存环境：

```bash
cd /data/world_model/wm3d
./run_wm3d.sh 5b env "$SITE"
```

完成标志：

```text
/data/wm3d/envs/wm3d-cu128/environment_receipt.json
```

再次执行同一条 `env` 命令应输出 `verified-skip`。不要在训练期间升级 package。

## 4. 验证并复用现有数据

AgiBotWorld2026 必须保持官方目录：

```text
/data/wm3d/raw/agibot_world_2026/ImitationLearning
/data/wm3d/raw/agibot_world_2026/RichInteraction
/data/wm3d/raw/agibot_world_2026/ReinforcementLearning
```

先做只读检查：

```bash
cd /data/world_model/wm3d
SITE=/data/wm3d/control/5b_canary1k.env
source "$SITE"
./run_wm3d.sh agibot-existing-check \
  --snapshot-root "$RAW_ROOT/agibot_world_2026"
```

输出必须包含 `"passed": true`，三个 prefix 都必须有非零 archive 数量。该检查不解压、不改写
数据，也不会哈希整个多 TB 数据。

随后固定所有数据版本并验证已有文件：

```bash
cd /data/world_model/wm3d
SITE=/data/wm3d/control/5b_canary1k.env
./run_wm3d.sh 5b data-template "$SITE"
./run_wm3d.sh 5b lock "$SITE"
./run_wm3d.sh 5b download "$SITE"
```

`download` 会复用现有文件，只补缺失内容，并生成训练所需的 download receipt。不要绕过
`lock` 或手工伪造 receipt。

确认数据控制包和 download receipt 都存在：

```bash
SITE=/data/wm3d/control/5b_canary1k.env
source "$SITE"
test -f "$SOURCE_LOCK"
test -f "$DATA_PROFILE"
test -f "$RAW_ROOT/receipts/agibot_world_2026.json"
./run_wm3d.sh 5b doctor "$SITE"
./run_wm3d.sh 5b plan "$SITE"
```

`doctor` 必须报告：

```text
model=native_5b_v8_exact_v7_factual_high_frequency_refiner
parameters=5,440,933,496
world_size=64
data mode=direct_raw
```

出现 `data_profile=WAITING`、旧 P256/teacher 配置或参数量不一致时停止，不要启动 GPU 作业。

## 5. 生成训练 metadata 和 runtime

依次执行：

```bash
cd /data/world_model/wm3d
SITE=/data/wm3d/control/5b_canary1k.env
./run_wm3d.sh 5b task-bank "$SITE"
./run_wm3d.sh 5b cache-plan "$SITE"
./run_wm3d.sh 5b streaming-prepare "$SITE"
./run_wm3d.sh 5b runtime "$SITE"
./run_wm3d.sh 5b doctor "$SITE"
./run_wm3d.sh 5b status "$SITE"
```

这些命令只生成 task bank、episode/window metadata、normalization 和 sealed runtime。
`direct_raw` 不生成 episode visual cache，训练时按 sealed window 在线解码并运行 frozen VGGT。

## 6. 运行 64 卡 1K canary

先申请 8 个完整 H200 节点。进入 Slurm allocation 后执行：

```bash
export CODE_ROOT=/data/world_model/wm3d
export SITE=/data/wm3d/control/5b_canary1k.env
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)

run_5b () {
  operation=$1
  shift
  srun --nodes=8 --ntasks=8 --ntasks-per-node=1 \
    --kill-on-bad-exit=1 \
    --export=ALL,CODE_ROOT,SITE,MASTER_ADDR \
    bash -lc 'cd "$CODE_ROOT" && exec ./run_wm3d.sh 5b "$@"' \
    _ "$operation" "$SITE" "$@"
}
```

按以下顺序运行，不能合并成一个长进程：

```bash
run_5b preflight
run_5b train 100

run_5b preflight
run_5b resume 100 500

run_5b preflight
run_5b resume 500 1000

run_5b preflight
run_5b eval 1000

cd "$CODE_ROOT"
./run_wm3d.sh 5b verify "$SITE" 1000
```

通过条件：

- 64 个 rank 全部存在，GPU 型号、HBM、ECC、NVLink 和 IB preflight 全部通过；
- loss、grad norm 和所有梯度所有权指标有限；
- factual decoder、RGB decoder、action head 和 policy 都有非零梯度；
- future action 对 policy/action-free 输出的逐元素差异为 0；
- step100、step500、step1000 都有完整 COMMITTED checkpoint；
- 每个 checkpoint 都有 64 份分片状态，独立进程 resume 成功；
- step1000 eval 和 `verify` 通过。

任一条件失败都不要启动正式训练。

## 7. Fresh 启动正式 600K

1K canary 通过后，从 canary site 复制固定站点设置，只修改 preset：

```bash
export CODE_ROOT=/data/world_model/wm3d
export CANARY_SITE=/data/wm3d/control/5b_canary1k.env
export SITE=/data/wm3d/control/5b_formal600k.env

install -m 600 "$CANARY_SITE" "$SITE"
sed -i 's/^WM3D_5B_PRESET=canary1k$/WM3D_5B_PRESET=formal600k/' "$SITE"

cd "$CODE_ROOT"
./run_wm3d.sh 5b doctor "$SITE"
./run_wm3d.sh 5b runtime "$SITE"
```

重新进入正式训练的 8 节点 Slurm allocation，定义与第 6 节相同的 `run_5b` 函数，然后执行：

```bash
run_5b preflight
run_5b train
```

正式训练必须从自己的 step 0 开始。不要从 1K canary、旧 V8、旧 5B 或任何 1B checkpoint
初始化。

查看状态：

```bash
cd /data/world_model/wm3d
./run_wm3d.sh 5b status /data/wm3d/control/5b_formal600k.env
```

训练中断后只从本正式 run 最新的完整 COMMITTED checkpoint 恢复：

```bash
run_5b preflight
run_5b resume 完整checkpoint的step号
```

## 8. 必须停止的情况

出现以下任一情况，停止作业并保留日志和 checkpoint：

- 任一 rank 丢失或 NCCL/IB/NVLink 报错；
- loss、梯度或模型输出出现 NaN/Inf；
- GPU ECC 非零；
- 20 分钟内日志、CPU 和 GPU 都没有进展；
- 数据、runtime、normalization 或 environment receipt 不一致；
- checkpoint 没有 COMMITTED 标记或分片数量不足；
- future action 进入 policy/action-free trunk；
- RGB 输出单色塌缩，或 factual action 路径没有梯度。

不要通过改模型、改 loss、减少 RGB horizon、切换数据模式或跳过 preflight 来绕过错误。
