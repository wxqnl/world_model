# WM3D 5B 训练流程

WM3D 5B 使用 `configs/model/native_5b.yaml`，精确参数量为
`5,108,342,963`。默认集群由 16 个 8×H200 节点组成，world size 为 128；每个模型副本
在节点内 8 卡 FSDP2 分片，16 个节点组成 data-parallel replicas。训练使用 BF16 参数、
FP32 reduce、activation checkpoint、原生连续时间戳、native 3D world state 和 grouped robot
action。

本流程从公开数据下载开始，依次生成 source lock、data profile、task bank、episode cache、
5B window、normalization、sealed runtime、分布式 checkpoint 和 offline-eval receipt。Stage1
使用训练完成的 Stage0 checkpoint，见 [WM3D Stage1](WM3D_STAGE1_UNIFIED.md)。

## 一、训练预设

四套 H200 预设使用相同的模型、数据和缓存格式，只改变训练日程与独立 run identity：

| preset | runtime profile | optimizer steps | 主要 checkpoint | 用途 |
|---|---|---:|---|---|
| `canary1k` | `h200_128_fsdp2_canary1k.yaml` | 1,000 | 100、500、1,000 | 集群和训练链验收 |
| `validation10k` | `h200_128_fsdp2_validation10k.yaml` | 10,000 | 100、500、每 1,000 step | 验证性训练 |
| `validation100k` | `h200_128_fsdp2_validation100k.yaml` | 100,000 | 每 1,000 step | 中程稳定性验证 |
| `formal600k` | `h200_128_fsdp2.yaml` | 600,000 | 1,000、5,000、20,000、此后每 20,000 step | 正式训练 |

四套预设均固定为 128×H200、8-way FSDP2、global batch 128，并要求节点内完整
NVLink、400 Gb/s InfiniBand、ECC 为 0。每个 preset 物化独立 runtime，不能跨 preset
恢复 checkpoint。episode cache 可复用；5B 的 window index 和 normalization 也可在四套
preset 间复用。

`5b init` 根据 preset 自动选择 runtime profile、最终 step、run name、run root 和 eval
输出路径：

```bash
./run_wm3d.sh 5b init validation10k /shared/wm3d/control/5b_validation10k.env
```

同一 preset 再开一次独立实验时，复制 site 文件并修改 `WM3D_5B_RUN_ID`。已有 runtime、
checkpoint 和 receipt 不需要改名或覆盖。

## 二、服务器和环境

### 2.1 集群条件

- Linux x86_64、Python 3.10；
- 16 个 8×H200 SXM 节点，节点内 NVLink；
- 400 Gb/s InfiniBand；
- 所有节点以相同绝对路径挂载代码、原始数据、cache 和 run 目录；
- `git`、`curl`、`ffmpeg`、Slurm、CUDA 12.8 驱动环境；
- Hugging Face 账号已经接受 AgiBotWorld 数据许可。

完整公开数据建议准备约 200 TB，其中训练 cache 和 checkpoint 所在文件系统使用高速存储。
5B DCP 实测约 61 GB；10K 预设约需 0.8 TB checkpoint 空间。正式 preflight 还要求数据和
输出文件系统分别至少保留 10 TB。

### 2.2 克隆和安装

```bash
git clone --branch v8 --single-branch https://github.com/wxqnl/world_model.git
cd world_model/wm3d
git status --short
```

创建只读 Hugging Face token 文件：

```bash
install -d -m 700 /shared/secrets
umask 077
read -rsp "Hugging Face token: " HF_TOKEN
printf '%s\n' "$HF_TOKEN" > /shared/secrets/huggingface_token
unset HF_TOKEN
chmod 600 /shared/secrets/huggingface_token
```

创建 10K site 配置并建立 Python 环境：

```bash
SITE=/shared/wm3d/control/5b_validation10k.env
./run_wm3d.sh 5b init validation10k "$SITE"
vim "$SITE"
./run_wm3d.sh 5b env "$SITE"
./run_wm3d.sh 5b doctor "$SITE"
./run_wm3d.sh 5b plan "$SITE"
```

site 文件保存共享存储、token、模型资产和集群 rendezvous 路径。通常需要修改：

```bash
WORK_ROOT=/shared/wm3d
HF_TOKEN_FILE=/shared/secrets/huggingface_token
ACCEPT_DATA_LICENSES=YES
MASTER_ADDR=TRAIN_NODE_0
WM3D_VGGT_SOURCE_ROOT=/shared/models/vggt/a288dd0f14786c93483e45524328726ab7b1b4ce
WM3D_VGGT_MODEL_SNAPSHOT=/shared/models/huggingface/facebook-VGGT-1B/860abec7937da0a4c03c41d3c269c366e82abdf9
QWEN3_VL_EMBEDDING_PATH=/shared/models/huggingface/Qwen3-VL-Embedding-2B/9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda
```

VGGT 源码固定 revision 为 `a288dd0f14786c93483e45524328726ab7b1b4ce`；VGGT-1B
模型固定 revision 为 `860abec7937da0a4c03c41d3c269c366e82abdf9`；任务编码器
固定为 `Qwen/Qwen3-VL-Embedding-2B` revision
`9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda`。已有镜像可直接填绝对路径；没有镜像时可用
[真实 smoke 流程](WM3D_REAL_SMOKE.md)下载并生成资产 receipt。

## 三、数据准备

### 3.1 数据清单

扩展数据配置为 `configs/data/public_robot_6106h.template.yaml`：

| 数据源 | Hugging Face 仓库 | 规划时长 | 下载目录 |
|---|---|---:|---|
| DROID | [`lerobot/droid_1.0.1`](https://huggingface.co/datasets/lerobot/droid_1.0.1) | 350 h | `raw/droid` |
| Bridge V2 | [`ember-lab-berkeley/bridge_v2`](https://huggingface.co/datasets/ember-lab-berkeley/bridge_v2) | 100 h | `raw/bridge` |
| RoboCasa Atomic | [`ember-lab-berkeley/robocasa365-pretrain-atomic`](https://huggingface.co/datasets/ember-lab-berkeley/robocasa365-pretrain-atomic) | 21 h | `raw/atomic` |
| RoboCasa Composite | [`ember-lab-berkeley/robocasa365-pretrain-composite`](https://huggingface.co/datasets/ember-lab-berkeley/robocasa365-pretrain-composite) | 383 h | `raw/composite` |
| RoboCasa MG | [`ember-lab-berkeley/robocasa365-pretrain-mg`](https://huggingface.co/datasets/ember-lab-berkeley/robocasa365-pretrain-mg) | 1,615 h | `raw/mg` |
| AgiBotWorld2026 | [`agibot-world/AgiBotWorld2026`](https://huggingface.co/datasets/agibot-world/AgiBotWorld2026) | 661 h | `raw/agibot_world_2026` |
| AgiBotWorld Beta | [`agibot-world/AgiBotWorld-Beta`](https://huggingface.co/datasets/agibot-world/AgiBotWorld-Beta) | 2,976.4 h | `raw/agibot_beta` |

规划总量约 6,106.4 小时。实际训练统计来自 sealed data profile 中的 episode、frame 和
window 数量，而不是表格中的预算值。

### 3.2 锁定 revision 和下载

```bash
./run_wm3d.sh 5b lock "$SITE"
./run_wm3d.sh 5b download "$SITE"
```

`lock` 将每个上游 revision 固定为 40 位 commit，并保存完整 file list。`download` 支持断点
重入；同一命令会验证已有文件并继续未完成部分。下载状态可这样检查：

```bash
source "$SITE"
test -s "$SOURCE_LOCK"
find "$RAW_ROOT/receipts" -name '*.json' -type f | sort
df -h "$RAW_ROOT"
```

### 3.3 数据转换、schema 和 adapter

AgiBotWorld2026 archive materialization、AgiBot Beta 官方转换器、schema audit、adapter audit、
inventory 和 data-profile 的完整命令见 [从零数据流程](WM3D_FROM_ZERO.md)。处理顺序为：

```text
download -> archive/conversion -> schema-audit -> adapter-audit
         -> inventory -> data-profile
```

最终文件写入 site 配置的 `DATA_PROFILE`：

```text
/shared/wm3d/control/public_robot_6106h.yaml
```

该文件绑定每个 source manifest、adapter contract 和 inventory receipt。数据字段完成审计后，
下面的命令会打印 profile 名称、source 数量和 SHA：

```bash
./run_wm3d.sh 5b doctor "$SITE"
```

处理后的共享目录结构如下：

```text
/shared/wm3d/
├── raw/                         # 固定 revision 的下载快照与下载 receipt
├── control/                     # source lock、data profile 和 sealed runtime
├── cache/native_p144/           # task bank、episode cache、5B window、normalization
├── runs/5b_<run-id>/            # metrics、launch qualification、DCP 和 eval
├── envs/wm3d-cu128/             # 固定依赖的 Python 环境
└── huggingface/                 # Hugging Face 下载缓存
```

### 3.4 Task bank 和 episode cache

```bash
./run_wm3d.sh 5b task-bank "$SITE"
./run_wm3d.sh 5b cache-plan "$SITE"
./run_wm3d.sh 5b status "$SITE"
```

cache worker 在 Slurm 上按 GPU 展开：

```bash
export SITE=/shared/wm3d/control/5b_validation10k.env
srun --nodes=16 --ntasks=128 --ntasks-per-node=8 --gpus-per-task=1 \
  --export=ALL,SITE bash -lc '
  ./run_wm3d.sh 5b cache-worker "$SITE" \
    "$SLURM_PROCID" 128 inherited
'
```

每个 worker 负责固定 task partition。中断后以相同 worker index 和 worker count 重启，已完成
并通过 SHA 验证的 episode 会跳过。全部 worker 正常退出后生成 episode seal、5B window、
grouped normalization 和 runtime：

```bash
./run_wm3d.sh 5b cache-seal "$SITE"
./run_wm3d.sh 5b window "$SITE"
./run_wm3d.sh 5b normalization "$SITE"
./run_wm3d.sh 5b runtime "$SITE"
./run_wm3d.sh 5b status "$SITE"
```

cache 正常时，task、episode、train window 和 val window 数量都大于 0；episode/window seal
通过；runtime 报告 `native_5b`、`5,108,342,963 params` 和 `world 128`。

## 四、1K 集群验收

1K canary 与正式训练使用同一 128×H200 拓扑。它验证 5B 参数分片、forward/backward、
gradient ownership、DCP save、独立进程 exact resume 和 offline eval。

创建独立 site 并物化 runtime：

```bash
SITE=/shared/wm3d/control/5b_canary1k.env
./run_wm3d.sh 5b init canary1k "$SITE"
vim "$SITE"
./run_wm3d.sh 5b doctor "$SITE"
./run_wm3d.sh 5b runtime "$SITE"
```

如果数据 cache 已由 10K site 生成，只需让两个 site 指向同一 `CACHE_ROOT`、`DATA_PROFILE`
和模型资产。

## 五、多节点启动

每个 Slurm task 对应一个节点 launcher；launcher 再启动本机 8 个训练进程。site 文件会从
`SLURM_PROCID` 读取 node rank。

```bash
export SITE=/shared/wm3d/control/5b_canary1k.env
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)

srun --nodes=16 --ntasks=16 --ntasks-per-node=1 \
  --export=ALL,SITE,MASTER_ADDR bash -lc '
  ./run_wm3d.sh 5b preflight "$SITE"
'

srun --nodes=16 --ntasks=16 --ntasks-per-node=1 \
  --export=ALL,SITE,MASTER_ADDR bash -lc '
  ./run_wm3d.sh 5b train "$SITE" 100
'
```

`preflight` 验证 128 个 rank、GPU 型号和 UUID、显存、ECC、空闲状态、NVLink、IB、
memlock、共享内存、磁盘空间和真实 all-reduce。resource receipt 有效期为 30 分钟；每次
fresh、resume 或 eval 启动前重新执行 preflight。

step 100 checkpoint 完成后，以新进程恢复到 step 500：

```bash
srun --nodes=16 --ntasks=16 --ntasks-per-node=1 \
  --export=ALL,SITE,MASTER_ADDR bash -lc '
  ./run_wm3d.sh 5b preflight "$SITE"
'

srun --nodes=16 --ntasks=16 --ntasks-per-node=1 \
  --export=ALL,SITE,MASTER_ADDR bash -lc '
  ./run_wm3d.sh 5b resume "$SITE" 100 500
'
```

继续恢复并评测 canary：

```bash
srun --nodes=16 --ntasks=16 --ntasks-per-node=1 \
  --export=ALL,SITE,MASTER_ADDR bash -lc '
  ./run_wm3d.sh 5b preflight "$SITE"
'
srun --nodes=16 --ntasks=16 --ntasks-per-node=1 \
  --export=ALL,SITE,MASTER_ADDR bash -lc '
  ./run_wm3d.sh 5b resume "$SITE" 500 1000
'
srun --nodes=16 --ntasks=16 --ntasks-per-node=1 \
  --export=ALL,SITE,MASTER_ADDR bash -lc '
  ./run_wm3d.sh 5b preflight "$SITE"
'
srun --nodes=16 --ntasks=16 --ntasks-per-node=1 \
  --export=ALL,SITE,MASTER_ADDR bash -lc '
  ./run_wm3d.sh 5b eval "$SITE" 1000
'
./run_wm3d.sh 5b verify "$SITE" 1000
```

数字 step 会自动解析为当前 run 下的 `checkpoints/step_XXXXXXXX`；也可以传完整 checkpoint
绝对路径。

## 六、10K 验证性训练

10K 是 10,000 个 optimizer steps，global batch 为 128，共消费约 128 万个确定性采样位置。
它使用独立的 `validation10k` runtime，从 step 0 开始：

```bash
SITE=/shared/wm3d/control/5b_validation10k.env
./run_wm3d.sh 5b init validation10k "$SITE"
vim "$SITE"
./run_wm3d.sh 5b doctor "$SITE"
./run_wm3d.sh 5b runtime "$SITE"
```

启动命令与 1K 相同，训练日程为：

```bash
./run_wm3d.sh 5b preflight "$SITE"
./run_wm3d.sh 5b train "$SITE" 100

./run_wm3d.sh 5b preflight "$SITE"
./run_wm3d.sh 5b resume "$SITE" 100 500

./run_wm3d.sh 5b preflight "$SITE"
./run_wm3d.sh 5b resume "$SITE" 500 10000

./run_wm3d.sh 5b preflight "$SITE"
./run_wm3d.sh 5b eval "$SITE" 10000
./run_wm3d.sh 5b verify "$SITE" 10000
```

多节点环境中，每一行 `preflight`、`train`、`resume` 和 `eval` 均放入第五节的 16-task
`srun`；`verify` 在一个节点执行。

## 七、100K 和 600K 训练

10K 结果通过后，分别创建 100K 与正式 600K site：

```bash
./run_wm3d.sh 5b init validation100k /shared/wm3d/control/5b_validation100k.env
./run_wm3d.sh 5b init formal600k /shared/wm3d/control/5b_formal600k.env
SITE100K=/shared/wm3d/control/5b_validation100k.env
SITE600K=/shared/wm3d/control/5b_formal600k.env
```

两套 site 使用同一个 sealed data profile、episode cache、5B window 和 normalization，但各自
运行 `runtime` 并从 step 0 训练。100K 默认每 1,000 step 保存和验证；600K 在 1,000、
5,000、20,000 step 保存早期 checkpoint，此后每 20,000 step 保存，validation 每 5,000
step 运行 100 个 batch。

100K 的首段和最终验收：

```bash
./run_wm3d.sh 5b runtime "$SITE100K"
./run_wm3d.sh 5b preflight "$SITE100K"
./run_wm3d.sh 5b train "$SITE100K" 1000
./run_wm3d.sh 5b preflight "$SITE100K"
./run_wm3d.sh 5b resume "$SITE100K" 1000 100000
./run_wm3d.sh 5b preflight "$SITE100K"
./run_wm3d.sh 5b eval "$SITE100K" 100000
./run_wm3d.sh 5b verify "$SITE100K" 100000
```

600K 的首段、恢复和最终验收：

```bash
./run_wm3d.sh 5b runtime "$SITE600K"
./run_wm3d.sh 5b preflight "$SITE600K"
./run_wm3d.sh 5b train "$SITE600K" 1000
./run_wm3d.sh 5b preflight "$SITE600K"
./run_wm3d.sh 5b resume "$SITE600K" 1000 5000
./run_wm3d.sh 5b preflight "$SITE600K"
./run_wm3d.sh 5b resume "$SITE600K" 5000 20000
./run_wm3d.sh 5b preflight "$SITE600K"
./run_wm3d.sh 5b resume "$SITE600K" 20000 600000
./run_wm3d.sh 5b preflight "$SITE600K"
./run_wm3d.sh 5b eval "$SITE600K" 600000
./run_wm3d.sh 5b verify "$SITE600K" 600000
```

## 八、状态和结果

训练期间查看汇总：

```bash
./run_wm3d.sh 5b status "$SITE"
tail -f /shared/wm3d/runs/5b_validation10k/train_metrics.jsonl
watch -n 2 nvidia-smi
```

`status` 和 `verify` 汇总以下内容：

| 范围 | 输出 |
|---|---|
| data/cache | source、episode、window 数量及 seal 状态 |
| model/runtime | `native_5b`、精确参数量、world size、runtime SHA |
| training | 当前 step、loss、learning rate、grad norm、samples/s |
| gradients | required owner 非零、nonfinite 为 0 |
| checkpoint | step、COMMITTED/MANIFEST、payload SHA 和大小 |
| evaluation | 全部指标有限、每条声明 supervision lane coverage 大于 0 |

主要训练指标：

| 指标 | 含义 |
|---|---|
| `token_mse` | native future token 预测误差 |
| `rgb` | 未来 RGB 重建误差 |
| `depth`、`point`、`pose` | 未来 3D 几何与相机误差 |
| `action_fine`、`action_coarse` | data profile 声明的 grouped-action lane |
| `grad_norm` | 全模型梯度范数 |
| `samples/s` | 全局训练吞吐 |

正常运行时，step 单调增加，loss 和 grad norm 为有限数，required gradient owners 全部非零，
eval expected coverage 全部大于 0。稳定训练阶段每个节点的 8 张 GPU 都有训练进程；长时间低
利用率通常对应数据读取、共享存储或 DataLoader 吞吐不足，可结合 `samples/s`、GPU util 和
I/O 延迟定位。

`verify` 的 PASS 表示数据闭包、训练数值、梯度、checkpoint、resume 和 offline eval 工作
正常。模型能力使用固定下游任务和机器人闭环 benchmark 单独评估。

## 九、恢复和常见错误

完整 checkpoint 目录包含 `COMMITTED.json`。恢复时传编号 step 或完整目录：

```bash
./run_wm3d.sh 5b preflight "$SITE"
./run_wm3d.sh 5b resume "$SITE" 500 10000
```

常见错误及处理方式：

| 错误 | 处理 |
|---|---|
| `data_profile=WAITING` | 完成 schema、adapter、inventory 和 data-profile |
| gated dataset 401/403 | 接受上游许可并检查 token 权限 |
| source revision mismatch | 重新运行 `lock`，不使用 `main` 或 `latest` |
| cache worker `failed > 0` | 修复第一个失败 episode，以相同 partition 重跑 |
| resource receipt stale | 重新运行 preflight |
| GPU busy、ECC、NVLink、IB 失败 | 更换空闲节点或修复集群资源 |
| `COMMITTED.json` 缺失 | 等待 checkpoint 完成，或从上一个 committed step 恢复 |
| zero coverage | 检查 val split 与 adapter supervision lane |
| OOM | 检查 8-way FSDP2、BF16 和 activation checkpoint 配置 |

底层数据物化命令见 [WM3D 从零数据、训练与评测](WM3D_FROM_ZERO.md)，模型与分布式设计见
[WM3D 统一训练与扩展](WM3D_SCALING.md)，发布验收边界见
[WM3D 发布验证](WM3D_RELEASE_VALIDATION.md)。
