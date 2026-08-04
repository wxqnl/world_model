# WM3D

WM3D 是在显式时空 3D lattice 上联合学习世界状态与机器人动作的世界模型。模型直接预测
未来 RGB、depth、point、camera 与 grouped action；世界状态由原生 3D 主干维护，动作由独立
时序主干建模，再通过双向 bridge 交换信息。

仓库提供从公开数据下载、数据转换、VGGT 特征缓存、分布式预训练、精确恢复到 checkpoint
评测的完整流程。日常操作统一通过 `./wm3d.sh`；实现脚本按数据、资产、集群、Slurm 和
smoke 职责分类，目录说明见 `scripts/README.md`。

当前发布分支为 `v7`。V7 是数据契约和 checkpoint 协议的版本号，不是项目名称。模型规模由
YAML 配置决定；`configs/train/5b_h200.yaml` 是当前的大规模训练预设之一。

## 代码接手导航

仓库各职责目录都带中文 README：

- [`configs/README.md`](configs/README.md)：站点、数据、smoke 与训练预设；
- [`environments/README.md`](environments/README.md)：普通 Python venv 与环境 receipt；
- [`scripts/README.md`](scripts/README.md)：一键流水线背后的脚本调用关系；
- [`wm3d/README.md`](wm3d/README.md)：data、VGGT、model 与 training 运行时；
- [`wm3d/models/README.md`](wm3d/models/README.md)：V7 native-3D 血统和 5B 架构所有权。

安装环境后可直接核验当前正式 5B 配置是否仍与清理前 V7 anchor 一致：

```bash
./wm3d.sh audit site.env
```

该命令检查 Git anchor/blob、允许重命名后的逐字一致性、V7 配置继承段、后续架构依赖边界和
精确参数预算；不是根据 README 文本作判断。

## 一、从新服务器开始

### 1. 集群条件

- Linux x86_64、Python 3.10；
- `git`、`curl`、`ffmpeg`；
- Slurm：`sbatch`、`srun`、`scontrol`；
- 所有计算节点挂载同一共享存储；
- 正式 5B 预设使用每节点 8 张 H200 SXM，节点内 NVLink；
- 多机训练建议使用 400 Gb/s InfiniBand；
- Hugging Face 账号已接受 AgiBot Alpha/Beta 的数据许可。

推荐为完整数据准备 200 TB 可用空间，其中 80–100 TB 为高速训练热层。5B 预设最低使用
64 张 H200，推荐使用 128 张 H200。

### 2. 克隆代码

```bash
git clone --branch v7 --single-branch \
  https://github.com/wxqnl/world_model.git
cd world_model
```

如果所在网络的 GitHub HTTP/2 连接不稳定，使用完整历史的 HTTP/1.1 clone；不要用源码
tarball 替代 Git checkout，因为 V7 血统审计和 code receipt 需要 Git 历史：

```bash
git -c http.version=HTTP/1.1 clone --branch v7 --single-branch \
  https://github.com/wxqnl/world_model.git
cd world_model
```

### 3. 配置 Hugging Face 凭据

先在 Hugging Face 页面接受 AgiBot Alpha/Beta 许可，再创建只读 token 文件：

```bash
install -d -m 700 /shared/secrets
umask 077
read -rsp "Hugging Face token: " HF_TOKEN
printf '%s\n' "$HF_TOKEN" > /shared/secrets/huggingface_token
unset HF_TOKEN
chmod 600 /shared/secrets/huggingface_token
```

token 只从该文件读取，不会写入数据 lock、Slurm 参数或训练日志。

### 4. 创建站点配置与 Python 环境

```bash
./wm3d.sh init site.env
```

编辑 `site.env` 中的必填项：

```bash
WORK_ROOT=/shared/wm3d
HF_TOKEN_FILE=/shared/secrets/huggingface_token
SLURM_PARTITION=h200
SLURM_ACCOUNT=your_account
ACCEPT_DATA_LICENSES=YES
```

`WORK_ROOT` 必须位于所有节点可见的共享存储。环境使用普通 Python 3.10 venv，依赖版本由
`environments/requirements.lock` 固定：

```bash
./wm3d.sh setup site.env
./wm3d.sh doctor site.env
```

`setup` 只安装训练与数据处理共用的主环境。运行 `prepare` 时，流水线会按需创建
独立的 AgiBot Beta 转换环境；转换器依赖不会污染训练环境，也不需要手工配置。

PyPI 访问较慢时可临时指定镜像：

```bash
PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple \
  ./wm3d.sh setup site.env
```

集群无法直连 huggingface.co 时，正式数据流水线在 `site.env` 中设置
`HF_ENDPOINT=https://hf-mirror.com`。下载完成后仍会用 source lock 中的 40 位 commit SHA
和本地 receipt 校验数据身份。小样本 smoke 不读取 `site.env`，对应的镜像命令见第三节。

## 二、公开数据

默认数据配置位于 `configs/data/public_6106h.yaml`，原始仓库与下载目录如下：

| 数据源 | 公开仓库 | 规划时长 | `WORK_ROOT/raw/snapshots/` 下的目录 |
|---|---|---:|---|
| DROID | [`lerobot/droid_1.0.1`](https://huggingface.co/datasets/lerobot/droid_1.0.1) | 约 350 h | `droid` |
| Bridge V2 | [`ember-lab-berkeley/bridge_v2`](https://huggingface.co/datasets/ember-lab-berkeley/bridge_v2) | 约 100 h | `bridge` |
| RoboCasa365 Atomic | [`ember-lab-berkeley/robocasa365-pretrain-atomic`](https://huggingface.co/datasets/ember-lab-berkeley/robocasa365-pretrain-atomic) | 约 21 h | `atomic` |
| RoboCasa365 Composite | [`ember-lab-berkeley/robocasa365-pretrain-composite`](https://huggingface.co/datasets/ember-lab-berkeley/robocasa365-pretrain-composite) | 约 383 h | `composite` |
| RoboCasa365 MG | [`ember-lab-berkeley/robocasa365-pretrain-mg`](https://huggingface.co/datasets/ember-lab-berkeley/robocasa365-pretrain-mg) | 约 1,615 h | `mg` |
| AgiBotWorld2026 真机 | [`agibot-world/AgiBotWorld2026`](https://huggingface.co/datasets/agibot-world/AgiBotWorld2026) | 约 661 h | `agibot_world_2026_snapshot` |
| AgiBotWorld Beta | [`agibot-world/AgiBotWorld-Beta`](https://huggingface.co/datasets/agibot-world/AgiBotWorld-Beta) | 2,976.4 h | `agibot_beta_snapshot` |

规划总量约 **6,106.4 小时**。AgiBotWorld2026 采用 Imitation Learning、Rich
Interaction 和 Reinforcement Learning 的真机部分；Simulation 不计入该预设。AgiBot Alpha
快照用于取得 Beta 官方转换器，不参与训练小时统计。

采样周期中，DROID、Bridge、Atomic、Composite、MG 合计占 40%，五路内部比例为
35/15/10/20/20；AgiBot 数据占 60%。所有比例都写在数据 YAML 中，不写死在训练代码里。

### 一条命令完成下载与处理

```bash
./wm3d.sh data site.env
```

该命令依次执行：

1. 解析每个公开仓库的 40 位 commit SHA，生成不可变 source lock；
2. 断点下载固定 revision 的原始快照；
3. 安全解包 AgiBotWorld2026；到该阶段才自动安装固定版本的官方工具并转换 AgiBot Beta；
4. 审计 RGB、action、时间戳、episode 和 embodiment schema；
5. 生成统一 episode plan、grouped action 与可变维度 mask；
6. 统计 action 分布，生成 task bank 和 VGGT 3D cache；
7. 合并 shard，检查缺失、重复、覆盖率和 SHA，发布 dataset seal。

需要分阶段运行时：

```bash
./wm3d.sh lock site.env       # 固定上游 revision
./wm3d.sh download site.env   # 下载或断点续传
./wm3d.sh prepare site.env    # 转换并生成 episode plan
./wm3d.sh cache site.env      # action/task/VGGT cache 与 dataset seal
```

正式训练读取
`WORK_ROOT/datasets/wm3d_v7_public6106h_v1/receipts/dataset_seal.json`。README 中的小时数用于
容量规划；source scan 和 dataset seal 记录的实测帧数与时长才是训练统计。

处理后的目录结构：

```text
WORK_ROOT/
├── raw/
│   ├── snapshots/          # 固定 revision 的公开原始快照
│   └── materialized/       # 解包与转换结果
├── datasets/
│   ├── wm3d_v7_public6106h_v1/
│   │   ├── control/        # contract、episode plan、action 统计
│   │   ├── shards/         # 训练 cache
│   │   └── receipts/       # worker 与 dataset seal
│   └── wm3d_v7_encoder_assets_v1/
├── release/                # source/code/environment/config receipts
├── runs/                   # checkpoint 与验证输出
├── logs/
└── envs/
```

## 三、先跑小数据全流程

正式下载前，先在一台双卡机器验证整个软件链：

```bash
./wm3d.sh smoke /shared/wm3d-smoke
```

smoke 会下载固定 revision 的 ALOHA 小样本（约 91 MB），生成真实 VGGT cache，在
GPU0–1 上执行一步 5B preset 的 FSDP2 训练、validation、原子 checkpoint 和 eval。环境和
数据准备可在 GPU 忙时完成；每个 GPU 阶段启动 worker 前都会重新检查 GPU 空闲、ECC 和
磁盘空间，不会占用已有计算进程。

smoke 直接继承标准 `HF_ENDPOINT`。无法直连 huggingface.co 时，从同一个 work-root 原地
重试，已完成的环境和下载缓存会复用：

```bash
HF_ENDPOINT=https://hf-mirror.com \
  ./wm3d.sh smoke /shared/wm3d-smoke
```

成功标志是 `/shared/wm3d-smoke/smoke_report.json` 中 `pass=true`。报告包含原始数据
revision、dataset seal、精确参数量、checkpoint 哈希以及 RGB/depth/point/action 指标。
全过程会追加写入 `/shared/wm3d-smoke/logs/smoke.log`；最近一次尝试的阶段、退出码和日志
位置会原子写入 `/shared/wm3d-smoke/smoke_status.json`。失败时先查看这两个文件，再用同一
命令和 work-root 重试，不要删除半成品。

如果已有固定 revision 的 VGGT 模型快照：

```bash
VGGT_MODEL_SNAPSHOT=/abs/hf-cache/models--facebook--VGGT-1B/snapshots/860abec7937da0a4c03c41d3c269c366e82abdf9 \
  ./wm3d.sh smoke /shared/wm3d-smoke
```

## 四、训练与评测

先查看即将提交的命令：

```bash
./wm3d.sh plan site.env
```

提交 canary，门禁通过后提交正式训练：

```bash
./wm3d.sh train site.env
./wm3d.sh status site.env
```

从环境、数据一路执行到训练：

```bash
./wm3d.sh all site.env
```

### 训练正确性评测

`train` 会先跑同构 canary，并且只有 canary checkpoint 的 eval 通过后才提交正式训练。正式
训练期间可对任意带 `COMMITTED.json` 的完整编号 checkpoint 单独评测：

```bash
./wm3d.sh eval site.env \
  /shared/wm3d/runs/RUN/checkpoints/step_XXXXXXXX
```

评测会使用 checkpoint 所属 run 的物化配置和固定 validation sampler，在与训练相同的
FSDP2/HSDP 拓扑上读取 validation split。输出位于：

```text
RELEASE_ROOT/eval/RUN_step_XXXXXXXX/
├── report.json
└── rgb_target_top_prediction_bottom.png
```

`report.json` 同时检查：

| 范围 | 指标或门禁 |
|---|---|
| checkpoint | 编号、commit、run lineage、配置、代码、环境、dataset seal 全部绑定 |
| 总体 | 所有 native loss 和直接指标均为有限数，监督值数量非零 |
| RGB | MAE/MSE/RMSE、PSNR、预测方差，并输出 target/prediction 对照图 |
| 3D | depth、point、geometry confidence 的 MAE/MSE/RMSE |
| camera | 9D camera pose 的 MAE/MSE/RMSE |
| action | grouped-action 的 MAE/MSE/RMSE、NLL/velocity loss |
| contact | 概率误差、accuracy、contact loss |

要判断后续 checkpoint 是否发生明显回退，应先用相同 `site.env`、validation 数据和
`EVAL_STEPS` 分别生成两个报告，再运行：

```bash
./wm3d.sh compare-eval site.env \
  /shared/wm3d/release/eval/RUN_step_00001000/report.json \
  /shared/wm3d/release/eval/RUN_step_00005000/report.json \
  /shared/wm3d/release/eval/compare_00001000_to_00005000.json
```

比较器要求 dataset seal、training contract、代码、参数量、run lineage、world size 和评测步数
完全相同；默认不允许 lower-is-better 指标相对退化超过 20%、RGB PSNR 下降超过 1 dB、contact
accuracy 下降超过 0.05。阈值用于拦截明显回退，不代表论文质量标准。

eval PASS 的含义是训练链、checkpoint、监督覆盖和 native 输出工作正常；它不等于机器人闭环
成功率，也不能单独证明模型优于其他版本。闭环 action 成功率仍需在具体机器人 benchmark 上
使用同一完整 checkpoint 评测。

训练恢复只选择同时含 `COMMITTED.json` 的最高编号 checkpoint，不读取 `latest`。

## 五、5B H200 训练预设

相关配置：

- `configs/train/5b_h200.yaml`：正式训练；
- `configs/train/5b_h200_canary.yaml`：1,000-step 同构 canary；
- `configs/train/5b_smoke.yaml`：双卡基础设施验证；
- `configs/cluster/h200.env.example`：站点与 Slurm 参数。

5B 是配置层的模型规模，通用类名、数据加载器、FSDP2、checkpoint 和 eval 代码不依赖这个
名字。修改 YAML 中的宽度、层数与 `model_budget` 即可定义其他规模。

### 时空与主干参数

| 参数 | 值 | 原因 |
|---|---:|---|
| `T` | 24 | 5 Hz 下使用 4.8 秒历史，比 T16 提供更完整的接触前后状态 |
| `P` | 144 | 每帧 12×12 显式 3D 空间格，提高小物体和边界细节 |
| `K` | 16 | 显式预测未来 3.2 秒，不局限于 K8 |
| 外部 token `D` | 2048 | 保持 VGGT/cache 接口，避免无信息增益的存储膨胀 |
| state hidden/layers | 2560 / 32 | 主要容量投入世界状态动力学 |
| action hidden/layers | 2048 / 24 | 独立建模多 embodiment、高频 grouped action |
| state↔action bridge | 10 层 | 深层交换动作条件与世界状态，不把模型退化为 action head |

长度 `(T+K)×P = 5,760`，因此主干使用帧内空间 attention、同 patch 因果时间 attention和
低频 memory，而不是对全部 token 做 dense attention。

### 参数组成

精确总参数为 **4,956,589,929**：

| 模块 | 参数量 | 占比 |
|---|---:|---:|
| world state trunk | 3,250,831,360 | 65.5860% |
| grouped-action trunk | 1,195,474,944 | 24.1189% |
| state↔action bridge | 424,719,360 | 8.5688% |
| 接口、memory、位置与 query | 55,055,872 | 1.1108% |
| 三视角 fuser | 16,783,360 | 0.3386% |
| RGB head | 9,357,443 | 0.1888% |
| depth/point/camera/confidence head | 3,959,840 | 0.0799% |
| action distribution head | 407,750 | 0.0082% |

state trunk 持有显式未来 3D 世界，约占 65.6%；action trunk 约占 24.1%。RGB、depth、point、
camera 共享未来 world state，action 使用独立时序主干，并通过 bridge 与 world state 互相条件化。

复核配置对应的精确参数量：

```bash
./wm3d.sh params site.env configs/train/5b_h200.yaml
```

## 六、代码结构

```text
wm3d/                 # 模型、数据契约、训练与 checkpoint 实现
configs/
├── cluster/          # 站点配置示例
├── data/             # 数据清单、source lock 与字段映射
├── smoke/            # ALOHA 小样本契约
└── train/            # 不同规模的训练预设
environments/         # Python 3.10 venv 与固定依赖
scripts/
├── pipeline.py       # 下载、处理、缓存、训练与评测编排
├── data/             # 数据下载、转换、统计、缓存与封存
├── assets/           # VGGT 等离线编码资产
├── cluster/          # 配置物化、代码封存、启动前检查
├── slurm/            # Slurm 作业入口
├── smoke/            # 公开小样本全流程验证
└── tools/            # 参数统计等独立工具
tests/                # 数据、模型、恢复、发布与 smoke 契约测试
wm3d.sh               # 用户入口
```

## 七、常见问题

- `site.env 缺少 ...`：补齐站点配置中的必填项；
- `HF_TOKEN_FILE 权限`：执行 `chmod 600 TOKEN_FILE`；
- gated dataset 返回 401/403：先在 Hugging Face 网页接受对应许可；
- revision 不一致：检查 `release/raw_sources.lock.yaml`，不要手改已发布 lock；
- 已有目录但缺少 receipt：保留目录，使用同一阶段命令继续；
- `No space`：扩容共享存储后重跑同一阶段；
- Slurm 状态：`./wm3d.sh status site.env`。
