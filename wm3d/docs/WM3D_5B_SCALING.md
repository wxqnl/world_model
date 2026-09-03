# WM3D V8 5B 训练交付

这份文档只保留当前主推流程：检查本地模型和数据，准备训练 metadata，完成 64 卡
1K canary，再 fresh 启动正式 600K。数据从魔搭、Hugging Face 或内部存储取得均可，
训练只检查本地文件内容与 WM3D 数据合同，不依赖下载来源。

## 1. 当前训练合同

- GitHub 分支：`v8`
- 模型配置：`configs/model/native_5b_v8_action_owned_transport.yaml`
- objective：`configs/objective/stage0_v8_action_owned_transport.yaml`
- 参数量：`5,087,822,644`
- 数据模式：`direct_raw`
- 集群：8 个节点，每节点 8 张 H200
- micro batch：每卡 4
- global batch：256
- canary：1,000 steps
- 正式训练：600,000 steps，从正式 run 的 step 0 开始

V8 使用 group-preserving 的 factual action conditioner 学习未来状态。数据张量先按每个
source 封存的 offset/scale 精确还原为共同物理单位，再进入 block-0 direct projection；
factual command 与同 mask 的物理 no-op 使用同一编码相减，因此 zero update 严格为 0。
factual P144 是唯一运动所有者，并预测 renderer 实际应用的 backward flow；最后观测 RGB
只能通过该 flow 传输到未来帧，motion head 不会衰减它。高频 refiner 只补充有界高通细节，
不能改变低频位置或运动。future candidate 不进入 policy/action-free trunk。

当前正式配置不使用 absolute future P256、P256 自回归、teacher forcing、unwarped copy 或
full-frequency redraw。RAFT 只在训练时生成 backward-flow 监督 target，不属于模型或 serving。

## 2. 只填写模型和数据目录

先拉取代码：

```bash
cd /data
git clone --branch v8 --single-branch https://github.com/wxqnl/world_model.git
cd /data/world_model/wm3d
```

只需要确认两个路径：

```bash
MODEL_ROOT=/共享目录/模型
DATA_ROOT=/共享目录/已下载数据

./run_wm3d.sh 5b configure "$MODEL_ROOT" "$DATA_ROOT"
SITE=/data/wm3d/control/5b_canary1k.env
```

`MODEL_ROOT` 应同时包含 VGGT 源码、VGGT-1B 权重和
Qwen3-VL-Embedding-2B 权重。目录层级可以不同，脚本会自动寻找并写入 site。

`DATA_ROOT` 可以直接指向魔搭下载目录，也可以指向已经整理好的 WM3D 数据目录。
脚本会向下寻找 AgiBotWorld2026 的三个正式子集，并同时支持压缩包和已经解压的
LeRobot 目录。它只做有界抽查，不复制或改写原始数据。

检查结果中的 `data_state` 有四种：

- `RAW_COMPATIBLE`：下载内容和目录可识别，但还缺已审计 data profile，不能启动训练。
- `PROFILE_PATH_MISMATCH`：control 包来自另一台机器，里面的原始数据或 manifest 路径在
  当前机器不存在，需要先按当前共享目录重新物化 control 包。
- `PROFILE_READY`：data profile 已就绪，可以生成 task bank 和训练 metadata。
- `TRAIN_METADATA_READY`：数据 metadata 已封存，可以生成 runtime。

`ready_to_train` 只有在模型、data profile、训练 metadata、环境和 runtime 全部就绪时才会
变为 `true`。如果结果是 `RAW_COMPATIBLE`，把项目随数据交付的 `control` 目录放到
`/data/wm3d/control`，然后重新执行同一条 `configure` 命令；不需要改下载脚本或数据适配器。

## 3. 创建环境和训练 metadata

```bash
cd /data/world_model/wm3d
SITE=/data/wm3d/control/5b_canary1k.env

./run_wm3d.sh 5b env "$SITE"
./run_wm3d.sh 5b doctor "$SITE"
./run_wm3d.sh 5b task-bank "$SITE"
./run_wm3d.sh 5b cache-plan "$SITE"
./run_wm3d.sh 5b streaming-prepare "$SITE"
./run_wm3d.sh 5b runtime "$SITE"
./run_wm3d.sh 5b doctor "$SITE"
./run_wm3d.sh 5b status "$SITE"
```

已有文件会按原合同验证并复用。使用本地现成数据时不需要 Hugging Face token；只有主动
执行 `lock` 或 `download` 时才需要配置 token。

最后重新检查一次：

```bash
./run_wm3d.sh 5b configure "$MODEL_ROOT" "$DATA_ROOT"
```

确认输出包含：

```text
"input_check": "PASS"
"data_state": "TRAIN_METADATA_READY"
"ready_to_train": true
```

## 4. 运行 64 卡 1K canary

申请 8 个完整 H200 节点并进入 Slurm allocation。启动脚本会自动取得 master 节点，
不需要手动填写地址、端口、节点编号或 torchrun 参数。

```bash
cd /data/world_model/wm3d
SITE=/data/wm3d/control/5b_canary1k.env

./run_wm3d.sh 5b slurm "$SITE" preflight
./run_wm3d.sh 5b slurm "$SITE" train 100

./run_wm3d.sh 5b slurm "$SITE" preflight
./run_wm3d.sh 5b slurm "$SITE" resume 100 500

./run_wm3d.sh 5b slurm "$SITE" preflight
./run_wm3d.sh 5b slurm "$SITE" resume 500 1000

./run_wm3d.sh 5b slurm "$SITE" preflight
./run_wm3d.sh 5b slurm "$SITE" eval 1000
./run_wm3d.sh 5b verify "$SITE" 1000
```

canary 通过需要满足：

- 64 个 rank、H200、NVLink、IB 和 ECC preflight 正常；
- loss、grad norm 和梯度所有权指标全部有限；
- factual decoder、RGB decoder、action head 和 policy 都有非零梯度；
- 同一物理 action 在不同 source normalization 下产生一致的 direct conditioning；
- correct action 在多个 source/seed 上优于 physical no-op 与 shuffle；
- future action 对 policy/action-free 输出的差异严格为 0；
- step100、step500、step1000 checkpoint 完整，resume、eval 和 verify 均通过。

任一项失败时保留日志和 checkpoint，不启动正式训练。

## 5. Fresh 启动正式 600K

canary 通过后复制 site，只把 preset 改为正式训练：

```bash
cd /data/world_model/wm3d
CANARY_SITE=/data/wm3d/control/5b_canary1k.env
SITE=/data/wm3d/control/5b_formal600k.env

install -m 600 "$CANARY_SITE" "$SITE"
sed -i 's/^WM3D_5B_PRESET=canary1k$/WM3D_5B_PRESET=formal600k/' "$SITE"

./run_wm3d.sh 5b doctor "$SITE"
./run_wm3d.sh 5b runtime "$SITE"
```

进入正式训练的 8 节点 Slurm allocation 后执行：

```bash
./run_wm3d.sh 5b slurm "$SITE" preflight
./run_wm3d.sh 5b slurm "$SITE" train
```

正式训练不得从 canary、旧 V8、旧 5B 或 1B checkpoint 初始化。中断后只从本次正式 run
最新的完整 checkpoint 恢复：

```bash
./run_wm3d.sh 5b slurm "$SITE" preflight
./run_wm3d.sh 5b slurm "$SITE" resume 完整checkpoint的step号
```

## 6. 需要立即停止的情况

- rank 丢失，或出现 NCCL、IB、NVLink、ECC 错误；
- loss、梯度或模型输出出现 NaN/Inf；
- 数据、normalization、runtime 或 environment receipt 不一致；
- checkpoint 不完整；
- future action 泄漏到 policy/action-free trunk；
- RGB 单色塌缩，或 factual action/RGB 路径没有梯度。

不要通过改模型、改 loss、减少 RGB horizon 或跳过 preflight 来绕过错误。
