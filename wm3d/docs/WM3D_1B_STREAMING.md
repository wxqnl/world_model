# WM3D 1B：全 OXE 按需缓存训练

本文给第一次接触具身世界模型的操作者使用。目标是在一台 8×H100 80GB 服务器上，使用
WM3D 1B、全部官方 LeRobot OXE 数据和现有 DROID、Bridge、RoboCasa 数据，先完成 1K
验证训练，再从零训练 100K steps。

唯一入口是 `run_wm3d.sh`。不要直接运行内部 Python trainer，也不要把旧 checkpoint、旧 PCA
cache 或未审计的数据 profile 混入新 run。

## 1. 这次训练包含什么

正式组合共 60 个 source：

- DROID；
- Bridge V2；
- RoboCasa365 atomic、composite、mg；
- 官方 LeRobot Open X-Embodiment collection。collection 中的 DROID 与现有 DROID 去重，
  因此新增 55 个 OXE source。

默认不加入 AgiBotWorld 2026 和 AgiBotWorld Beta。OXE 新 source 各自权重为 1；已有五个
source 的权重保持 `14/6/4/8/8`，不会让某一个 OXE 数据集仅凭体量压倒其它来源。

模型使用 `configs/model/native_1b_dual_path.yaml`：融合 P64 geometry 继续承担 3D、动作、
状态和动力学；逐视角 P256 appearance latent 专门保留 RGB 高频信息。RGB decoder 同时读取
预测 appearance 与 3D future state，输出 8 个 256×256 future RGB 帧。目标使用
`configs/objective/stage0_native_dual_path.yaml`。默认 site 文件已填好这些配置，操作者不需要
手动替换。数据访问使用 `streaming_raw`。

### PCA 与磁盘

这条新路径不使用旧的 2048→384 PCA token cache。冻结 VGGT 第一次访问 episode 时生成完整
2048D token，只做逐向量 int8 量化并放入有上限的 LRU；RGB 监督来自原始视频并保存为 JPEG
pack。PCA 过去压缩的是视觉/几何表征，不是 RGB 图像本身。旧图像模糊还与 decoder 容量、
预测帧数和损失有关，这些已在当前 1B profile 中修正。

同一次 VGGT forward 会同时生成 P64 geometry 和 P256 appearance，不会把 encoder 计算做
两遍。旧 LRU 只有 geometry token；启动 dual-path run 时必须使用新的独立 LRU 根目录。

`streaming_raw` 省掉完整视觉 cache，但仍要求冻结 revision 的原始视频可从本地文件系统读取。
默认 OXE 组合原始数据约 15–20TB；准备阶段还要给下载临时文件留空间。因此不要把 7TB 单盘
误认为足够。建议：

- 原始数据盘或共享挂载：至少 25TB 可用，稳妥配置 30TB；
- 训练输出与 checkpoint：至少 500GB 可用；
- 8 rank LRU：默认每 rank 16GiB，共 128GiB，可按命中率提高；
- metadata、task bank、日志：预留 500GB–1TB。

若这些条件不满足，`doctor`/resource preflight 必须失败；不要悄悄删 source 或改成未封存的
网络随机读取。

## 2. 安装与 site 文件

```bash
git clone --branch v8 --single-branch https://github.com/wxqnl/world_model.git
cd world_model/wm3d
./run_wm3d.sh env
source .venv/bin/activate
PYTHON_BIN=.venv/bin/python ./run_wm3d.sh check
```

先创建 1K canary site 文件：

```bash
SITE=/data/wm3d_1b_oxe/control/1b_canary1k.env
./run_wm3d.sh 1b init canary1k "$SITE"
chmod 600 "$SITE"
vim "$SITE"
```

至少检查以下字段：

- `WORK_ROOT`、`RAW_ROOT` 和 `STREAMING_LRU_ROOT` 指向容量足够的真实存储；
- `HF_TOKEN_FILE` 存在、权限为 `0600`，并已接受所有上游许可；
- `WM3D_VGGT_SOURCE_ROOT`、VGGT 权重和 Qwen embedding 权重存在；
- node43 使用 `NNODES=1`、`GPUS_PER_NODE=8`、`MASTER_ADDR=127.0.0.1`；
- `WM3D_DATA_MODE=streaming_raw`；
- `INCLUDE_AGIBOT_2026=NO`、`INCLUDE_AGIBOT_BETA=NO`。

打印计划并检查模型、拓扑和最终 step：

```bash
./run_wm3d.sh 1b plan "$SITE"
./run_wm3d.sh 1b doctor "$SITE"
```

## 3. 下载与数据合同

生成全 OXE 模板，然后锁定远端 commit：

```bash
./run_wm3d.sh 1b data-template "$SITE"

# 人工接受所有上游许可后，编辑 site 文件：ACCEPT_DATA_LICENSES=YES
./run_wm3d.sh 1b lock "$SITE"
./run_wm3d.sh 1b download "$SITE"
```

`data-template` 会生成 P64/256px 表征，不会生成 5B 的 P144/384px cache，也不会加入
AgiBot。生成后应看到 60 个训练 source：已有 5 个，加 OXE 新增 55 个。

下载完成不等于可以训练。每个 source 还必须依次完成：

```text
schema-audit → adapter-audit → inventory → data-profile
```

具体参数和 receipt 字段见 [从零数据、训练与评测](WM3D_FROM_ZERO.md)。OXE 自动生成的
adapter 只把上游 controller 向量保存在 `source_controller_native` frame；负责人仍须核对
action/state 维度、时间戳、相机映射、夹爪语义和许可。最终 `DATA_PROFILE` 必须是普通文件，
并把 60 个 source manifest、adapter SHA、raw root 和 split 全部封存。模板中的
`__MATERIALIZE_REQUIRED__` 任一残留都会被拒绝。

验收数据 profile：

```bash
./run_wm3d.sh 1b doctor "$SITE"
```

输出应包含：

```text
model=native_1b expected_parameters=1,319,203,443
world_size=8 total_steps=1000
data_profile=public_robot_1b_oxe sources=60
```

## 4. 轻量 metadata 与按需 LRU

```bash
./run_wm3d.sh 1b task-bank "$SITE"
./run_wm3d.sh 1b cache-plan "$SITE"
./run_wm3d.sh 1b streaming-prepare "$SITE"
./run_wm3d.sh 1b runtime "$SITE"
./run_wm3d.sh 1b status "$SITE"
```

这里不运行 `cache-worker`、`cache-seal`、`window` 或 `normalization`。`streaming-prepare`
一次生成 episode metadata、窗口索引和 grouped normalization；视觉 token、depth、point 和
RGB pack 在训练首次访问 episode 时才进入 rank-local LRU。

开始训练前检查：

- metadata seal 中 `model_profile` 是 `native_1b`，representation 是 P64、2048D、256px；
- window 的 train/val/test 都非空；
- normalization 所有要求的 lane coverage 大于 0、数值有限；
- LRU 总预算不会把数据盘挤到资源门禁以下。

## 5. 1K 验证训练

先运行资源和数据 preflight：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  ./run_wm3d.sh 1b preflight "$SITE"
```

先跑 100 steps，确认 checkpoint 后用独立进程恢复到 500，再到 1000：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  ./run_wm3d.sh 1b train "$SITE" 100

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  ./run_wm3d.sh 1b resume "$SITE" 100 500

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  ./run_wm3d.sh 1b resume "$SITE" 500 1000

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  ./run_wm3d.sh 1b eval "$SITE" 1000

./run_wm3d.sh 1b verify "$SITE" 1000
```

每个训练命令必须自然退出后再启动下一个。恢复只接受完整编号目录中的
`COMMITTED.json`，不使用 `latest`。

### 怎么判断运行正确

`status` 和训练日志应显示：

- 8 个 rank 都存活；GPU 长期高利用率，显存分布接近；ECC 为 0；
- `total`、`token_mse`、RGB、depth、point、pose、action loss 全部有限；
- gradient ownership gate 通过，所有 required owner 的 nonzero gradient 大于 0；
- 冷启动时 `generated_episodes` 增长，之后 `cache_hits` 持续增长；
- `resident_bytes` 不超过每 rank 16GiB；若每步都 eviction，扩大 LRU；
- step 100/500/1000 都有完整 DCP，独立恢复的 sampler step 连续；
- eval receipt 中所有 required coverage 大于 0，`all_metrics_finite=true`。

默认 runtime 使用训练 `micro_batch_size=8`、验证 `validation_micro_batch_size=2`、
`gradient_accumulation=1`、global batch 64。该配置已经在 8 张 H100 80GB 上完成从零训练、
step 20 独立恢复至 step 100、完整验证和 DCP 提交；训练计算阶段 GPU 接近满载，峰值显存约
55GB/卡。验证单独使用较小 micro batch，避免小型 canary 的单 source validation split 容量
不足；不得在已封存 runtime 上原地改值。

### RGB、depth 直观检查

offline eval receipt 是数值门禁；同时使用评测输出保存的固定样本对比 input/target/prediction：

- RGB 看边缘、纹理、运动物体和多步清晰度，不只看均值 loss；
- depth 看有效 mask 内的尺度、物体轮廓和时序一致性；
- point/camera pose 检查几何是否有限且跨视角一致；
- 同一固定样本在 step 100、500、1000 使用相同颜色映射和范围，禁止挑最好看的样本。

1K canary 证明 pipeline 和可学习性，不代表最终画质。

## 6. 正式 100K

canary 全部通过后，创建全新的 site 文件和 run ID：

```bash
FORMAL=/data/wm3d_1b_oxe/control/1b_formal100k.env
./run_wm3d.sh 1b init formal100k "$FORMAL"
chmod 600 "$FORMAL"
vim "$FORMAL"
```

让 `FORMAL` 复用已封存的 source lock、data profile、task bank 和 streaming metadata，但使用
新的 runtime/run output。然后：

```bash
./run_wm3d.sh 1b doctor "$FORMAL"
./run_wm3d.sh 1b runtime "$FORMAL"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./run_wm3d.sh 1b preflight "$FORMAL"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./run_wm3d.sh 1b train "$FORMAL"
```

建议在 1K、5K、10K 后分别用新进程 eval；正式选择 best checkpoint 时以固定 validation
schedule 的总 loss、RGB perceptual/gradient 指标、depth/point coverage 和固定 demo 为依据，
不要只选单个 RGB loss 最小的 step。100K 完成后：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./run_wm3d.sh 1b eval "$FORMAL" 100000
./run_wm3d.sh 1b verify "$FORMAL" 100000
```

若训练中断，先确认最后一个完整 `COMMITTED.json`，重新运行 fresh preflight，再使用
`./run_wm3d.sh 1b resume "$FORMAL" <step>`。不要修改原 runtime，也不要从不完整目录恢复。

## 7. 常见问题

- **磁盘仍然很大**：这是原始视频，不是 PCA/视觉 cache。要训练全部 OXE 就必须提供原始数据
  的稳定存储；缩小 LRU 只能减少派生 cache。
- **冷启动 GPU 利用率波动**：先看视频读取、`prepare_seconds` 和 `encode_seconds`。命中 LRU
  后仍低，才调整解码线程、存储带宽或 micro batch。
- **RGB 仍模糊**：先确认加载的是当前 native RGB v2 profile、8 个 future 帧和完整损失；再看
  固定 validation demo。不要把所有问题归因于 PCA。
- **某个 OXE source 读不通**：该 source 必须在 adapter/inventory 阶段 fail closed；不能静默
  从 60 个 source 中移除后继续宣称“全部 OXE”。
