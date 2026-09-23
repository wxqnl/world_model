# WM3D Native 预训练

## 目标与初始化边界

训练自己的 native 3D world model，改善 RGB 预测并支持 action 下游任务。
WSA 是代码和训练流程的参考，不是成品权重来源。保留通用 Wan2.2 视频/VAE 先验，
action、3D 专家的通用 Transformer 层沿用 WSA 官方的 Wan 插值与 alpha 缩放初始化；
机器人输入输出头、native 3D 状态和 WM3D 接口仍是新参数。已获用户确认。
这不是继承 WSA 的机器人能力，也不等于从随机参数训练整个视频生成器；
不能期待一百步就获得 WSA 成品模型的能力。

| 组件 | 采用的实现 | 权重与监督 |
| --- | --- | --- |
| 视频 | 官方 Wan VideoDiT、VAE、flow scheduler 与 sampler | Wan 视频先验；骨干低学习率，VAE 冻结 |
| Action | 官方 ActionDiT 与 flow objective | Wan 派生骨干、新输入输出头；真实 grouped action、timestamp、mask |
| 3D | MoT 内部的逐视角 K8×P64 trajectory queries | Wan 派生骨干、新状态接口；冻结 VGGT 的所有未来时刻 target |
| Robot/task | WM3D 已有数据桥与 conditioning | 当前状态、历史动作、任务和同一归一化 profile |

Wan 派生资产由官方 `preprocess_expert_backbones.py` 在 CPU 上生成，WM3D 只提供薄入口，
不改插值方法。当前双机资产已放在 `/data/Minko/models/WM3D-Wan-Backbones`。
新环境首次准备时运行一次；命令拒绝覆盖既有文件：

```bash
FOUNDATION_PYTHON_BIN=/data/Minko/.venvs/wm3d-foundation-v10-final/bin/python \
  ./run_wm3d.sh native-backbones --model configs/model/native_wan.yaml
```

没有新增 RGB decoder、第二套 action head 或 posthoc dynamics loss。输出是视频、
生成 action 与显式未来 native 3D tokens；不能将 token 预测直接宣称为高精度度量点云。
候选 action 反事实排序暂不作为本阶段的主要验收条件。

## 3D 如何参与 RGB 与 action

当前多视角图像经冻结 VGGT 得到每视角 P64 状态。它被投影到同视角 K8 个未来 query，
与可学习的位置/query 参数一起进入 Future3D。每个未来 query 保留视角、时间和空间槽位，
在官方 MoT 的每层与 RGB/action 专家交互，最后一对一读出 2048D native 3D token。
不把最终单帧 token 复制成 K 帧，也不在 RGB/action 算完后另加一个预测头冒充核心状态。

沿用 `WSALarge` 的注意力边界，不使用 `WSALargeJoint`：当前视频只读当前视频，
未来视频读取 3D；3D/action 读取观测和彼此，但不能读取加噪的未来 RGB。
训练中的未来 RGB/action 是 flow 目标对应的加噪变量；推理从噪声生成，不能把干净未来
标签塞进 context。未来 VGGT target 只在 loss 回调读取。VGGT 各时间行独立编码，
避免把未来信息混入当前观测特征。

3D 监督沿用 upstream 的 cosine + layer-normalized MSE，覆盖所有有效 K8 时刻，
权重为 0.1。视频/action 各为 1.0。逐 horizon 的 delta 方向、幅度和 copy-last error
只用于诊断，不再叠加一组运动 loss。缺失视角、动作和维度保留原 mask。

## 数据

复用 `/data/Minko/wm3d_v7_world_1000h_data_20260909/train_episode_allowlist.json`，
对应约 999.999 小时、253,207 个唯一训练 episode、4,922,451 个训练窗口、18 个 source。
小时数按 episode 时长，不重复累计重叠窗口。保留所有 OXE，以及原清单选定的 RoboCasa
episode；按既有选中窗口数比例采样，不重新做 source 均衡或运动筛选。
127,991 个 validation 窗口不受训练 allowlist 过滤。

数据 profile 和 metadata 位于 `/data/Minko/wm3d_full_robot_data_20260904`。
新增的 `episode_selection.py` 只应用既定清单，核对 source、episode 和窗口覆盖；
source adapter、物理语义、归一化和原始文件都不改。
WSA 使用的其他高质量数据尚未加入；先核实下载位置、许可证和 adapter，不能把未读过的
数据写成已参与训练。

## 双机试训配置

使用 `configs/model/native_wan.yaml`、`configs/objective/native_joint.yaml` 和
`configs/runtime/h100_16_native_pilot.yaml`。仅在 43、44 现有任务全部结束后启动；不使用 41。

- 16 GPU，micro-batch 1、累积 1、global batch 16，100 个更新步。
- FP32 主参数及优化器状态，BF16 计算；Wan 视频骨干学习率为新模块的 0.1 倍。
- FSDP2 节点内 8 卡分片、节点间复制；保留官方 activation checkpoint。
- 官方 MoT 会直接调用 block 子函数，因此本路线先用 root-only FSDP，不能机械套逐层 wrap。
- 每 20 步验证；只在第 100 步保存完整 checkpoint。预计单 checkpoint 约 76 GiB，开训前
  同时预留 checkpoint 和临时写入空间。试训数据流不变，后续按实测显存提高 micro-batch。
- `h100_32_native.yaml` 仅为 4×8 卡拓扑配置起点；预算、batch 和质量尚未验证，不能自动长训。

保持现有 `run_wm3d.sh` 入口、trainer 和 DCP checkpoint，不新增平行训练器。
下面命令在 43 上 `ssh New-H100-3` 后、`cd /data/Minko` 执行；44 也经 43 进入。
代码根为 `/data/Minko/wm3d_native_pretrain_1000h_20260923`，44 使用同路径同步副本。

```bash
cd /data/Minko/wm3d_native_pretrain_1000h_20260923/wm3d
export PYTHON_BIN=/data/Minko/.venvs/wm3d-foundation-v10-final/bin/python
export FOUNDATION_PYTHON_BIN="$PYTHON_BIN"
./run_wm3d.sh check

RUN=/data/Minko/wm3d_native_1000h_pilot_2node16_20260923
./run_wm3d.sh runtime \
  --model configs/model/native_wan.yaml \
  --data /data/Minko/wm3d_full_robot_data_20260904/data_profile.yaml \
  --runtime configs/runtime/h100_16_native_pilot.yaml \
  --objective configs/objective/native_joint.yaml \
  --data-mode direct_raw \
  --streaming-metadata-seal /data/Minko/wm3d_full_robot_data_20260904/streaming_metadata_seal.json \
  --environment-lock /data/Minko/.venvs/wm3d-foundation-v10-final/environment_receipt.json \
  --run-name native_1000h_pilot --run-lineage native_1000h \
  --output-root "$RUN" --output "$RUN/runtime.yaml"
```

先确认双方模型、metadata、原始视频、text context 和共享输出目录可访问。双机以相同
runtime 执行以下命令，`NODE_RANK` 在 43 为 0、44 为 1；端口须先确认未被其他任务使用：

```bash
export NCCL_IB_DISABLE=1 NCCL_NVLS_ENABLE=0
export NCCL_SOCKET_IFNAME=bond0.1411 NCCL_SOCKET_FAMILY=AF_INET
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 MALLOC_ARENA_MAX=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export WM3D_VGGT_SOURCE_ROOT=/data/Minko/wm3d_vggt_source_20260822
export WM3D_VGGT_MODEL_SNAPSHOT=/data/Minko/models/VGGT-1B
export WM3D_DIRECT_PREPARED_ROW_CACHE_BYTES_PER_RANK=8589934592
export WM3D_DIRECT_PREFETCH_WORKERS=2 WM3D_DIRECT_ASYNC_PIPELINE=0
./run_wm3d.sh foundation-preflight \
  --nnodes=2 --nproc_per_node=8 --node_rank="$NODE_RANK" \
  --master_addr=172.27.0.6 --master_port=29983 -- --runtime "$RUN/runtime.yaml"

# preflight 成功且两个节点仍空闲后，以同一组参数调用 foundation-train。
./run_wm3d.sh foundation-train \
  --nnodes=2 --nproc_per_node=8 --node_rank="$NODE_RANK" \
  --master_addr=172.27.0.6 --master_port=29983 -- --runtime "$RUN/runtime.yaml"
```

## 下载期间先用可读数据测试

用户于 2026-09-23 要求不等待全部下载，先测已有数据。44 已空闲，43 仍有其他任务，
因此先用 `h100_8_native_qualification.yaml` 在 44 做完整模型 8 卡、20 步资格测试。
这不是缩小模型的模拟实验：模型、RGB 分辨率、K8、三视角、action 与 3D loss 均不变；
仅 batch 为 8，资格调度为 20 步，第 20 步保存，以验证真实前后向与 checkpoint。

资格数据位于 `/data/Minko/wm3d_native_available_qualification_data_20260923`。
使用三个已存在的 RoboCasa source 及已恢复的 OXE mutex/hydra：每源取既定 1000h
清单中前 8 个训练 episode、原 split 的前 2 个验证 episode，不按动作或预测质量挑选。
继承原数据的物理标签、camera 绑定、数值归一化和文本向量，不改正式数据文件。
小集只判断工程是否跑通，不证明 1000h 收敛或跨 source 能力。

首次资格根 `/data/Minko/wm3d_native_available_q20_8gpu_20260923` 已保留：8 卡预检通过，
首个真实前后向完成，但指标累积时发现 upstream 部分日志为 Python float，训练器要求
Tensor，尚未完成优化器更新。现仅把日志统一转成 Tensor，total 的反传图保持原样。
修复后 fresh 重测使用 44 的
`/data/Minko/wm3d_native_available_q20_8gpu_20260923_attempt2`，不复用失败运行。

attempt2 已完成真实 20 次优化器更新并正常退出。step20 的 COMMITTED、
8 个 distcp 分片和 8 个 rank state 已齐全，checkpoint 约 76 GiB。
MoT、native 3D、action time、proprio 和 robot conditioning 均有有限非零梯度。
step20 total 1.711914、video 0.676270、action 0.803711、3D 2.324219、grad norm 8.433914。
各步使用不同样本及扩散噪声，不能据此推断质量趋势。
排除冷启动与验证后的 step12–20 平均 5.39 秒/步，范围 4.71–6.20 秒；
单机 global batch 8、短程小集吞吐约 1.48 samples/s，不能外推为全量双机吞吐。

Demo 导出也已补齐 native FSDP 路径：按保存时的 8/16 卡拓扑严格读回 checkpoint，
各 rank 显式 unshard 后独立采样并分目录输出，不伪装单卡同拓扑；生成输入去除未来
3D 标签，逐 horizon 3D 指标仅在预测完成后计算。独立读回/生成服务为
`wm3d-native-available-q20-eval-20260923`，日志在 attempt2 根的 `logs/eval.log`；
读回和生成已正常退出，8 份 receipt 均生成，future action 标签对独立 policy 的差异严格为 0。
`foundation-demo` 接受 `--` 分隔的 torchrun 拓扑与应用参数。

生成质量尚不达标：8 个样本未来帧间变化保持率为 13.1%–37.3%，方向余弦约 -0.004–0.048；
RGB L1 均差于 copy-last，action MAE 均高于归一化零值基线。不能把画面有变化当作正确运动。
对 Hydra 和 RoboCasa 各一组完整 9 帧做时序检查，保留场景结构，但生成机械臂运动仍弱，
后期细节模糊；GIF 时间 0–2.8 秒仅为播放时间，非机器人控制时钟。没有单色或逐帧完全相同。
这轮仅训练 20 步，说明工程链可运行，不能确认质量，也不能单凭它归因为模型结构错误。

接续双机资格根为 `/data/Minko/wm3d_native_available_q100_2node16_20260923`。
43/44 均已空闲，使用上述相同资格数据与模型，fresh Wan 初始化；16 卡 micro1/acc1/global16，
100 步，不 resume 8 卡 checkpoint。配置从 `h100_16_native_pilot.yaml` 仅改资格身份、
移除全量 allowlist 字段（资格 profile 已只含原清单选中的 episode），未改正式清单或 source 比例。
运行配置 `/data/Minko/wm3d_native_pretrain_checks_20260923/h100_16_available_qualification.yaml`。
服务 `wm3d-native-available-q100-n43-20260923` 与 `wm3d-native-available-q100-n44-20260923`；
日志在运行根 `logs/preflight_n43.log`、`preflight_n44.log`、`train_n43.log`、`train_n44.log`。
只有 NFS、16 rank preflight 均通过才启动训练；100 步后检查 checkpoint 与同 seed 无目标生成。
完整数据恢复和必要验收通过后再正式训练。首轮正式更新预算已向用户确认，尚待回复；
不能把 32 卡规划配置的 100000 步自动当成已批准预算。

## 验收与当前证据

截至 2026-09-23，全仓检查为 652 passed、1 skipped。真实 Wan 权重的 CPU 建模已完成，
未加载 WSA checkpoint；总参数约 7.247B、可训练参数约 6.542B，包含约 5.000B 视频骨干、
1.021B action、0.433B 3D。参数注册已去重，避免重复保存同一 MoT 权重。
真实 8 卡训练、checkpoint 保存、同拓扑独立读回及无标签生成均已运行；双机同步与生成质量仍待验收。
不能表述为 WM3D 已收敛或优于 WSA。

试训先确认真实数据能读、各训练 owner 梯度有限非零、优化器确实更新参数、训练/推理边界
一致和 checkpoint 可恢复，再记录稳态秒/步、samples/s、峰值显存、解码与 VGGT 耗时。
用固定真实样本比较无目标输入的 joint RGB、独立 action 和每个未来时刻的 3D：

- RGB：target/prediction/copy-last 按同 K8 时序并排，区分运动方向、幅度和静态清晰度。
- 3D：逐 horizon 方向、幅度、error 相对 copy-last；测试 observation/3D 消融是否影响生成。
- Action：物理反归一化误差、方向和幅值、gripper、task/vision/state/history 消融；不能用
  world token gain 代替 policy 能力，也不能把训练标签参与的 flow loss 当作生成质量。
- 保存完整 step100 后做独立进程读回。有限的 100 步只资格验证和测速，不证明最终 VLA
  或约 1000h 的充分收敛；闭环任务留给后续阶段，不能只用 LIBERO 下结论。

原始数据恢复正在进行：43/44/42 未找到可直接替换的完整 OXE/DROID 文件，RoboCasa
原文件仍在。已按原清单逐文件核对公开资产，14 个 OXE source 对应 main，DROID 对应
v3.0；恢复约 412 GiB 的原文件，不更换索引、adapter 或采样比例，不使用 41。
下载服务 `wm3d-native-raw-recovery-20260923` 在 43 运行，数据写入 44 的
`/data/Minko/raw_recovery_20260923`，43 通过 NFS 访问同一路径。
进度、日志与完成记录在 `/data/Minko/wm3d_native_pretrain_checks_20260923`，分别为
`raw_recovery_progress.json`、`raw_recovery.log`、`RAW_RECOVERY_COMPLETE.json`。
双方 `/shared/Minko/datasets` 已只读映射恢复目录，以支持已恢复 source 的独立资格小集。
目前 14 个 OXE source 已恢复，DROID 仍在下载；目录存在不代表全量就绪。
只有全部完成且文件可读后，才能使用完整 1000h 清单。开训前核对 NFS 和磁盘空间。

43 的其他任务和 44 单机资格读回均已结束。每次启动前仍需查占用，
不得停止或共享其他项目 GPU。两机空闲且数据恢复完成后，
继续执行真实 16 卡 preflight 和 100 步试训；不得绕过缺失 source 开训。

## 来源与实验材料说明

本计划使用 academic-research-suite 的 experiment-agent / code experiment plan 工作流，
形成于 2026-09-23。材料来源：本机实际代码/资产、既定 1000h receipt、官方仓库；
验证状态为代码、8 卡训练/保存/独立读回/推理工程检查已验；双机试训与质量验收继续。本文不证明生成质量。

- [WSA 官方实现](https://github.com/zaleni/WSA)：MoT、WSALarge mask、flow training 和 sampler。
- [WSA 预训练入口](https://github.com/zaleni/WSA/blob/main/launch/wsa_large_pretrain.sh)：
  pipeline 参考。已采用其窄专家 Wan 派生初始化，不使用 WSA 成品 checkpoint。
- [Wan2.2 官方实现](https://github.com/Wan-Video/Wan2.2)：通用视频/VAE 先验。
