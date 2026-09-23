> 本文记录使用 WSA 成品 checkpoint 的历史适配路线，保留用于已有实验复现。
> 当前重启不使用这些权重，入口已改为
> [WM3D Native 预训练](WM3D_NATIVE_PRETRAIN.md)。不要把本文的 foundation-fetch
> 或旧 adapter profile 用于新的 1000h 训练。

# WM3D V10：WSA-Large foundation 路线

## 结论与边界

V10 是一条独立的 foundation 实验路线，不是对当前 V8 长训目录的原地修改。它以官方
WSA-Large 为模型基座，保留 Wan2.2 视频专家、ActionDiT、Future3D 与 Mixture-of-Transformers
联合训练主干，再接入 WM3D 已封存的数据、机器人物理语义和原生 3D 监督。

WSA-Large 的基座是 Wan2.2-TI2V-5B，不使用 Qwen3-VL。Qwen3-VL 属于 WSA-Base，V10
因此不加载 Qwen、不创建 Qwen LoRA，也不要求 Qwen image processor。任务文本沿用
WSA-Large 的 UMT5 上下文，并预先写入只读 context store，训练时不常驻文本编码器。

V10 在独立分支中增加了受限的 native-3D causal expert。它不替换 WSA 的视频、action 或
Future3D 主链，而是在同一组 Future3D token 上显式接入当前原生 3D 状态，并解码 K8 原生 3D
轨迹。替代正式训练前仍需做短程受控训练，同时检查视频、物理 action、K8 轨迹和最终 Future3D；
不能直接套用 WSA 官方 benchmark 数字。

## 当前训练优先级

当前先训练基础 WM 与可执行 action：从观测、任务、状态和历史，学会生成未来视频、
原生 3D 状态和动作。保留 WSA 现有 video/action/3D 联合目标；不删除真实动作标签，
不把 action loss 关闭，也不为候选动作控制新增损失、decoder 或训练分支。

当前验收分别检查视频运动与清晰度、Future3D/K8 预测、独立 policy 的物理误差及
task/vision/state/history 依赖。主 demo 不提供真实未来 action，而由官方联合 sampler
自己生成视频与动作。联合样本中的 action 和部署用独立 policy 都要报告，不能互相替代。

给定候选动作的反事实预测、动作可控性与候选排序留待后续专项训练/验收。修好的接口保留，
但 real/zero 视频响应不再是当前阶段的主要门槛。训练稳定、没有目标泄漏仍是必要条件；
q20 只证明接口可运行，学习趋势应在短适配中验证，不要求 20 个样本就达到最终质量。

## 冻结架构

```text
三视角视频 + 任务/机器人条件 ───────────→ Wan2.2 video expert ─────┐
noisy action + 任务/状态/历史 ──────────→ ActionDiT policy ────────┤
最后观测帧 → frozen VGGT P144 ─zero-gate→ Future3D P144 expert ───┤→ WSA MoT
                                                       │          │
真实 future action/time/mask ───────────────────────────┴→ causal K8×P64 rollout
抽样未来帧 → frozen VGGT P64 ─────────────────────────────→ rollout delta loss
最后未来帧 → frozen VGGT P144 ────────────────────────────→ 官方 P1296 final 3D loss
```

这里的 `P` 是每个视角的空间 token 数，不是 RGB 分辨率。当前采用分层预算：

- `P144`（12×12）用于 WSA 已有的联合 Future3D expert 和当前状态；
- `K8×P64`（每时刻 8×8）用于完整时间轨迹，保持 causal rollout 计算可控；
- `P1296`（36×36）保留 WSA 官方最终高分辨率 3D 输出。

不把 30 层 MoT 扩成 `K8×P144`。那会显著增加序列注意力、activation 和训练时间，而当前状态
与最终细节已经分别由 P144/P1296 覆盖。只有短训证明 P64 是明确空间瓶颈后，才单独做 P64/P144
rollout A/B，不能在首个版本里盲目扩大。

必须保持以下合同：

- 视频沿用 WSA-Large 的三视角画布和完整 K8 未来监督；不引入 copy-last、P256 teacher forcing
  或另一套 RGB decoder。
- 最后观测三视角只作为 `P144` current native state，经 zero-gated、逐视角 projection 注入
  Future3D query。gate 初始严格为零，只保证这个新增支路初始不扰动 Future3D 输入；动作和
  状态边界投影因尺寸变化而重置，整模不能宣称与原 WSA 等价。缺失视角始终 mask 为零，
  视角之间不池化。
- native-3D rollout 是单一轻量 causal recurrence。它读取 shared Future3D、当前 P144 状态和
  同一个 source-native future action/time/mask；每个 world interval 内的 fine command 不插值，
  按真实时间编码后聚合。rollout 只能写 world 分支，不能写回 ActionDiT policy。
- rollout 预测相对最后观测状态的 P64 delta，而不是复制绝对未来外观；一次只抽样一个未来
  horizon 作 teacher target，horizon 由封存 sample index 确定，跨样本均匀覆盖 K8。当前、抽样
  horizon 和最终帧在一次 frozen VGGT 调用中编码，future target 从不进入模型输入或 serving。
- action 保留 WM3D 的 grouped ABI。维度容量为 3 组 × 14 维；时间轴保留 K 个原始 world
  interval，每个 interval 使用同一个槽位容量 S。只裁掉所有 interval 都未使用的尾部容量，
  不压缩中间空槽。例如 8 个区间共 31 条有效命令时，可以保留 32 个槽位，缺失的那个槽位
  由 mask 排除。有效动作值、真实时间戳、所属区间和数据权重不变，不插值、不补造命令。
  WSA action loss 忽略空槽；K8 rollout 按每个区间的有效命令数聚合，空槽不稀释动作更新。
  独立 action 与联合 video/action 采样都约束填充位：每轮输入为原始噪声乘当前 sigma，
  对应已知零标签的 flow 轨迹；填充位速度为同一原始噪声，最终输出为零。有效坐标仍由
  官方模型和 sampler 更新。不额外抽噪声，不改变训练 loss，也不改真实动作标签。
- action 时间等于 world interval 起点加该 interval 内的真实 fine-command offset。训练和推理
  使用同一个 `ActionTimeConditioner`；时间空槽的编码输出与梯度严格为零。
- 当前状态为 3 组 × 10 维；offset、scale、semantic、embodiment 与 action history 通过
  zero-gated 条件融合进入 WSA 文本上下文，不传 source ID。
- WSA-Large 发布权重的 action 输入/输出和 proprio 边界原为 24 维。V10 改为 42/30 维，因此
  只重置这些尺寸不兼容的边界投影；视频、ActionDiT、Future3D 的 MoT 主干权重继续加载。
- Future3D 仍是 WSA-Large 主干；最终监督教师从官方 DA3 改成当前项目冻结的 VGGT token。
  WSA 最终 3D 与 K8 rollout 共用原来的总 3D 权重，分别为 0.05/0.05，避免额外 objective
  压过 video/action。rollout 只用一个按 target-delta 尺度归一化的 robust regression；方向余弦
  与幅值比仅作验收指标，不作为第二项竞争 loss，也不增加第二套 3D owner。
- 物理反归一化严格使用同一 source/group profile：`physical = normalized * scale + offset`，
  padded 维保持零。

## 动作填充位修正与验收状态

WSA-Large 的类默认配置对填充维度计算 action denoising loss，但官方 RoboTwin 微调脚本也
启用了 padding mask；不能把这个风险说成 WM3D 独有。WM3D grouped-action 只监督有效维度，
此前 sampler 却自由更新全部 42 维。未受监督的填充输出会作为下一轮 ActionDiT 输入，
训练中的填充输入则是 `sigma * initial_noise`，形成训练/采样失配。已有 18-source 审计样本
每条动作只有 2–12 个有效维度。修正后，填充坐标沿已知加噪轨迹更新，最终严格为零。

现在的修正只约束上述已知零值的填充坐标。Wan、ActionDiT、MoT、VGGT/K8 结构和权重不变，
不新增参数或 loss；训练前向及数据比例不变。无填充的请求逐元素保持上游行为。旧 step500
可直接做只读修正前后对照，无须先重新训练 500 步。

2026-09-08 已完成同 checkpoint 的修正前后 A/B，以及修正后 step100/500 的同样本、
同噪声对比。独立 action 的完整序列归一化 MAE：

| 样本集合 | step100 | step500 | neutral |
| --- | ---: | ---: | ---: |
| 16 个留出窗口 | 0.85739 | 0.69970 | 0.58052 |
| 18 sources、36 个已见训练诊断窗口 | 0.90235 | 0.72442 | 0.45917 |

step500 相对 step100 分别有 14/16、27/36 样本改善，18 个 source 均值中 16 个改善。
H1 归一化 MAE 从 0.65232/0.85656 降至 0.35804/0.47377；对应 neutral 为
0.51113/0.47223。H1 先在单样本内平均有效 group，再跨样本平均，不能与混池口径混用。
这些指标证明继续适配能学到动作；完整序列仍未稳定超过 neutral，不能声称 VLA 已验收。

视觉消融在已见诊断集的误差增量为 -0.00747，尚未证明正确视觉依赖。history/state 依赖较强。
policy future-label 差异、反归一化差异和 padding 轨迹误差均为零。16 个 validation 窗口仍只
来自同一 source 的一个 episode，不代表全量数据泛化；真实夹爪切换样本覆盖也不足。

三组视频的帧间幅值保持率从 47.0%/46.4%/23.1% 升至 73.2%/51.0%/75.5%，但其中两组的
RGB L1 变差，后段仍有形变和轨迹偏离。运动增加不等于预测正确。完整结果位于
`wm3d_v10_joint_adaptation_8h100_g64_20260907_r4/eval/checkpoint_trend_20260908`。

### 当前阶段：先训练适配模块

2026-09-08 用户确认先冻结 WSA 主体，训练 WM3D 新接口。当前模型 profile 显式设为
`finetune_mode: adapters`；旧 runtime 未写此项时仍按原来的 full 模式解释，不能混用恢复。

冻结 released video、ActionDiT、Future3D 的 Transformer 主体及 Wan VAE。训练已有
robot/context fusion、action-time、native-3D state/rollout，以及 ActionDiT action_encoder/head、
proprio_encoder、Future3D output_decoder 的最终 output_norm/output_proj。最后一项负责
DA3 到 VGGT 输出表示的适配，不解冻 3D attention/FFN。没有新增模块或 loss。

冻结仅排除参数梯度与 optimizer 状态，不包裹整个 WSA 为 no_grad：输入适配模块仍须
通过冻结主干反传。保留 activation checkpointing、BF16、完整视频和真实 action/3D 目标。
冻结不保证视频输出完全不变，因为新条件仍会改变主干输入；必须做真实样本质量对照。

首轮短对照复用已验证的 18-source 小集：182 episodes、11,682 train windows、
292 validation windows。保留该小集原有 split、封存归一化和 source 权重，不修改全量数据。
这个小集的归一化统计不等同全量闭包，故比较冻结/全量策略时必须使用同一小集。
小集只做接口适配与方向验证，不能据此宣布对全部任务泛化，也不预设必须用全部数据长训。
后续数据规模依据独立 episode/多 source 验证是否饱和决定，不只按训练 loss 决定。

之前 full/global128 从 WSA fresh 的任务已在 step20 保存失败后退出，没有完整 COMMITTED；
最后一次训练 loss/gradient 有限，错误出现在 DCP/collective 保存阶段。现场保留，禁止
把它当作成功的 200-step 验收或恢复来源。本阶段资格必须包含真实保存与 exact resume，
不能只以八步容量 benchmark 代替。新的本地错误日志保留原始 DCP 异常。

### 适配训练的数据与计算缓存

当前短资格之后的适配训练使用既有完整索引，限定 1000 个 optimizer 更新。
source 权重、split、动作标签、时间戳和归一化沿用全量数据，不反复训练 182-episode 小集。
仍从官方 WSA-Large/Wan 权重开始，只训练上述适配接口；不恢复资格或旧 WM3D checkpoint。
用户最新要求 gradient accumulation 为 1，micro-batch 按真实八卡容量与吞吐测试选取。
1000 步使用 100 步 warmup；start/peak/min LR 保持 1e-5/6e-5/6e-6，不按 batch 倍增 LR。

可选环境变量 `WM3D_FOUNDATION_VGGT_CACHE_BYTES_PER_RANK` 给每个 rank 的冻结 VGGT
CPU 特征缓存设定字节上限，默认 0 关闭。缓存键包含封存 episode/物理帧身份和完整视角 mask；
保留同时刻多视角编码，不缓存跨时间 teacher，不将 future target 传给模型或 serving。
缓存输出保持原 FP32 token 精度；同批重复物理帧只编码一次。没有可靠帧身份的调用走原路径。
该缓存仅复用固定预处理、固定冻结 encoder 的结果，不能用于训练中的随机图像增强。
启用前须通过真实多视角 VGGT 输入/target 张量对照及实际 optimizer 前后向。

raw-frame prepared cache 与此特征缓存相互独立，均属于 host 内存预算。
冻结主干仍需前向和输入梯度反传，activation checkpointing 也仍会重算；因此不能按
可训练参数减少比例推算加速。吞吐报告按 samples/s，同时标注 batch、冷缓存与数据来源。

### 历史记录：全量训练准备

后续沿用已准备的 `wm3d_full_robot_data_20260904`：18 sources、670,760 episodes、
12,765,292 总窗口；不改 source 权重、原始任务、归一化或 split。原 Wan 缓存只包含
79 条适配集任务，现通过冻结 UMT5 补齐全部 56,400 条不同文本。context builder 支持
`--task-bank-index` 和单机 torchrun 分片，输出同一只读格式。

2026-09-08 用户改为从新阶段 step0 开始：只加载官方 WSA-Large/Wan 基座，不读取旧
WM3D step500，也不恢复旧 optimizer、schedule 或 sampler。预训练基座不做随机重置。
之前 global64/model-only warmstart 的配置仅保留记录，不能用作当前启动入口。

单机 8 H100，micro2 × accumulation8 = global128，验证 micro-batch 保持 1。
上限 50K，先 `--stop-after-step 200` 保存并停止。50K 对应 640 万次样本使用，
不保证遍历约 1,251 万个训练窗口。学习率 start1e-5、peak6e-5、weight decay0.01、
warmup2000，不因 batch 扩大而自动翻倍。

主体约 98.7% 参数为 BF16，约 1.3% 新增模块参数保留 FP32。此次效率改动只在
foundation 输入适配器中跳过未被使用的冻结 VGGT depth/point/camera 输出头；
current、抽样 horizon、final 的 native-3D token 监督不变。八卡、16 个真实 DROID
三视角样本的六项输入/target 张量前后逐元素相同，624 项测试通过、4 项跳过。

完整模型、真实数据、AdamW 与八次累积的容量测试：原 micro1 平均 12.64 秒/步、
5.06 样本/秒；优化后 micro2 独立冷启动平均 16.90 秒/步、7.57 样本/秒，
峰值分配 75.56 GiB。每次配置完成 8 个更新，统计排除前两步。micro4 会 OOM，不采用。
顺序测试的 micro2 曾达 11.62 样本/秒，但有数据缓存复用，不能把它当作冷启动或
全量训练速度。短测的八步权重全部丢弃，不用于正式初始化。

启动使用 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 减少显存碎片，
这是 [PyTorch 2.7 提供的分配器选项](https://docs.pytorch.org/docs/2.7/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf)，
不更改模型或目标。micro2 的显存余量仍有限，正式训练需继续观察不同 source 和保存窗口。

200 步后必须复查多个 source 的留出视频和独立 policy，尤其视觉依赖及视频后段形变，
再决定是否在本根继续。当前没有“全部问题已解决”的结论。

## 目录与模型资产

建议固定四个目录：

```bash
export WSA_SOURCE=/data/Minko/third_party/WSA
export WSA_MODEL=/data/Minko/models/WSA-Large
export WAN_MODEL=/data/Minko/models/Wan2.2-TI2V-5B
export WAN_CONTEXT=/data/Minko/wm3d_foundation_context
```

先获取官方源码：

```bash
git clone https://github.com/zaleni/WSA.git "$WSA_SOURCE"
```

## 从空白环境开始

基础 WM3D 与 WSA 增量依赖分开封存：

```bash
./run_wm3d.sh env
WM3D_WSA_SOURCE_ROOT="$WSA_SOURCE" ./run_wm3d.sh foundation-env
```

`foundation-env` 会复用已验证的 CUDA/PyTorch 基础环境，只在 `.venv-foundation` 安装固定的
WSA 增量依赖，并应用仓库内的动作时间空槽修正：原上游只在 action MSE 中屏蔽维度，
此修正让时间空槽也不参与 loss 分子、分母和分维统计，不改变已有损失系数。
模型加载会拒绝未应用该修正的上游源码。环境 receipt 首次创建后只验证、不覆盖；
环境准备必须在训练启动前执行。完成后可运行：

```bash
FOUNDATION_PYTHON_BIN=.venv-foundation/bin/python \
  ./run_wm3d.sh check
```

再下载并检查 WSA-Large 与 Wan2.2 权重：

```bash
./run_wm3d.sh foundation-fetch \
  --output "$WSA_MODEL" \
  --wan-output "$WAN_MODEL"
```

网络受限时可额外传入兼容 Hugging Face Hub 的 `--endpoint`。已有本地资产时不需要重复下载；
配置文件中的四个路径必须指向实际目录。

## 构建任务 context store

context store 只需为当前 sealed data closure 创建一次：

```bash
./run_wm3d.sh foundation-context \
  --metadata-root /path/to/streaming_metadata \
  --wan-model-root "$WAN_MODEL" \
  --upstream-source-root "$WSA_SOURCE" \
  --output "$WAN_CONTEXT"
```

脚本要求每个任务文本非空并唯一封存；未知任务不会在训练时临时编码或静默回退。

## 生成 runtime

先确认 `configs/model/foundation_large.yaml` 中的四个资产路径与本机一致，然后用现有
direct-raw closure 物化 runtime：

```bash
export DATA_ROOT=/path/to/sealed_data
export RUN_ROOT=/path/to/wm3d_v10_run

PYTHON_BIN=.venv-foundation/bin/python ./run_wm3d.sh runtime \
  --model configs/model/foundation_large.yaml \
  --data "$DATA_ROOT/data_profile.yaml" \
  --runtime configs/runtime/h100_8_foundation.yaml \
  --objective configs/objective/foundation_joint.yaml \
  --data-mode direct_raw \
  --streaming-metadata-seal "$DATA_ROOT/streaming_metadata_seal.json" \
  --environment-lock .venv-foundation/environment_receipt.json \
  --run-name wm3d-v10-foundation \
  --run-lineage fresh-foundation \
  --output-root "$RUN_ROOT" \
  --output "$RUN_ROOT/runtime.yaml"
```

正式物化要求 Git 工作树干净。不要使用 `--allow-dirty-smoke` 生成正式 runtime。

## 单机 8×H100 预检与训练

当前 profile 使用 DDP、BF16、micro batch 1、累积 8 次、global batch 64。先跑 preflight：

```bash
./run_wm3d.sh foundation-preflight \
  --standalone --nproc_per_node=8 -- \
  --runtime "$RUN_ROOT/runtime.yaml"
```

通过后才启动短程资格训练：

```bash
./run_wm3d.sh foundation-train \
  --standalone --nproc_per_node=8 -- \
  --runtime "$RUN_ROOT/runtime.yaml" \
  --stop-after-step 20
```

不要直接从 20 步跳到正式长训。至少继续验证 step100/500 的视频运动、清晰度、ActionDiT
物理误差、task/state/history 消融与 Future3D 指标；只有三条能力同时向好才扩大训练。

## Checkpoint 视频与 Action 对照

默认用完整 checkpoint 导出观测条件下的联合视频/动作，以及独立 policy 预测：

```bash
FOUNDATION_PYTHON_BIN=.venv-foundation/bin/python \
  ./run_wm3d.sh foundation-demo \
  --runtime "$RUN_ROOT/runtime.yaml" \
  --checkpoint "$RUN_ROOT/checkpoints/step_00000020" \
  --output "$RUN_ROOT/eval/joint_demo" \
  --seed 7340 --num-inference-steps 20
```

输出目录必须尚不存在。入口使用指定 Python 的 torch.distributed.run，不依赖旁边是否有
torchrun 可执行文件。

默认 mode 为 joint。导出器先移除未来 action 和 3D 标签，视频输入仅保留观测帧；
predict_joint 调用官方 WSA 联合 sampler，从随机噪声同时生成未来视频和 action。
面板标为 joint prediction。报告联合视频相对 copy-last 的误差、逐帧变化，以及联合 action
和独立 policy 各自相对常量零基线的误差、逐 query 误差和分维度物理误差。
分维度物理误差必须按 source 语义解释，不能混合米、弧度和夹爪单位当成单一质量分数。

只有传入 --mode candidate 才运行 real/zero 反事实诊断；该模式不是当前默认验收。
predict_action 只根据观测、任务、状态和历史预测动作。predict_world 则根据给定候选动作
预测视频：每个采样步始终把 clean、masked 候选送入 ActionDiT，并固定 action noise timestep
为 0；仅未知的未来视频走原 WSA sampler。已知动作不能再随视频加噪。WSA 的训练本来就
独立采样两个噪声级别；这种条件推理与 [UWM 的 forward-dynamics 定义](https://arxiv.org/html/2504.02792v1#S3.SS2)
一致。这里的噪声 timestep 不是机器人真实 action timestamp。该分支的 action 输出不作为
policy 预测返回。
候选对照调用结束后会恢复原采样函数。两个模式都检查真实未来 action 标签不会改变独立
policy 输出；joint 模式不需要该标签作为输入，也不会启用 candidate clamp。

发布的 WSA-Large 关闭了 VideoDiT 的 action_conditioned 选项。旧导出器仅向那个未启用的
入口传 real/zero action，实际上生成了两次相同的联合样本；旧对照不能说明模型的动作因果
能力。修正后的真实响应仍需用 checkpoint 测量，不能凭接口修复认定方向或任务能力已经正确。

面板中的 normalized zero action 指归一化坐标零值，即物理坐标中的 source offset，
不等于机械臂不动。时序指标仅统计 h1 至 h8 的有效视角，不把输入首帧的 VAE 重建差计作
预测运动。所有 GIF 共用同一调色板，各帧另存为无损 PNG；定量分析应使用 PNG 或
receipt 中量化前的指标。GIF 播放速度只用于展示，真实动作时间仍由原数据时间戳决定。

此前 q20 的逐步加噪候选对照（seed 7340、sample 139、20 次采样、两个有效视角）：
RGB 响应 RMS 为 0.00217，未启用入口的旧调用为 0；独立 policy 前后差异为 0。未来帧运动
幅度保持率约 50.1%，方向一致性约 0.017。这些是旧采样方式的结果，不是 clean-action
修复后的成绩，也不足以证明动作方向或任务能力。此次仅修正推理条件，不改参数或训练
objective。全仓检查 610 项通过、4 项按环境条件跳过。

q20 的 global batch 为 1、累积为 1，合计仅 20 次训练样本曝光。实际加载审计确认视频、
ActionDiT、Future3D 主干没有遗漏；尺寸不兼容而重置的张量仅有 action_encoder.weight、
action head.weight/bias 和 proprio_encoder.weight。它们与新增机器人条件模块需要在当前
物理语义和归一化坐标上适配。不能把 q20 等同于已适配好的 WSA 官方模型；也不能仅凭训练步数
就排除残余实现问题。是否学到动作跟随，应分别检查条件视频和独立 policy 的真实样本结果。

同 q20、同 sample 139/seed 7340 的 clean-action 对照已完成：视频 factual/zero 响应 RMS
从 0.00217 升至 0.02303，但方向一致性从 0.0171 变为 0.0121，帧间变化保持率从 50.1%
变为 48.0%。factual L1 为 0.119612，zero L1 为 0.119583，没有正确动作优于 zero 的证据。
独立 policy MAE 为 0.82532，而该样本的归一化常量零基线为 0.63345；candidate 对 policy
差异仍为 0。这里只确认已知动作条件不再被采样器人为加噪削弱，不能宣称动作预测质量已经解决。
单个高运动样本不代表全部 source；正式长训之前仍需短适配与多样本动作/视频趋势验收。

## 已完成的实现证据

- 现有 sealed direct-raw 数据继续使用 18 个 source 的原始物理值和时间戳。8 卡连续训练
  暴露了旧 bridge 的边界错误：全局删除空槽后，31 条命令无法分成 K8 区间；即使命令数碰巧
  可整除，也可能改变原区间归属。bridge 现保留区间内 mask，新增回归覆盖不等命令数、
  样本间不同空槽位置、物理值/时间戳不变，以及 rollout 对额外空槽的输出和梯度不变。
- 聚焦测试与全仓检查均通过：606 项通过、1 项因可选依赖跳过。覆盖 P144/P64 形状、逐视角
  隔离、zero-gate 恒等、future target 隔离、action 时间分组和 rollout loss 反向传播。
- 移除 Qwen 后，带 native-3D causal expert 的模型共有 7,254,131,334 个参数，其中
  6,549,442,666 个可训练；新增 current-state conditioner 与 K8 rollout 合计 5,202,304 个
  参数，不到整模的 0.1%。
- 真实 WSA-Large/Wan2.2/VGGT 权重已在 8×H100 上完成资源预检，并用 18 个真实 source、
  11,682 个训练窗口和 292 个验证窗口完成一次真实 AdamW 更新。该步 `video=0.20438`、
  `action=0.88790`、官方 final-3D 加权项 `0.15110`、K8 rollout 加权项 `0.01726`、
  `total=1.26063`、`grad_norm=3.53067`，全部有限。
- current、确定性抽样 horizon 与 final 三个时间点在一次 frozen VGGT 调用中编码；实测每个
  sample 为 3 帧、约 0.77 秒。输出在退出 inference context 后转成普通无梯度 tensor，已验证
  可安全参与可微 loss。
- rollout objective 只保留归一化 delta Smooth-L1。零初始化时方向余弦不进入 loss，避免零向量
  附近梯度奇异；direction 与 amplitude 仅作为 detached 验收指标。真实更新的整体梯度范数稳定，
  不再出现该竞争项导致的异常放大。
- 单卡真实优化器峰值分配显存约 51.35 GiB；真实一阶 ActionDiT 推理峰值约 13.90 GiB。
- 真实推理输出保持 8×42 normalized action，并还原为 8×3×14 physical action；mask 与原生
  时间戳保持一致。
- 干净 foundation 环境已经完成首次创建和第二次只读复验。

这些证据证明“实现接通、可训练、可推理”，不证明收敛后的任务能力。完整 8 卡吞吐和稳定性
仍需在不占用现有训练资源的窗口做连续短训；至少完成 step20/100 的 video、ActionDiT、K8
rollout 与 final-3D 联合趋势验收后，才能决定是否扩大 P64 或进入长训。

## 参考基座

- WSA 官方仓库：https://github.com/zaleni/WSA
- WSA-Large 权重：https://huggingface.co/zaleni/WSA-Large
- Wan2.2-TI2V-5B：https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B
- WSA 论文：https://arxiv.org/abs/2607.03941

代码仓库为 Apache-2.0；发布或分发 WSA-Large 权重前，还应单独确认模型仓库的权重许可元数据。
